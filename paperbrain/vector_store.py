"""DuckDB VSS/HNSW vector index for note retrieval.

The module never falls back to a Python full scan. If DuckDB or its VSS extension is
unavailable, callers must use lexical BM25 and report the degraded health state.
"""
import hashlib
import atexit
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import config

try:
    import duckdb  # type: ignore
except ImportError:  # optional locally; required by the production container
    duckdb = None

_LOCK = threading.RLock()
_CON = None
_CON_PATH = ""


def _table(model: str, dim: int) -> str:
    token = hashlib.sha256(f"{model}\0{dim}".encode("utf-8")).hexdigest()[:20]
    return f"note_vec_{token}"


def _registered_tables(con, model: Optional[str] = None) -> List[Tuple[str, int, str]]:
    """Return only registry rows whose table name matches our deterministic derivation."""
    if model is None:
        rows = con.execute("SELECT model, dim, table_name FROM vector_registry").fetchall()
    else:
        rows = con.execute(
            "SELECT model, dim, table_name FROM vector_registry WHERE model=?", [model]
        ).fetchall()
    valid = []
    for stored_model, dim, name in rows:
        try:
            dimension = int(dim)
        except (TypeError, ValueError):
            continue
        if str(name) == _table(str(stored_model), dimension):
            valid.append((str(stored_model), dimension, str(name)))
    return valid


def _connect():
    global _CON, _CON_PATH
    if duckdb is None:
        raise RuntimeError("缺 duckdb 依赖")
    target = config.vector_db()
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(target.resolve())
    if _CON is not None and _CON_PATH == resolved:
        return _CON
    if _CON is not None:
        try:
            _CON.close()
        except Exception:
            pass
        _CON = None
    # Load VSS before attaching the durable catalog. DuckDB documents this ordering as
    # required for recovery when a persistent HNSW database has a WAL after interruption.
    con = duckdb.connect(":memory:")
    try:
        con.execute("LOAD vss")
        con.execute("SET hnsw_enable_experimental_persistence=true")
        escaped = str(target.resolve()).replace("'", "''")
        con.execute(f"ATTACH '{escaped}' AS pb_vectors")
        con.execute("USE pb_vectors")
        con.execute("CREATE TABLE IF NOT EXISTS vector_registry("
                    "model VARCHAR, dim INTEGER, table_name VARCHAR, "
                    "PRIMARY KEY(model, dim))")
    except Exception:
        con.close()
        raise
    _CON, _CON_PATH = con, resolved
    return _CON


def _close() -> None:
    global _CON, _CON_PATH
    with _LOCK:
        if _CON is not None:
            try:
                _CON.close()
            except Exception:
                pass
        _CON, _CON_PATH = None, ""


atexit.register(_close)


def health() -> Dict:
    if duckdb is None:
        return {"ok": False, "backend": "duckdb-vss", "reason": "缺 duckdb 依赖"}
    try:
        with _LOCK:
            con = _connect()
            version = con.execute("SELECT extension_name, loaded FROM duckdb_extensions() "
                                  "WHERE extension_name='vss'").fetchone()
            registry = con.execute("SELECT COUNT(*) FROM vector_registry").fetchone()[0]
            indexes = con.execute(
                "SELECT COUNT(*) FROM duckdb_indexes() "
                "WHERE index_name LIKE 'hnsw_note_vec_%'").fetchone()[0]
        return {"ok": bool(version and version[1]), "backend": "duckdb-vss",
                "extension_loaded": bool(version and version[1]),
                "registry_entries": int(registry), "indexes": int(indexes)}
    except Exception as exc:
        return {"ok": False, "backend": "duckdb-vss", "reason": str(exc)[:180]}


def available() -> bool:
    if duckdb is None:
        return False
    try:
        with _LOCK:
            _connect()
        return True
    except Exception:
        return False


def count(model: str, dim: int) -> int:
    try:
        with _LOCK:
            con = _connect()
            name = _table(model, dim)
            registered = con.execute(
                "SELECT 1 FROM vector_registry WHERE model=? AND dim=?", [model, dim]).fetchone()
            value = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] if registered else 0
        return int(value)
    except Exception:
        return 0


def content_hashes(model: str) -> Dict[Tuple[int, int], str]:
    """Return {(owner_id, dim): content_hash} for cache/index reconciliation."""
    try:
        with _LOCK:
            con = _connect()
            out: Dict[Tuple[int, int], str] = {}
            for _, dim, name in _registered_tables(con, model):
                for owner_id, digest in con.execute(
                        f"SELECT owner_id, content_hash FROM {name}").fetchall():
                    out[(int(owner_id), dim)] = str(digest)
        return out
    except Exception:
        return {}


def upsert(model: str, rows: Iterable[Tuple[int, str, Sequence[float]]]) -> Dict:
    materialized = [(int(owner_id), str(content_hash), [float(x) for x in vector])
                    for owner_id, content_hash, vector in rows if vector]
    if not materialized:
        return {"ok": True, "indexed": 0}
    dims = {len(row[2]) for row in materialized}
    if len(dims) != 1 or 0 in dims:
        return {"ok": False, "indexed": 0, "reason": "向量维度不一致"}
    dim = dims.pop()
    name = _table(model, dim)
    idx = f"hnsw_{name}"
    try:
        with _LOCK:
            con = _connect()
            con.execute(f"CREATE TABLE IF NOT EXISTS {name}("
                        f"owner_id BIGINT PRIMARY KEY, content_hash VARCHAR, vec FLOAT[{dim}])")
            con.execute("INSERT OR REPLACE INTO vector_registry VALUES (?,?,?)",
                        [model, dim, name])
            # DuckDB explicitly warns against executemany for bulk inserts. Multiple LIST
            # parameters are unnested side-by-side in one vectorized statement.
            for start in range(0, len(materialized), 1000):
                chunk = materialized[start:start + 1000]
                con.execute(
                    f"INSERT OR REPLACE INTO {name} "
                    f"SELECT unnest(?), unnest(?), CAST(unnest(?) AS FLOAT[{dim}])",
                    [[row[0] for row in chunk], [row[1] for row in chunk],
                     [row[2] for row in chunk]])
            # DuckDB's conditional index DDL still performs the expensive build attempt before
            # checking existence. Inspect the catalog first.
            exists = con.execute(
                "SELECT 1 FROM duckdb_indexes() WHERE index_name=?", [idx]).fetchone()
            if not exists:
                con.execute(f"CREATE INDEX {idx} ON {name} "
                            "USING HNSW (vec) WITH (metric='cosine')")
        return {"ok": True, "indexed": len(materialized), "dim": dim, "table": name}
    except Exception as exc:
        return {"ok": False, "indexed": 0, "reason": str(exc)[:180]}


def search(model: str, query: Sequence[float], limit: int,
           allowed_ids: Optional[Sequence[int]] = None) -> List[int]:
    if not query or limit <= 0:
        return []
    dim = len(query)
    name = _table(model, dim)
    try:
        with _LOCK:
            con = _connect()
            registered = con.execute(
                "SELECT 1 FROM vector_registry WHERE model=? AND dim=?", [model, dim]).fetchone()
            if not registered:
                return []
            params: List[object] = []
            where = ""
            if allowed_ids is not None:
                ids = sorted({int(x) for x in allowed_ids})
                if not ids:
                    return []
                where = " WHERE owner_id IN (" + ",".join("?" for _ in ids) + ")"
                params.extend(ids)
            params.extend([[float(x) for x in query], int(limit)])
            rows = con.execute(
                f"SELECT owner_id FROM {name}{where} "
                f"ORDER BY array_cosine_distance(vec, CAST(? AS FLOAT[{dim}])) LIMIT ?",
                params).fetchall()
        return [int(row[0]) for row in rows]
    except Exception:
        return []


def delete_ids(owner_ids: Sequence[int]) -> int:
    ids = sorted({int(x) for x in owner_ids})
    if not ids or duckdb is None:
        return 0
    try:
        with _LOCK:
            con = _connect()
            names = [row[2] for row in _registered_tables(con)]
            total = 0
            placeholders = ",".join("?" for _ in ids)
            for name in names:
                found = con.execute(
                    f"SELECT COUNT(*) FROM {name} WHERE owner_id IN ({placeholders})", ids
                ).fetchone()[0]
                con.execute(f"DELETE FROM {name} WHERE owner_id IN ({placeholders})", ids)
                total += int(found)
        return int(total)
    except Exception:
        return 0


def clear() -> None:
    if duckdb is None:
        return
    try:
        with _LOCK:
            con = _connect()
            names = [row[2] for row in _registered_tables(con)]
            for name in names:
                con.execute(f"DROP TABLE IF EXISTS {name}")
            con.execute("DELETE FROM vector_registry")
            con.execute("CHECKPOINT")
    except Exception:
        pass


def self_test() -> Dict:
    """Run a non-destructive backend probe in a temporary database."""
    if duckdb is None:
        return {"ok": False, "backend": "duckdb-vss", "reason": "缺 duckdb 依赖"}
    old = os.environ.get("PAPERBRAIN_VECTOR_DB")
    try:
        with tempfile.TemporaryDirectory(prefix="pb_vss_probe_") as td:
            _close()
            os.environ["PAPERBRAIN_VECTOR_DB"] = str(Path(td, "probe.duckdb"))
            result = upsert("probe", [(1, "a", [1.0, 0.0, 0.0]),
                                      (2, "b", [0.0, 1.0, 0.0]),
                                      (3, "c", [0.9, 0.1, 0.0])])
            hits = search("probe", [1.0, 0.0, 0.0], 2)
            name = _table("probe", 3)
            plan = _connect().execute(
                f"EXPLAIN SELECT owner_id FROM {name} "
                "ORDER BY array_cosine_distance(vec, CAST(? AS FLOAT[3])) LIMIT 2",
                [[1.0, 0.0, 0.0]]).fetchone()[1]
            _close()
            persisted = search("probe", [1.0, 0.0, 0.0], 2)
            deleted = delete_ids([1])
            after = search("probe", [1.0, 0.0, 0.0], 2)
            clear()
            cleared = count("probe", 3) == 0 and health().get("indexes") == 0
            _close()
            ok = (result.get("ok") is True and hits == [1, 3] and persisted == [1, 3] and
                  "HNSW_INDEX_SCAN" in plan and deleted == 1 and 1 not in after and cleared)
            return {"ok": ok, "backend": "duckdb-vss", "hnsw_plan":
                    "HNSW_INDEX_SCAN" in plan, "persistence": persisted == [1, 3],
                    "delete_sync": deleted == 1 and 1 not in after,
                    "clear_sync": cleared}
    except Exception as exc:
        return {"ok": False, "backend": "duckdb-vss", "reason": str(exc)[:180]}
    finally:
        _close()
        if old is None:
            os.environ.pop("PAPERBRAIN_VECTOR_DB", None)
        else:
            os.environ["PAPERBRAIN_VECTOR_DB"] = old
