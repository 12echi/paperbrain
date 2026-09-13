"""学习记忆库 (v5.0): 把每次解读沉淀成可检索记忆, 供日后快速调用。

- 全局 SQLite (默认 out/memory.sqlite, 可用 PAPERBRAIN_MEMORY_DB 覆盖)。
- 每篇每类任务一条记录 (paper_id+task 幂等 REPLACE): 标题/摘要/要点/实体/关系。
- 搜索: 标题/摘要/实体/要点 的 LIKE 匹配; 也支持按 paper_id 精确取。
无外部依赖 (stdlib sqlite3)。
"""
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id TEXT, task TEXT, title TEXT,
  summary TEXT, key_points TEXT, entities TEXT, relations TEXT,
  created_at REAL,
  UNIQUE(paper_id, task)
);
CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id TEXT, task TEXT, kind TEXT,
  content TEXT, keywords TEXT, tags TEXT,
  context TEXT, links TEXT, source TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS concepts(
  paper_id TEXT, concept TEXT, weight REAL, PRIMARY KEY(paper_id, concept)
);
CREATE TABLE IF NOT EXISTS paper_edges(
  a TEXT, b TEXT, shared TEXT, weight REAL, PRIMARY KEY(a, b)
);
CREATE TABLE IF NOT EXISTS aliases(
  alias TEXT PRIMARY KEY, canonical TEXT
);
CREATE TABLE IF NOT EXISTS note_links(
  src INTEGER, dst INTEGER, rel TEXT, PRIMARY KEY(src, dst, rel)
);
CREATE TABLE IF NOT EXISTS vectors(
  owner_type TEXT, owner_id INTEGER, model TEXT, dim INTEGER, hash TEXT, vec BLOB,
  PRIMARY KEY(owner_type, owner_id, model)
);
CREATE TABLE IF NOT EXISTS preferences(
  term TEXT PRIMARY KEY, weight REAL, updated REAL
);
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY, value TEXT
);
"""


def _fts_ok(con) -> bool:
    try:
        con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(note_id UNINDEXED, body, tokenize='unicode61')")
        return True
    except Exception:
        return False


def _expand(text: str) -> str:
    """CJK 2-gram+unigram 与 ASCII 词混合展开, 供 FTS 分词 (免 ICU 的中文检索方案)。"""
    t = text or ""
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{1,}|\d+(?:\.\d+)?", t)
    for run in re.findall(r"[\u4e00-\u9fff]+", t):
        toks += [run[i:i + 2] for i in range(len(run) - 1)]
        toks += list(run)
    return " ".join(toks)


def _fts_body(content: str, keywords) -> str:
    try:
        kws = json.loads(keywords) if isinstance(keywords, str) else (keywords or [])
    except Exception:
        kws = []
    return _expand((content or "") + " " + " ".join(str(k) for k in kws))


def _match_expr(q: str, cap: int = 24) -> str:
    toks = [t for t in _expand(q).split() if len(t) >= 1][:cap]
    return " OR ".join('"%s"' % t.replace('"', "") for t in toks if t)


def _migrate(con):
    """老库平滑升级: 补列 + 建 FTS 并把已有笔记灌入。"""
    # 早期版本误建过普通表 notes_fts, 需拆除重建为 FTS 虚表
    row = con.execute("SELECT sql FROM sqlite_master WHERE name='notes_fts'").fetchone()
    if row and row[0] and "VIRTUAL TABLE" not in row[0].upper():
        try:
            con.execute("DROP TABLE notes_fts")
            con.commit()
        except Exception:
            pass
    cols = {r[1] for r in con.execute("PRAGMA table_info(notes)")}
    for col, ddl in (("anchor", "ALTER TABLE notes ADD COLUMN anchor TEXT"),
                     ("status", "ALTER TABLE notes ADD COLUMN status TEXT DEFAULT 'active'"),
                     ("confidence", "ALTER TABLE notes ADD COLUMN confidence REAL DEFAULT 1.0"),
                     # Generative Agents: 重要性 + 近因 (检索三因子) + 访问强化
                     ("importance", "ALTER TABLE notes ADD COLUMN importance REAL DEFAULT 0"),
                     ("last_access", "ALTER TABLE notes ADD COLUMN last_access REAL"),
                     ("access_count", "ALTER TABLE notes ADD COLUMN access_count INTEGER DEFAULT 0"),
                     ("valid_at", "ALTER TABLE notes ADD COLUMN valid_at REAL"),
                     ("invalid_at", "ALTER TABLE notes ADD COLUMN invalid_at REAL")):  # 双时间: 失效不删除
        if col not in cols:
            try:
                con.execute(ddl)
            except Exception:
                pass
    # 回填: 历史笔记补 importance/valid_at
    try:
        for nid, kind, content, created in con.execute(
                "SELECT id, kind, content, created_at FROM notes"
                " WHERE importance IS NULL OR importance=0 OR valid_at IS NULL"):
            con.execute("UPDATE notes SET importance=?, valid_at=COALESCE(valid_at,?) WHERE id=?",
                        (_importance(kind, content), created, nid))
    except Exception:
        pass
    # v5 provenance gate: early databases treated every note as active. Preserve those
    # rows, but quarantine them once when no current verification evidence is recorded.
    # Newly generated notes carry artifact-status/source-bound tags; a human decision can
    # explicitly promote a candidate later through decide_note().
    try:
        key = "legacy_unverified_quarantine_v1"
        done = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if not done:
            con.execute(
                "UPDATE notes SET status='candidate' "
                "WHERE COALESCE(status,'active')='active' AND NOT ("
                "COALESCE(tags,'') LIKE '%artifact-status:CLEAN%' OR "
                "COALESCE(tags,'') LIKE '%source-bound:true%' OR "
                "COALESCE(tags,'') LIKE '%human-reviewed:true%')")
            con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, "done"))
            con.commit()
    except Exception:
        pass
    if _fts_ok(con):
        n_fts = con.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0]
        n_notes = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        if n_fts < n_notes:  # 补灌历史笔记
            have = {r[0] for r in con.execute("SELECT note_id FROM notes_fts")}
            for r in con.execute("SELECT id, content, keywords FROM notes"):
                if r[0] not in have:
                    con.execute("INSERT INTO notes_fts(note_id, body) VALUES(?,?)",
                                (r[0], _fts_body(r[1], r[2])))
        con.commit()


def db_path() -> str:
    from . import config
    return config.memory_db()


_MIGRATED_PATHS: set = set()


def _conn():
    p = db_path()
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    # I/O 优化: 关掉 FULL 同步 (小事务不再每次 fsync); 不用 WAL (短连接每次 close 都会 checkpoint)
    try:
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA busy_timeout=3000")
    except Exception:
        pass
    if p not in _MIGRATED_PATHS:  # 建表+迁移每库每进程只做一次 (省 executescript/PRAGMA)
        con.executescript(_SCHEMA)
        try:
            _migrate(con)
        except Exception:
            pass
        _MIGRATED_PATHS.add(p)
    return con


def _clean(t: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]", "", t or "").strip()


# ------------------------------------------------- 检索三因子 (Generative Agents)
_HIGH_KINDS = ("一句话结论", "核心洞见", "claim", "质疑", "局限", "反思", "reflection")
_MID_KINDS = ("追问", "证据", "方法", "可复现", "假设", "Gap", "空白")


def _importance(kind: str, content: str = "") -> float:
    """0~1 重要性: 高信号笔记(结论/洞见/质疑/局限)高, 实体/列表低; 含数字略加分。"""
    k = kind or ""
    if any(h in k for h in _HIGH_KINDS):
        s = 0.9
    elif any(h in k for h in _MID_KINDS):
        s = 0.7
    elif ("实体" in k) or (k.strip().lower() in ("entity", "note")):
        s = 0.3
    else:
        s = 0.55
    if re.search(r"\d", content or ""):
        s = min(1.0, s + 0.05)
    return round(s, 3)


def _recency(last_access: Optional[float], created_at: Optional[float],
             half_life_h: float = 168.0) -> float:
    """近因: 距上次访问(或创建)的指数衰减, 半衰期默认 7 天。"""
    ts = last_access or created_at or time.time()
    hours = max(0.0, (time.time() - float(ts)) / 3600.0)
    return round(0.5 ** (hours / half_life_h), 4)


def touch_notes(note_ids: List[int]) -> int:
    """访问强化: 更新 last_access/access_count (被召回即'用过一次')。"""
    ids = [int(i) for i in note_ids if str(i).isdigit()]
    if not ids:
        return 0
    con = _conn()
    now = time.time()
    qm = ",".join("?" * len(ids))
    try:
        con.execute(f"UPDATE notes SET last_access=?, access_count=COALESCE(access_count,0)+1"
                    f" WHERE id IN ({qm})", tuple([now] + ids))
        con.commit()
    except Exception:
        pass
    con.close()
    return len(ids)


def save(paper_id: str, task: str, title: str = "", summary: str = "",
         key_points: Optional[List[str]] = None, entities: Optional[List] = None,
         relations: Optional[List] = None) -> Dict:
    con = _conn()
    con.execute(
        "INSERT OR REPLACE INTO memories(paper_id,task,title,summary,key_points,entities,relations,created_at)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (paper_id, task, _clean(title), _clean(summary),
         json.dumps(key_points or [], ensure_ascii=False),
         json.dumps(entities or [], ensure_ascii=False),
         json.dumps(relations or [], ensure_ascii=False), time.time()))
    con.commit()
    con.close()
    return {"ok": True, "paper_id": paper_id, "task": task}


def save_from_out(out_dir: str, paper_id: str, task: str) -> Dict:
    """从一次运行的产物自动沉淀记忆。"""
    out = Path(out_dir)
    def _j(name, dflt):
        try:
            return json.loads((out / name).read_text(encoding="utf-8"))
        except Exception:
            return dflt
    passes = _j("passes.json", {})
    mem = _j("memory.json", {})
    # 标题: 取论文首行或 paper_id
    title = paper_id
    try:
        secs = _j("sections.json", [])
        for s in secs:
            txt = (s.get("text", "") or "").strip().splitlines()
            if txt and len(txt[0]) < 120:
                title = txt[0]
                break
    except Exception:
        pass
    key_points = []
    for line in (passes.get("pass1", "") or "").splitlines():
        line = line.strip(" -•\t")
        if len(line) > 6:
            key_points.append(line[:160])
    return save(paper_id, task, title=title,
                summary=(passes.get("pass1", "") or "")[:600],
                key_points=key_points[:12],
                entities=mem.get("entities", []),
                relations=mem.get("relations", []))


def search(q: str = "", limit: int = 50) -> List[Dict]:
    con = _conn()
    con.row_factory = sqlite3.Row
    if q:
        like = f"%{q}%"
        rows = con.execute(
            "SELECT * FROM memories WHERE paper_id LIKE ? OR title LIKE ? OR summary LIKE ?"
            " OR entities LIKE ? OR key_points LIKE ? ORDER BY created_at DESC LIMIT ?",
            (like, like, like, like, like, limit)).fetchall()
    else:
        rows = con.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
                           (limit,)).fetchall()
    con.close()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("key_points", "entities", "relations"):
            try:
                d[k] = json.loads(d.get(k) or "[]")
            except Exception:
                d[k] = []
        out.append(d)
    return out


def get(paper_id: str, task: Optional[str] = None) -> List[Dict]:
    con = _conn()
    con.row_factory = sqlite3.Row
    if task:
        rows = con.execute("SELECT * FROM memories WHERE paper_id=? AND task=?",
                           (paper_id, task)).fetchall()
    else:
        rows = con.execute("SELECT * FROM memories WHERE paper_id=? ORDER BY created_at DESC",
                           (paper_id,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


def delete(mem_id: int) -> Dict:
    con = _conn()
    con.execute("DELETE FROM memories WHERE id=?", (mem_id,))
    con.commit()
    con.close()
    return {"ok": True}


def clear() -> Dict:
    con = _conn()
    con.execute("DELETE FROM memories")
    con.commit()
    con.close()
    return {"ok": True}


# ============================================================ 原子笔记 (D)
_STOP = {"the", "a", "an", "of", "and", "or", "is", "are", "was", "were", "be", "to",
         "in", "on", "for", "with", "as", "by", "we", "our", "this", "that", "these",
         "those", "it", "at", "from", "using", "based", "can", "not", "no", "such",
         "which", "than", "then", "also", "into", "over", "under", "between", "的", "了",
         "和", "与", "及", "中", "对", "为", "在", "是", "等", "通过", "进行", "本文",
         "我们", "该", "这", "其", "被", "而", "以", "并", "或"}

# 实体类型词/学术泛词: 不能当关键词展示 (否则 "RBE（Evaluation Metric）" 抽出一堆 Evaluation/Metric)
_KW_STOP = {"evaluation", "metric", "metrics", "algorithm", "model", "models", "dataset",
            "datasets", "benchmark", "benchmarks", "theoretical", "component", "components",
            "problem", "problems", "task", "tasks", "limitation", "limitations", "artifact",
            "method", "methods", "approach", "result", "results", "paper", "figure", "table",
            "study", "data", "analysis", "performance", "accuracy", "baseline"}


def _keywords(text: str, k: int = 12) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{2,}|\d+(?:\.\d+)?", text or "")
    # 中文无词边界: 用 2-gram 近似分词, 提升链接召回
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text or ""):
        toks += [run[i:i + 2] for i in range(len(run) - 1)]
    freq: Dict[str, int] = {}
    for t in toks:
        tl = t.lower()
        if tl in _STOP or t in _STOP or tl in _KW_STOP:
            continue
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [t for t, _ in ranked[:k]]


def _kwset(kws) -> set:
    return {str(k).lower() for k in (kws or [])}


def _jacc(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _should_link(a: set, b: set, th: float) -> bool:
    """链接判定: 共享专名(ASCII 长词) 或 共享≥2 词 或 Jaccard 达标。"""
    shared = a & b
    if any(len(x) >= 5 and x.isascii() for x in shared):
        return True
    if len(shared) >= 2:
        return True
    return len(shared) / len(a | b) >= th if (a and b) else False


def _fts_candidates(con, kws: List[str], limit: int = 12, max_terms: int = 6) -> List[int]:
    """链接候选的快查询: 只取前 max_terms 个关键词且不排序 (只需可比, 不需最优)。
    比 _bm25_ids 快数倍 (省 OR 展开与 bm25 排序), 大库下是写入主成本。"""
    toks: List[str] = []
    for k in kws:
        for t in _expand(str(k)).split():
            if t not in toks:
                toks.append(t)
        if len(toks) >= max_terms:
            break
    expr = " OR ".join('"%s"' % t.replace('"', "") for t in toks[:max_terms])
    if not expr:
        return []
    try:
        rows = con.execute(
            "SELECT n.id FROM notes_fts JOIN notes n ON n.id = notes_fts.note_id"
            f" WHERE notes_fts MATCH ? AND n.kind!='entity' AND {_status_sql('n')} LIMIT ?",
            (expr, limit)).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _link_candidates(con, kws: List[str], lit: List[Dict], limit: int = 12) -> List[Dict]:
    """候选链接集: 用 FTS 以本 note 关键词检索近邻 (O(log) 而非 O(n)),
    不足时回退最近 limit 条; 同批 lit 始终纳入。返回 [{id, keywords}]。
    修复: 原实现每条新笔记都与全库两两比较 (O(n²), 5 万次 _kwset/500 条)。"""
    cand: Dict[int, List] = {c["id"]: c.get("keywords", []) for c in lit}
    try:
        ids = [i for i in _fts_candidates(con, kws, limit) if i not in cand]
        if ids:
            qm = ",".join("?" * len(ids))
            for r in con.execute(f"SELECT id, keywords FROM notes WHERE id IN ({qm})", tuple(ids)):
                try:
                    kw = json.loads(r[1] or "[]")
                except Exception:
                    kw = []
                cand[r[0]] = kw
    except Exception:
        pass
    if not cand and len(kws) >= 2:
        # FTS 无命中: 取最近 limit 条兜底 (有界, 不再全库扫描)
        try:
            for r in con.execute(
                    "SELECT id, keywords FROM notes ORDER BY id DESC LIMIT ?", (limit,)):
                try:
                    kw = json.loads(r[1] or "[]")
                except Exception:
                    kw = []
                cand[r[0]] = kw
        except Exception:
            pass
    return [{"id": i, "keywords": k} for i, k in cand.items()]


def add_notes(paper_id: str, task: str, notes: List[Dict], link_th: float = 0.28,
              replace_ids: Optional[List[int]] = None) -> Dict:
    """写入原子笔记并自动建链。
    note: {kind, content, source?, anchor?, confidence?, status?, rel?}。
    rel: 可选类型化边 (supports/contradicts/extends) → note_links。
    replace_ids: 同一事务内"先插新、后清旧"的旧笔记 id (防中途失败丢整篇)。
    返回 {added, linked, fts, ids, replaced, kept_existing?}。"""
    con = _conn()
    con.row_factory = sqlite3.Row
    repl = {int(i) for i in (replace_ids or [])}
    added = linked = fts_ok = 0
    ids: List[int] = []
    lit: List[Dict] = []
    for n in notes:
        content = _clean(str(n.get("content", "")))[:1200]
        if len(content) < 6:
            continue
        kws = _keywords(content)
        cur = {"paper_id": paper_id, "task": task, "kind": str(n.get("kind", "note")),
               "content": content, "keywords": kws, "tags": n.get("tags", []),
               "context": content[:120], "links": [], "source": str(n.get("source", "")),
               "anchor": str(n.get("anchor", "")),
               "status": str(n.get("status", "active")),
               "confidence": float(n.get("confidence", 1.0))}
        links: List[int] = []
        my_kw = _kwset(kws)
        for e in _link_candidates(con, kws, lit, limit=12):
            if int(e["id"]) in repl:  # 不链接即将替换的旧笔记 (防新链变悬空)
                continue
            if _should_link(my_kw, _kwset(e.get("keywords")), link_th):
                links.append(e["id"])
                if len(links) >= 6:  # 上限: 只保留最强 6 条 (控写放大, 也提纯网络)
                    break
        cur["links"] = links
        curid = con.execute(
            "INSERT INTO notes(paper_id,task,kind,content,keywords,tags,context,links,source,created_at,anchor,status,confidence,importance,valid_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cur["paper_id"], cur["task"], cur["kind"], cur["content"],
             json.dumps(kws, ensure_ascii=False), json.dumps(cur["tags"], ensure_ascii=False),
             cur["context"], json.dumps(links), cur["source"], time.time(),
             cur["anchor"], cur["status"], cur["confidence"],
             float(cur.get("importance") or _importance(cur["kind"], content)),
             time.time())).lastrowid
        cur["id"] = curid
        # FTS 索引 (BM25 检索)
        try:
            con.execute("INSERT INTO notes_fts(note_id, body) VALUES(?,?)",
                        (curid, _fts_body(content, kws)))
            fts_ok += 1
        except Exception:
            pass
        # 反向链 + 类型化边 (记忆进化)
        for eid in links:
            row = con.execute("SELECT links FROM notes WHERE id=?", (eid,)).fetchone()
            try:
                el = json.loads(row["links"]) if row and row["links"] else []
            except Exception:
                el = []
            if curid not in el and len(el) < 24:  # 反向链封顶: 防单节点 links 无限膨胀
                el.append(curid)
                con.execute("UPDATE notes SET links=? WHERE id=?", (json.dumps(el), eid))
            if n.get("rel"):
                try:
                    con.execute("INSERT OR IGNORE INTO note_links VALUES(?,?,?)",
                                (curid, eid, str(n["rel"])))
                except Exception:
                    pass
        added += 1
        linked += len(links)
        ids.append(curid)
        lit.append(cur)
    replaced = 0
    if repl:  # 先插新、后清旧: 同事务, 失败则整体回滚 (旧笔记仍在)
        replaced = _purge_notes(con, repl)
    con.commit()
    con.close()
    if replaced:
        try:
            from .vector_store import delete_ids
            delete_ids(list(repl))
        except Exception:
            pass
    return {"added": added, "linked": linked, "fts": fts_ok, "ids": ids, "replaced": replaced}


# 单一"可检索状态"定义: active 正常; contested 参与但会被提示存在争议;
# candidate(未验证)/rejected(驳回)/superseded(被取代) 一律不进默认检索 (质检门生效点)。
RETRIEVABLE = ("active", "contested")


def _status_sql(alias: str = "n") -> str:
    return (f"COALESCE({alias}.status,'active') IN ('active','contested')"
            f" AND {alias}.invalid_at IS NULL")  # 双时间: 已失效的留档但不参与检索


def _bm25_ids(con, q: str, n: int) -> List[int]:
    try:
        expr = _match_expr(q)
        if not expr:
            return []
        rows = con.execute(
            "SELECT n.id FROM notes_fts JOIN notes n ON n.id = notes_fts.note_id"
            f" WHERE notes_fts MATCH ? AND n.kind!='entity' AND {_status_sql('n')}"
            " ORDER BY bm25(notes_fts) LIMIT ?",
            (expr, n)).fetchall()
        ids = [r[0] for r in rows]
        if ids:
            return ids
    except Exception:
        pass
    like = f"%{q}%"
    rows = con.execute(
        "SELECT id FROM notes WHERE (content LIKE ? OR keywords LIKE ? OR tags LIKE ?"
        f" OR source LIKE ?) AND kind!='entity' AND {_status_sql('notes')} LIMIT ?",
        (like, like, like, like, n)).fetchall()
    return [r[0] for r in rows]


def _vector_ids(con, q: str, n: int) -> List[int]:
    """DuckDB VSS/HNSW 向量召回；VSS 不可用时返回 [] 交给 BM25。"""
    try:
        from . import embeddings, vector_store
        if not embeddings.available() or not vector_store.available():
            return []
        qv = embeddings.embed_one(q)
        if not qv:
            return []
        meta = {r[0]: (r[1], r[2]) for r in con.execute(
            f"SELECT id, kind, COALESCE(status,'active') FROM notes"
            f" WHERE kind!='entity' AND {_status_sql('notes')}")}
        storage_key = embeddings.storage_model()
        # Existing SQLite BLOB vectors are migrated once into the HNSW index.
        if vector_store.count(storage_key, len(qv)) == 0:
            rows = con.execute(
                "SELECT owner_id, hash, dim, vec FROM vectors "
                "WHERE owner_type='note' AND model=?", (storage_key,)).fetchall()
            if not rows and storage_key != embeddings.model():
                rows = con.execute(
                    "SELECT owner_id, hash, dim, vec FROM vectors "
                    "WHERE owner_type='note' AND model=?", (embeddings.model(),)).fetchall()
            migration = [(oid, digest, embeddings.from_blob(blob))
                         for oid, digest, dim, blob in rows
                         if oid in meta and int(dim or 0) == len(qv)]
            if migration:
                vector_store.upsert(storage_key, migration)
        # Normal path remains an index-friendly Top-K query. Status changes and deletes are
        # synchronized into VSS; the final filter is defense in depth for interrupted upgrades.
        ranked = vector_store.search(storage_key, qv, max(n * 3, n))
        return [oid for oid in ranked if oid in meta][:n]
    except Exception:
        return []


def _rrf_scores(rank_lists: List[List[int]], c: int = 60,
                weights: Optional[List[float]] = None) -> Dict[int, float]:
    """RRF 排名倒数融合 (无需分数校准)。weights 可给各列表加权 (BM25 主导, 向量辅助)。"""
    score: Dict[int, float] = {}
    w = list(weights or [1.0] * len(rank_lists))
    for li, lst in enumerate(rank_lists):
        ww = w[li] if li < len(w) else 1.0
        for rank, oid in enumerate(lst):
            score[oid] = score.get(oid, 0.0) + ww / (c + rank + 1)
    return score


def _term_hit(term: str, low_text: str) -> bool:
    """画像词命中: ASCII 用词边界 (防 'let' 命中 'delete'), 中文用子串。"""
    t = str(term).strip().lower()
    if not t:
        return False
    if re.search(r"[\u4e00-\u9fff]", t):
        return t in low_text
    return re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low_text) is not None


def _profile_boost(con, note_ids: List[int]) -> Dict[int, float]:
    """按偏好加权: 命中偏好词的笔记最多 +20% (词边界匹配, 防误命中)。
    单条 IN 查询 (原实现按 id 逐条 SELECT = N+1)。"""
    if not note_ids:
        return {}
    try:
        prefs = [r[0] for r in con.execute("SELECT term FROM preferences")]
    except Exception:
        return {}
    if not prefs:
        return {}
    out = {}
    qm = ",".join("?" * len(note_ids))
    for nid, content, kws in con.execute(
            f"SELECT id, content, keywords FROM notes WHERE id IN ({qm})", tuple(note_ids)):
        low = ((content or "") + " " + (kws or "")).lower()
        hits = sum(1 for t in prefs if _term_hit(t, low))
        if hits:
            out[nid] = 1.0 + 0.2 * min(1.0, hits / 2.0)
    return out


def _final_score(rel_frac: float, imp: float, rec: float, boost: float = 1.0,
                 non_rel: float = 1.0) -> float:
    """检索终分 = 相关0.80(min-max) + (重要性0.12+近因0.08)×non_rel, 再乘偏好(≤1.2)。
    相关性占主导 (0.25 的相关优势 > 非相关项最大加成 0.20); non_rel 按词面置信度衰减:
    模糊查询下重要性/近因是噪声, 衰减为 0 让位给融合分。评分实证见 tools/eval_retrieval.py。"""
    return (0.80 * rel_frac + non_rel * (0.12 * imp + 0.08 * rec)) * boost


def _lexical_confidence(q: str, top_ids: List[int], con) -> float:
    """查询与 BM25 头部的词面置信度 = 查询词条命中头部内容的比例。
    精确查询≈1 (信 BM25); 口语化改写/跨语种≈0 (该让向量主导)。"""
    from .retrieval import tokenize
    qt = [t for t in tokenize(q) if len(t) >= 2][:12]
    if not qt or not top_ids:
        return 0.0
    qm = ",".join("?" * len(top_ids))
    try:
        rows = con.execute(f"SELECT content FROM notes WHERE id IN ({qm})", tuple(top_ids)).fetchall()
        blob = " ".join((r[0] or "") for r in rows).lower()
    except Exception:
        return 0.0
    hit = sum(1 for t in qt if t in blob)
    return hit / len(qt)


def _lex_trust(conf: float) -> float:
    """词面信任度: conf>0.6 线性升到 1 (信词面/三因子/画像); ≤0.6 归 0 (让位向量融合)。
    评测 conf 双峰 (关键词=1.00, 模糊≤0.58), 此分段给出两档行为且中间平滑。"""
    return max(0.0, min(1.0, (float(conf) - 0.6) / 0.4))


def search_notes(q: str = "", limit: int = 80, hybrid: bool = True,
                 include_pending: bool = False, touch: bool = False) -> List[Dict]:
    """检索笔记: 混合检索(BM25+向量, RRF) × 三因子(相关+重要性+近因) × 偏好加权。
    默认只返回 active/contested (质检门: candidate/rejected/superseded 不入检索);
    include_pending=True 时放开 (仅供待裁决视图/测试)。touch=True 时更新访问时间(强化)。
    任何一路失败自动降级。"""
    con = _conn()
    con.row_factory = sqlite3.Row
    if not q:
        where = "1=1" if include_pending else _status_sql("notes")
        rows = con.execute(
            f"SELECT * FROM notes WHERE {where} ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
        con.close()
        return [_note_dict(r) for r in rows]

    pool = max(limit, 40)
    bm = _bm25_ids(con, q, pool)
    vec = _vector_ids(con, q, pool) if hybrid else []
    if not (bm or vec):
        con.close()
        return []
    # 分段信任 (评测 conf 双峰: 关键词=1.00, 模糊 zh=0.00/en≤0.58):
    # 高置信 (>0.6) 才信词面排序/三因子/画像; 低于 0.6 完全让位给向量融合 (w=1.0)。
    conf = _lexical_confidence(q, bm[:3], con) if bm else 0.0
    trust = _lex_trust(conf)
    if trust < 0.5:
        # 模糊/跨语种: 两条列表的长尾都是噪声, 收缩候选池 (评测: 池12 显著优于池40)
        bm = bm[:12]
        vec = vec[:12]
    non_rel = trust
    _lists, _wts = [], []
    if bm:
        _lists.append(bm)
        _wts.append(1.0)
    if vec:
        w_vec = 1.0 - trust
        if w_vec > 0.05:
            _lists.append(vec)
            _wts.append(round(w_vec, 2))
    scores = _rrf_scores(_lists, weights=_wts)
    boost = _profile_boost(con, list(scores.keys()))
    vals = list(scores.values()) or [1.0]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0  # min-max: RRF 相邻名次差异极小, 直接除以 max 会让非相关因子反超
    meta = {r[0]: (r[1], r[2], r[3]) for r in con.execute(
        f"SELECT id, COALESCE(importance,0), last_access, created_at FROM notes"
        f" WHERE id IN ({','.join('?' * len(scores))})", tuple(scores.keys()))}
    final: Dict[int, float] = {}
    for i, rel in scores.items():
        imp, la, ca = meta.get(i, (0.0, None, None))
        rec = _recency(la, ca)
        # 偏好 boost 同样随信任度衰减: 低置信时画像也非可靠信号
        eff_boost = 1.0 + (boost.get(i, 1.0) - 1.0) * trust
        final[i] = _final_score((rel - lo) / span, float(imp), rec, eff_boost, non_rel)
    fused = sorted(final, key=lambda i: -final[i])[:limit]
    got = {}
    if fused:
        qm = ",".join("?" * len(fused))
        got = {r["id"]: r for r in con.execute(f"SELECT * FROM notes WHERE id IN ({qm})", tuple(fused))}
    con.close()
    rows = []
    for i in fused:
        if i not in got:
            continue
        d = _note_dict(got[i])
        d["score"] = round(final[i], 6)
        rows.append(d)
    if touch and rows:
        touch_notes([r["id"] for r in rows])
    return rows


def _note_dict(r) -> Dict:
    d = dict(r)
    for k in ("keywords", "tags", "links"):
        try:
            d[k] = json.loads(d.get(k) or "[]")
        except Exception:
            d[k] = []
    return d


def neighbors_of(note_ids: List[int], limit: int = 30) -> List[Dict]:
    """按 id 批量取邻居 (单条 IN 查询, 且过质检门状态过滤)。
    修复: 原 recall 按 id 逐条 SELECT (N+1) 且不过滤 candidate/rejected (门禁泄漏)。"""
    ids = [int(i) for i in note_ids if str(i).isdigit()][:limit]
    if not ids:
        return []
    con = _conn()
    con.row_factory = sqlite3.Row
    qm = ",".join("?" * len(ids))
    rows = con.execute(
        f"SELECT * FROM notes WHERE id IN ({qm}) AND kind!='entity' AND {_status_sql('notes')}",
        tuple(ids)).fetchall()
    con.close()
    return [_note_dict(r) for r in rows]


_FIELD_REFLECT_SUPPRESS = False


def suppress_field_reflection(on: bool = True) -> None:
    """批量处理期间抑制跨论文领域反思, 收尾时统一触发一次 (避免中段半库反思)。"""
    global _FIELD_REFLECT_SUPPRESS
    _FIELD_REFLECT_SUPPRESS = bool(on)


def _meta_get(key: str, default: str = "") -> str:
    con = _conn()
    try:
        row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default
    except Exception:
        return default
    finally:
        con.close()


def _meta_set(key: str, value: str) -> None:
    con = _conn()
    try:
        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, str(value)))
        con.commit()
    except Exception:
        pass
    finally:
        con.close()


def link_notes(src: int, dst: int, rel: str = "extends") -> Dict:
    """手工建立类型化记忆边 (supports/contradicts/extends) 并互链。"""
    rel = rel if rel in ("supports", "contradicts", "extends") else "extends"
    con = _conn()
    ok = False
    try:
        existing = {r[0] for r in con.execute("SELECT id FROM notes WHERE id IN (?,?)",
                                              (int(src), int(dst)))}
        if existing != {int(src), int(dst)}:
            raise ValueError("note endpoint missing")
        con.execute("INSERT OR IGNORE INTO note_links VALUES(?,?,?)", (int(src), int(dst), rel))
        for a, b in ((src, dst), (dst, src)):
            row = con.execute("SELECT links FROM notes WHERE id=?", (a,)).fetchone()
            try:
                arr = json.loads(row[0]) if row and row[0] else []
            except Exception:
                arr = []
            if int(b) not in arr:
                arr.append(int(b))
                con.execute("UPDATE notes SET links=? WHERE id=?", (json.dumps(arr), a))
        con.commit()
        ok = True
    except Exception as e:
        try:
            con.rollback()
        except Exception:
            pass
        error = str(e)[:120]
    con.close()
    out = {"ok": ok, "src": src, "dst": dst, "rel": rel}
    if not ok:
        out["error"] = error
    return out


def _notes_by_ids(ids: List[int]) -> List[Dict]:
    """按 id 批量取笔记 (过状态/有效性/实体过滤)。"""
    ids = [int(i) for i in ids if str(i).isdigit()][:50]
    if not ids:
        return []
    con = _conn()
    con.row_factory = sqlite3.Row
    qm = ",".join("?" * len(ids))
    rows = con.execute(
        f"SELECT * FROM notes WHERE id IN ({qm}) AND kind!='entity' AND {_status_sql('notes')}",
        tuple(ids)).fetchall()
    con.close()
    return [_note_dict(r) for r in rows]


def invalidate_note(note_id: int, when: Optional[float] = None) -> Dict:
    """双时间失效 (Graphiti 式): 保留历史但移出检索 (invalid_at 置时间戳)。"""
    con = _conn()
    n = 0
    try:
        cur = con.execute("UPDATE notes SET invalid_at=? WHERE id=?", (when or time.time(), int(note_id)))
        con.commit()
        n = cur.rowcount
    except Exception:
        n = 0
    con.close()
    if n:
        try:
            from .vector_store import delete_ids
            delete_ids([int(note_id)])
        except Exception:
            pass
    return {"ok": bool(n), "id": note_id}


# ---------------------------------------------------- 图多跳 (HippoRAG 式 PPR)
def _load_edges(con) -> Dict[int, List[int]]:
    edges: Dict[int, List[int]] = {}
    try:
        for nid, js in con.execute("SELECT id, links FROM notes"):
            try:
                arr = json.loads(js or "[]")
            except Exception:
                arr = []
            edges.setdefault(int(nid), [])
            edges[int(nid)] += [int(i) for i in arr if str(i).isdigit()]
        for s, d in con.execute("SELECT src, dst FROM note_links"):
            edges.setdefault(int(s), []).append(int(d))
            edges.setdefault(int(d), []).append(int(s))
    except Exception:
        pass
    return edges


def graph_expand(seed_scores: Dict[int, float], steps: int = 2,
                 damping: float = 0.85) -> Dict[int, float]:
    """Personalized PageRank 近似: 从种子沿 笔记互链+类型边 扩散, 多跳召回。
    比原"一跳邻居"能捞回间接相关记忆 (A—B—C 链上的 C)。"""
    if not seed_scores:
        return {}
    con = _conn()
    edges = _load_edges(con)
    con.close()
    ids = set(edges) | {int(i) for i in seed_scores}
    seeds = {int(i): float(s) for i, s in seed_scores.items()}
    out = dict(seeds)
    for _ in range(max(1, steps)):
        # 懒惰随机游走 (含自环): 收敛稳定, 近跳权重高于远跳
        nxt = {i: (1 - damping) * seeds.get(i, 0.0) + damping * 0.5 * out.get(i, 0.0)
               for i in ids}
        for i, s in out.items():
            nb = edges.get(i, [])
            if not nb:
                continue
            share = damping * 0.5 * s / len(nb)
            for j in nb:
                nxt[j] = nxt.get(j, 0.0) + share
        out = nxt
    return out


def recall(q: str, k: int = 5) -> Dict:
    """按查询召回笔记并沿链接扩展一跳, 返回 {seeds, neighbors} (邻居同样过状态过滤)。"""
    seeds = search_notes(q, limit=k)
    seed_ids = {s["id"] for s in seeds}
    want = set()
    for s in seeds:
        want.update(int(i) for i in s.get("links", []) if str(i).isdigit())
    want -= seed_ids
    return {"seeds": seeds, "neighbors": neighbors_of(list(want), limit=30)}


def gc_memory() -> Dict:
    """一次性全局清理: 悬空 links 引用 / 孤儿 note_links / 孤儿 vectors。
    历史库经多次重建后 links 会堆积已删除 id (评测正例污染根因), 用此回收。"""
    con = _conn()
    con.row_factory = sqlite3.Row
    ids = {int(r[0]) for r in con.execute("SELECT id FROM notes")}
    orphan_vector_ids = [int(r[0]) for r in con.execute(
        "SELECT owner_id FROM vectors WHERE owner_type='note' "
        "AND owner_id NOT IN (SELECT id FROM notes)")]
    pruned = 0
    fixes = []
    for nid, js in con.execute("SELECT id, links FROM notes"):
        try:
            arr = [int(i) for i in json.loads(js or "[]")]
        except Exception:
            arr = []
        new = [i for i in arr if i in ids]
        if len(new) != len(arr):
            fixes.append((json.dumps(new), nid))
            pruned += 1
    for js, nid in fixes:
        con.execute("UPDATE notes SET links=? WHERE id=?", (js, nid))
    try:
        cl = con.execute("DELETE FROM note_links WHERE src NOT IN (SELECT id FROM notes)"
                         " OR dst NOT IN (SELECT id FROM notes)").rowcount
    except Exception:
        cl = 0
    try:
        cv = con.execute("DELETE FROM vectors WHERE owner_type='note'"
                         " AND owner_id NOT IN (SELECT id FROM notes)").rowcount
    except Exception:
        cv = 0
    con.commit()
    con.close()
    if orphan_vector_ids:
        try:
            from .vector_store import delete_ids
            delete_ids(orphan_vector_ids)
        except Exception:
            pass
    return {"links_pruned": pruned, "note_links_removed": cl, "vectors_removed": cv}


def _purge_notes(con, ids) -> int:
    """在调用方事务内彻底删除笔记: notes/fts/note_links/vectors + 清理其余笔记对死 id 的引用。"""
    ids = {int(i) for i in (ids or [])}
    if not ids:
        return 0
    qm = ",".join("?" * len(ids))
    t = tuple(ids)
    con.execute(f"DELETE FROM notes WHERE id IN ({qm})", t)
    try:
        con.execute(f"DELETE FROM notes_fts WHERE note_id IN ({qm})", t)
    except Exception:
        pass
    try:
        con.execute(f"DELETE FROM note_links WHERE src IN ({qm}) OR dst IN ({qm})", t + t)
    except Exception:
        pass
    try:
        con.execute(f"DELETE FROM vectors WHERE owner_type='note' AND owner_id IN ({qm})", t)
    except Exception:
        pass
    # 悬空引用清理 (防 links 无限膨胀): 逐条剔除已删 id
    fixes = []
    for nid, js in con.execute("SELECT id, links FROM notes"):
        try:
            arr = json.loads(js or "[]")
        except Exception:
            arr = []
        new = [i for i in arr if int(i) not in ids]
        if len(new) != len(arr):
            fixes.append((json.dumps(new), nid))
    for js, nid in fixes:
        con.execute("UPDATE notes SET links=? WHERE id=?", (js, nid))
    return len(ids)


def delete_note(note_id: int) -> Dict:
    con = _conn()
    try:
        n = _purge_notes(con, [note_id])
        con.commit()
    except Exception:
        n = 0
    con.close()
    if n:
        try:
            from .vector_store import delete_ids
            delete_ids([int(note_id)])
        except Exception:
            pass
    return {"ok": bool(n), "id": note_id}


def clear_notes() -> Dict:
    con = _conn()
    ok = False
    try:
        con.execute("DELETE FROM notes")
        con.execute("DELETE FROM notes_fts")
        con.execute("DELETE FROM note_links")
        con.execute("DELETE FROM vectors WHERE owner_type='note'")  # 孤儿向量一并清 (防膨胀)
        con.commit()
        ok = True
    except Exception as e:
        try:
            con.rollback()
        except Exception:
            pass
        error = str(e)[:120]
    con.close()
    if ok:
        try:
            from .vector_store import clear as clear_vector_index
            clear_vector_index()
        except Exception:
            pass
    return {"ok": ok, **({"error": error} if not ok else {})}


def build_notes_from_out(out_dir: str, paper_id: str, task: str) -> Dict:
    """从 deepread 产物拆分原子笔记: 深度解读各小标题 + 论证地图主张 + 实体。
    幂等: 重跑不累积; 但产物为空时**保留旧笔记** (防误删已有记忆)。"""
    out = Path(out_dir)
    notes: List[Dict] = []
    md = ""
    try:
        md = (out / "deepread.md").read_text(encoding="utf-8")
    except Exception:
        pass
    # Artifact-level verification is a prerequisite, not a substitute, for the
    # claim-level check below. Missing/legacy reports fail closed.
    artifact_status = {"brief": "MISSING", "full": "MISSING"}
    try:
        verification = json.loads((out / "deepread_verify.json").read_text(encoding="utf-8"))
        artifacts = verification.get("artifacts") if isinstance(verification, dict) else None
        if isinstance(artifacts, dict):
            for name in artifact_status:
                result = artifacts.get(name)
                if isinstance(result, dict):
                    artifact_status[name] = str(result.get("status", "MISSING"))
        elif isinstance(verification, dict):
            # Legacy reports verified only the full artifact; the brief remains untrusted.
            artifact_status["full"] = str(verification.get("status", "MISSING"))
    except Exception:
        pass
    # 按 ## 小标题切
    if md:
        blocks = re.split(r"^##\s+", md, flags=re.M)
        for b in blocks[1:]:
            head, _, body = b.partition("\n")
            head, body = head.strip(), body.strip()
            if head and body and head not in ("一句话主张",):
                notes.append({"kind": head, "content": body[:900], "source": head})
    # 论证地图主张
    try:
        synth = json.loads((out / "deepread_synthesis.json").read_text(encoding="utf-8"))
        for a in (synth.get("argument_map") or []):
            if isinstance(a, dict) and a.get("claim"):
                notes.append({"kind": "claim",
                              "content": f"{a.get('claim')} | 证据: {a.get('evidence','')} | 局限: {a.get('limitation','')}",
                              "source": "argument_map"})
    except Exception:
        pass
    # 实体
    mem = {}
    try:
        mem = json.loads((out / "memory.json").read_text(encoding="utf-8"))
        for e in (mem.get("entities") or []):
            nm = e.get("name") if isinstance(e, dict) else str(e)
            if nm:
                notes.append({"kind": "entity", "content": f"{nm}（{(e or {}).get('type','') if isinstance(e,dict) else ''}）",
                              "source": "entity", "entity_name": str(nm)})
    except Exception:
        pass
    if not notes:
        return {"added": 0, "linked": 0, "fts": 0, "ids": [], "kept_existing": True}
    # 深读正文可能由模型生成。用论文材料批量验真；无材料或证据不足时只进入 candidate。
    source_ctx = ""
    try:
        state = json.loads((out / "state.json").read_text(encoding="utf-8"))
        source_ctx = "\n".join(str(v) for v in (state.get("ground_truth") or {}).values())
    except Exception:
        pass
    if not source_ctx:
        try:
            sections = json.loads((out / "sections.json").read_text(encoding="utf-8"))
            source_ctx = "\n".join(str(s.get("text", "")) for s in sections if isinstance(s, dict))
        except Exception:
            pass
    check_idx = [i for i, n in enumerate(notes) if n.get("kind") != "entity"]
    verdicts = _verify_claims([notes[i]["content"] for i in check_idx], source_ctx)
    for i, ok in zip(check_idx, verdicts):
        artifact = "full" if notes[i].get("source") == "argument_map" else "brief"
        artifact_clean = artifact_status.get(artifact) == "CLEAN"
        notes[i]["status"] = "active" if ok and artifact_clean else "candidate"
        notes[i]["tags"] = list(dict.fromkeys(list(notes[i].get("tags") or []) +
                                               [f"artifact:{artifact}",
                                                f"artifact-status:{artifact_status.get(artifact, 'MISSING')}"]))
    try:
        from .graph import load_graph_policy
        graph_policy = load_graph_policy()
    except Exception:
        graph_policy = {"valid": False, "review_due": True}
    policy_clean = bool(graph_policy.get("valid")) and not bool(graph_policy.get("review_due"))
    def source_mentions(entity_name: str) -> bool:
        escaped = re.escape(entity_name)
        if (entity_name and entity_name[0].isascii() and entity_name[0].isalnum() and
                entity_name[-1].isascii() and entity_name[-1].isalnum()):
            escaped = r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])"
        return re.search(escaped, source_ctx, re.I) is not None

    for note in notes:
        if note.get("kind") != "entity":
            continue
        entity_name = str(note.get("entity_name", "")).strip()
        source_bound = bool(entity_name and source_mentions(entity_name))
        note["status"] = "active" if policy_clean and source_bound else "candidate"
        note["tags"] = list(dict.fromkeys(list(note.get("tags") or []) + [
            f"graph-policy:{'CURRENT' if policy_clean else 'REVIEW_DUE'}",
            f"source-bound:{str(source_bound).lower()}",
        ]))
    for n in notes:
        n["tags"] = list(dict.fromkeys(list(n.get("tags") or []) +
                                        ["paperbrain:generated", f"task:{task}"]))
    con = _conn()
    try:
        # 只替换同任务的系统生成笔记；人工笔记、裁决和历史记录必须保留。
        old_ids = [r[0] for r in con.execute(
            "SELECT id FROM notes WHERE paper_id=? AND task=? AND tags LIKE ?",
            (paper_id, task, '%"paperbrain:generated"%'))]
    except Exception:
        old_ids = []
    con.close()
    # 先插新、后清旧 (add_notes 单事务): 中途失败不会丢整篇旧记忆
    return add_notes(paper_id, task, notes, replace_ids=old_ids)


# ======================================================= 跨论文知识层 (全局)
_GENERIC_CONCEPT = {"algorithm", "model", "models", "method", "methods", "methodology",
                    "evaluation", "metric", "metrics", "dataset", "datasets", "benchmark",
                    "paper", "result", "results", "approach", "table", "figure", "fig",
                    "baseline", "performance", "accuracy", "experiment", "experiments",
                    "study", "work", "analysis", "data", "training", "test", "task",
                    "问题", "方法", "模型", "结果", "评估", "指标", "数据", "实验", "分析"}


def _is_generic_concept(term: str) -> bool:
    from .graph import GENERIC_BLACKLIST
    t = str(term).strip().lower()
    if len(t) < 2:
        return True
    return t in GENERIC_BLACKLIST or t in _GENERIC_CONCEPT


def build_concepts(paper_id: str, top: int = 18) -> Dict:
    """从该篇笔记关键词+实体提炼概念, 写入 concepts 表 (weight=出现次数)。
    入库前经别名消解, 避免同一概念多种写法。"""
    con = _conn()
    con.row_factory = sqlite3.Row
    rows = con.execute(
        f"SELECT keywords, content FROM notes WHERE paper_id=? AND {_status_sql('notes')}",
        (paper_id,)).fetchall()
    freq: Dict[str, float] = {}
    for r in rows:
        try:
            kws = json.loads(r["keywords"] or "[]")
        except Exception:
            kws = []
        for k in kws:
            ks = resolve_alias(str(k))
            if _is_generic_concept(ks):
                continue
            freq[ks] = freq.get(ks, 0) + 1
    # 实体加权
    for e in get(paper_id):
        try:
            for ent in json.loads(e.get("entities") or "[]"):
                nm = ent.get("name") if isinstance(ent, dict) else str(ent)
                nm = resolve_alias(str(nm)) if nm else ""
                if nm and not _is_generic_concept(nm):
                    freq[nm] = freq.get(nm, 0) + 3
        except Exception:
            pass
    con.execute("DELETE FROM concepts WHERE paper_id=?", (paper_id,))
    kept = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top]
    for term, w in kept:
        con.execute("INSERT OR REPLACE INTO concepts VALUES(?,?,?)", (paper_id, term, w))
    con.commit()
    con.close()
    return {"paper_id": paper_id, "concepts": len(kept)}


# ------------------------------------------------------------ 别名/实体消解
def add_alias(alias: str, canonical: str) -> Dict:
    a, c = _clean(str(alias)), _clean(str(canonical))
    if not a or not c or a.lower() == c.lower():
        return {"ok": False}
    con = _conn()
    con.execute("INSERT OR REPLACE INTO aliases VALUES(?,?)", (a, c))
    con.commit()
    con.close()
    return {"ok": True, "alias": a, "canonical": c}


def resolve_alias(name: str) -> str:
    n = _clean(str(name))
    if not n:
        return n
    try:
        con = _conn()
        row = con.execute("SELECT canonical FROM aliases WHERE alias=?", (n,)).fetchone()
        con.close()
        return row[0] if row else n
    except Exception:
        return n


def list_aliases(limit: int = 200) -> List[Dict]:
    con = _conn()
    rows = con.execute("SELECT alias, canonical FROM aliases LIMIT ?", (limit,)).fetchall()
    con.close()
    return [{"alias": r[0], "canonical": r[1]} for r in rows]


def link_papers(min_shared: int = 2) -> Dict:
    """计算论文两两共享概念, 建立跨论文边 (全局知识网络)。"""
    con = _conn()
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT paper_id, concept FROM concepts").fetchall()
    byp: Dict[str, set] = {}
    for r in rows:
        byp.setdefault(r["paper_id"], set()).add(r["concept"].lower())
    papers = sorted(byp)
    con.execute("DELETE FROM paper_edges")
    edges = 0
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            shared = byp[papers[i]] & byp[papers[j]]
            if len(shared) >= min_shared:
                con.execute("INSERT OR REPLACE INTO paper_edges VALUES(?,?,?,?)",
                            (papers[i], papers[j], json.dumps(sorted(shared), ensure_ascii=False),
                             float(len(shared))))
                edges += 1
    con.commit()
    con.close()
    return {"papers": len(papers), "edges": edges}


def field_map(limit: int = 50) -> Dict:
    """领域地图: 跨论文共享概念 + 论文边 + 连通簇。"""
    con = _conn()
    con.row_factory = sqlite3.Row
    concepts: Dict[str, List[str]] = {}
    for r in con.execute("SELECT paper_id, concept FROM concepts"):
        concepts.setdefault(r["concept"], []).append(r["paper_id"])
    edges = []
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in con.execute("SELECT a,b,shared,weight FROM paper_edges ORDER BY weight DESC"):
        shared = json.loads(r["shared"] or "[]")
        edges.append({"a": r["a"], "b": r["b"], "weight": r["weight"], "shared": shared})
        union(r["a"], r["b"])
    papers = [r["paper_id"] for r in con.execute("SELECT DISTINCT paper_id FROM concepts")]
    for p in papers:
        find(p)
    clusters: Dict[str, List[str]] = {}
    for p in papers:
        clusters.setdefault(find(p), []).append(p)
    con.close()
    top_concepts = sorted(
        ({"term": k, "papers": v} for k, v in concepts.items() if len(v) >= 2),
        key=lambda x: -len(x["papers"]))[:limit]
    return {"edges": edges,
            "clusters": [sorted(v) for v in clusters.values() if len(v) >= 2],
            "shared_concepts": top_concepts,
            "n_papers": len(papers)}


def clear_concepts() -> Dict:
    con = _conn()
    con.execute("DELETE FROM concepts")
    con.execute("DELETE FROM paper_edges")
    con.commit()
    con.close()
    return {"ok": True}


def finalize_memory(out_dir: str, paper_id: str, task: str) -> Dict:
    """一次运行的收尾: 记忆 + 原子笔记 + 跨论文概念与连边 (统一入口)。"""
    out: Dict = {}
    try:
        out["memory"] = save_from_out(out_dir, paper_id, task)
    except Exception as e:
        out["memory_error"] = str(e)[:120]
    try:
        out["notes"] = build_notes_from_out(out_dir, paper_id, task)
    except Exception as e:
        out["notes_error"] = str(e)[:120]
    try:
        out["concepts"] = build_concepts(paper_id)
        out["paper_edges"] = link_papers()
    except Exception as e:
        out["concepts_error"] = str(e)[:120]
    try:
        out["memory_card"] = memory_card_from_out(out_dir, paper_id)
    except Exception as e:
        out["memory_card_error"] = str(e)[:120]
    try:
        out["embedded"] = embed_notes(paper_id)
    except Exception as e:
        out["embed_error"] = str(e)[:120]
    try:
        out["consolidate"] = consolidate(paper_id)
    except Exception as e:
        out["consolidate_error"] = str(e)[:120]
    try:
        from . import config
        if config.reflect_enabled():
            out["reflection"] = reflect(paper_id)
    except Exception as e:
        out["reflection_error"] = str(e)[:120]
    try:  # 跨论文领域反思: 每新增 3 篇触发一次 (meta 记账防重跑重复生成; 批量期抑制)
        from . import config as _cfg
        if _cfg.reflect_enabled() and not _FIELD_REFLECT_SUPPRESS:
            con = _conn()
            npapers = con.execute(
                "SELECT COUNT(DISTINCT paper_id) FROM notes WHERE paper_id!='__field__'").fetchone()[0]
            con.close()
            last = int(_meta_get("field_reflect_at", "0") or 0)
            if npapers >= max(3, last + 3):
                fr = reflect_global()
                out["field_reflection"] = fr
                if fr.get("ok"):
                    _meta_set("field_reflect_at", str(npapers))
    except Exception as e:
        out["field_reflection_error"] = str(e)[:120]
    try:
        from .adjudication import start
        out["automatic_review"] = start()
    except Exception as e:
        out["automatic_review_error"] = str(e)[:120]
    return out


# ------------------------------------------------------------ 巩固与合成
def consolidate(paper_id: Optional[str] = None) -> Dict:
    """去重巩固: 同篇同 kind 且内容前 160 字相同的重复笔记, 只留最早一条。
    删除时同步清理 FTS/向量/类型边/存活笔记里的 links 引用 (无死链)。
    返回 {removed, kept}。"""
    con = _conn()
    con.row_factory = sqlite3.Row
    sql = ("SELECT id, paper_id, kind, substr(content,1,160) AS sig FROM notes"
           + (" WHERE paper_id=?" if paper_id else "") + " ORDER BY id")
    rows = [dict(r) for r in con.execute(sql, ((paper_id,) if paper_id else ()))]
    seen: Dict[tuple, int] = {}
    removed_ids: List[int] = []
    kept = 0
    for r in rows:
        key = (r["paper_id"], r["kind"], r["sig"])
        if key in seen:
            removed_ids.append(r["id"])
            con.execute("DELETE FROM notes WHERE id=?", (r["id"],))
            try:
                con.execute("DELETE FROM notes_fts WHERE note_id=?", (r["id"],))
            except Exception:
                pass
        else:
            seen[key] = r["id"]
            kept += 1
    if removed_ids:
        qm = ",".join("?" * len(removed_ids))
        try:
            con.execute(f"DELETE FROM note_links WHERE src IN ({qm}) OR dst IN ({qm})",
                        tuple(removed_ids) * 2)
        except Exception:
            pass
        try:
            con.execute(f"DELETE FROM vectors WHERE owner_type='note' AND owner_id IN ({qm})",
                        tuple(removed_ids))
        except Exception:
            pass
        rset = set(removed_ids)
        for nid, links_json in con.execute("SELECT id, links FROM notes").fetchall():
            try:
                arr = json.loads(links_json or "[]")
            except Exception:
                arr = []
            new = [i for i in arr if i not in rset]
            if len(new) != len(arr):
                con.execute("UPDATE notes SET links=? WHERE id=?", (json.dumps(new), nid))
    con.commit()
    con.close()
    if removed_ids:
        try:
            from .vector_store import delete_ids
            delete_ids(removed_ids)
        except Exception:
            pass
    return {"removed": len(removed_ids), "kept": kept}


def _find_conflict(claim_text: str, limit: int = 8) -> Optional[Dict]:
    """为矛盾主张找真正的"对方": 排除自身与实体, 优先状态正常的旧笔记。"""
    for n in search_notes(claim_text[:160], limit=limit):
        if n.get("kind") == "entity":
            continue
        if n.get("content", "").strip() == claim_text.strip():
            continue
        return n
    return None


def _known_terms() -> set:
    """已知领域词: 概念 + 别名 (小写)。画像只认这些, 避免噪声词。"""
    con = _conn()
    known = set()
    try:
        for r in con.execute("SELECT DISTINCT concept FROM concepts"):
            known.add(str(r[0]).lower())
        for r in con.execute("SELECT alias, canonical FROM aliases"):
            known.add(str(r[0]).lower())
            known.add(str(r[1]).lower())
    except Exception:
        pass
    con.close()
    return known


def _bump_query_preferences(q: str, delta: float = 0.3) -> List[str]:
    """提问命中已知概念/别名时加权 (词边界匹配, 防 'let' 命中 'delete')。"""
    known = _known_terms()
    hit = []
    low = (q or "").lower()
    for term in known:
        if len(term) >= 2 and _term_hit(term, low):
            if bump_preference(term, delta=delta).get("ok"):
                hit.append(term)
    return hit


def _mmr(notes: List[Dict], k: int, lam: float = 0.7) -> List[Dict]:
    """MMR 选多样本, 减少近重复。相似度用关键词 Jaccard。"""
    picked: List[Dict] = []
    pool = list(notes)
    while pool and len(picked) < k:
        best, best_score = None, -1e9
        for n in pool:
            rel = 1.0 / (1 + notes.index(n))  # 近似位置相关性
            sim = 0.0
            if picked:
                a = _kwset(n.get("keywords"))
                sim = max(_jacc(a, _kwset(p.get("keywords"))) for p in picked)
            score = lam * rel - (1 - lam) * sim
            if score > best_score:
                best, best_score = n, score
        picked.append(best)
        pool.remove(best)
    return picked


def ask_memory(q: str, k: int = 6, compose: bool = True, expand: bool = True,
               rerank: bool = True) -> Dict:
    """问记忆(高质量): 查询扩展 → 混合检索(BM25+向量, RRF) → 模型重排 → MMR → 引用合成。
    任一步失败自动降级; 无模型时给离线摘要, 绝不假装。"""
    from . import config
    queries = [q]
    if expand:
        try:
            from .llm import expand_query
            from .llm_ops import _extract_json
            arr = _extract_json(expand_query(q))
            if isinstance(arr, list):
                queries += [str(x) for x in arr if isinstance(x, str)][:3]
        except Exception:
            pass
    # 多查询混合检索 → RRF
    rank_lists, pool_map = [], {}
    for v in queries:
        # Trusted Q&A excludes disputed records even though the advanced search surface
        # may expose them with a warning for human adjudication.
        hits = [h for h in search_notes(v, limit=12) if h.get("status") == "active"]
        rank_lists.append([h["id"] for h in hits])
        for h in hits:
            pool_map[h["id"]] = h
    fused_ids = [oid for oid, _ in sorted(_rrf_scores(rank_lists).items(), key=lambda x: -x[1])]
    candidates = [pool_map[i] for i in fused_ids if i in pool_map][:12]
    candidates = [c for c in candidates if c.get("kind") != "entity"]
    # 图多跳召回 (PPR): 沿互链/类型边扩散, 补回间接相关记忆
    try:
        base = {c["id"]: float(c.get("score") or 0.1) for c in candidates}
        ppr = graph_expand(base, steps=2)
        have = set(base)
        extra = [(i, s) for i, s in ppr.items() if i not in have and s > 0.02]
        extra.sort(key=lambda x: -x[1])
        mx = max((s for _, s in extra), default=0.0) or 1.0
        for i, s in extra[:6]:
            for n in _notes_by_ids([i]):
                n["score"] = 0.15 * (s / mx) + 0.01  # 图分略低于直接命中, 仅在相关面补齐
                n["via"] = "graph"
                candidates.append(n)
    except Exception:
        pass
    # 模型重排
    if rerank and len(candidates) > 1:
        try:
            from .llm import rerank as _rr
            from .llm_ops import _extract_json
            listing = "\n".join(f"[M{c['id']}] {c['content'][:160]}" for c in candidates)
            arr = _extract_json(_rr(q, listing))
            if isinstance(arr, list):
                order = [int(x) for x in arr if str(x).strip().lstrip("M").isdigit()]
                by_id = {c["id"]: c for c in candidates}
                reordered = [by_id[i] for i in order if i in by_id]
                reordered += [c for c in candidates if c["id"] not in order]
                candidates = reordered
        except Exception:
            pass
    picked = _mmr(candidates, k)
    try:  # 访问强化: 被召回即更新近因/次数 (Generative Agents recency)
        touch_notes([c["id"] for c in picked])
    except Exception:
        pass
    # 一跳图扩展 (直接由 picked 的 links 展开, 不再重复检索一次)
    seed_ids = {c["id"] for c in picked}
    want = set()
    for c in picked:
        want.update(int(i) for i in c.get("links", []) if str(i).isdigit())
    want -= seed_ids
    neighbors = neighbors_of(list(want), limit=20)
    notes, seen = [], set()
    for n in picked + neighbors:
        if (n["id"] not in seen and n.get("kind") != "entity" and
                n.get("status") == "active"):
            seen.add(n["id"])
            notes.append(n)
    notes = notes[:k + 5]
    # 隐式画像: 仅对"已知概念/别名"加权 (防中文 2-gram 噪声, 如"两篇/什么")
    try:
        _bump_query_preferences(q)
    except Exception:
        pass
    ctx = "\n".join(
        f"[M{n['id']}] ({n['paper_id']} · {n['kind']}"
        + (" · 争议" if n.get("status") == "contested" else "")
        + f") {n['content'][:300]}"
        for n in notes)
    answer = ""
    if compose and notes:
        try:
            from .llm import answer_memory
            answer = answer_memory(q, ctx)
        except Exception:
            answer = ""
    if not answer:
        lines = [f"（离线摘要 · 命中 {len(notes)} 条记忆）"]
        for n in notes[:k]:
            lines.append(f"- [{n['id']}]（{n['paper_id']} · {n['kind']}）{n['content'][:140]}")
        answer = "\n".join(lines) if notes else "（没有找到相关记忆）"
    return {"answer": answer, "used": [n["id"] for n in notes],
            "seeds": picked, "neighbors": neighbors,
            "queries": queries, "embed": bool(config.embed_enabled())}


def _reflect_candidates(paper_id: Optional[str] = None, top: int = 12) -> List[Dict]:
    """反思输入: 按 重要性×近因 取 top 条 (排除既有反思与实体, 只要可检索状态)。"""
    con = _conn()
    con.row_factory = sqlite3.Row
    where = [f"kind NOT IN ('entity','反思','领域反思','reflection')", _status_sql("notes")]
    args: List = []
    if paper_id:
        where.append("paper_id=?")
        args.append(paper_id)
    rows = con.execute(f"SELECT * FROM notes WHERE {' AND '.join(where)}", tuple(args)).fetchall()
    con.close()
    scored = []
    for r in rows:
        d = _note_dict(r)
        imp = float(d.get("importance") or _importance(d.get("kind", ""), d.get("content", "")))
        scored.append((imp * _recency(d.get("last_access"), d.get("created_at")), d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:top]]


def reflect(paper_id: Optional[str] = None, top: int = 12, n: int = 3,
            scope: str = "paper", verify: bool = True) -> Dict:
    """反思层 (Generative Agents): 对若干记忆条目综合高层洞见, 写回为可检索 note(带证据链)。
    scope="field" 时跨论文综合 (paper_id="__field__"); verify 时对洞见做证据蕴含质检,
    不过者降级 candidate。需模型; 无模型/条目不足则返回 ok=False, 不影响主流程。"""
    if scope == "field":
        cands = _reflect_candidates(None, top)
        pids = {c.get("paper_id") for c in cands
                if c.get("paper_id") not in (None, "", "__field__")}
        if len(cands) < 4 or len(pids) < 2:
            return {"ok": False, "reason": f"跨论文记忆不足({len(cands)}条/{len(pids)}篇)"}
        pid, kind = "__field__", "领域反思"
    else:
        cands = _reflect_candidates(paper_id, top)
        if len(cands) < 4:
            return {"ok": False, "reason": f"记忆不足({len(cands)}条)"}
        pid, kind = (paper_id or cands[0].get("paper_id") or "reflection"), "反思"
    ctx = "\n".join(f"[M{d['id']}] （{d.get('paper_id')} · {d.get('kind')}）{d.get('content','')[:200]}"
                    for d in cands)
    try:
        from .llm import reflect_notes
        from .llm_ops import _extract_json
        obj = _extract_json(reflect_notes(ctx, n))
    except Exception as e:
        return {"ok": False, "reason": f"模型不可用: {e}"[:120]}
    if not isinstance(obj, list):
        return {"ok": False, "reason": "反思解析失败"}
    made: List[int] = []
    verified = 0
    valid = {d["id"]: d.get("content", "") for d in cands}
    for item in obj:
        if not isinstance(item, dict) or not str(item.get("insight", "")).strip():
            continue
        insight = str(item["insight"]).strip()[:1200]
        ev_ids = []
        for ev in (item.get("evidence") or []):
            try:
                ev = int(ev)
            except Exception:
                continue
            if ev in valid:
                ev_ids.append(ev)
        status = "active"
        if verify:
            if ev_ids:
                try:
                    ev_text = "\n".join(valid[i][:300] for i in ev_ids)[:3000]
                    if not _verify_claims([insight], ev_text)[0]:
                        status = "candidate"  # 证据不足 → 待裁决, 不冒充已验证洞见
                    else:
                        verified += 1
                except Exception:
                    status = "candidate"  # 验证失败不得冒充 active
            else:
                status = "candidate"  # 无证据的洞见一律待裁决 (不放行无据断言)
        res = add_notes(pid, "reflect", [{"kind": kind, "content": insight, "status": status}])
        nid = (res.get("ids") or [None])[0]
        if not nid:
            continue
        con = _conn()
        try:
            for ev in ev_ids:
                con.execute("INSERT OR IGNORE INTO note_links VALUES(?,?,?)", (nid, ev, "supports"))
            con.commit()
        except Exception:
            pass
        finally:
            con.close()
        made.append(nid)
    return {"ok": bool(made), "insights": len(made), "verified": verified,
            "ids": made, "scope": scope}


def reflect_global(top: int = 16, n: int = 5) -> Dict:
    """跨论文领域反思: 从全库高价值记忆中综合领域级洞见 (手工/定期触发)。"""
    return reflect(None, top=top, n=n, scope="field")


def memory_stats() -> Dict:
    con = _conn()
    def c(sql, args=()):
        try:
            return con.execute(sql, args).fetchone()[0]
        except Exception:
            return 0
    st = {"papers": c("SELECT COUNT(DISTINCT paper_id) FROM notes"),
          "notes": c("SELECT COUNT(*) FROM notes"),
          "active_notes": c("SELECT COUNT(*) FROM notes WHERE COALESCE(status,'active')='active'"),
          "candidate_notes": c("SELECT COUNT(*) FROM notes WHERE status='candidate'"),
          "contested_notes": c("SELECT COUNT(*) FROM notes WHERE status='contested'"),
          "concepts": c("SELECT COUNT(*) FROM concepts"),
          "aliases": c("SELECT COUNT(*) FROM aliases"),
          "edges": c("SELECT COUNT(*) FROM paper_edges"),
          "typed_links": c("SELECT COUNT(*) FROM note_links")}
    con.close()
    return st


def _all_concepts(limit: int = 40) -> List[str]:
    con = _conn()
    try:
        rows = con.execute("SELECT DISTINCT concept FROM concepts LIMIT ?", (limit,)).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []
    finally:
        con.close()


def memory_card_from_out(out_dir: str, paper_id: str) -> Dict:
    """写时记忆卡 (1 次模型调用): 规范概念+别名 → aliases 表; 主张 → claim 笔记(带类型边)。
    写入质检门: 主张先过验证, 不足者标 candidate; 矛盾主张标记双方 contested。
    无模型/失败即返回 ok=False, 不影响主流程。"""
    out = Path(out_dir)
    ctx = ""
    try:
        ctx = (out / "deepread.md").read_text(encoding="utf-8")
    except Exception:
        pass
    try:
        mem = json.loads((out / "memory.json").read_text(encoding="utf-8"))
        ents = [e.get("name") for e in (mem.get("entities") or []) if isinstance(e, dict)]
        if ents:
            ctx += "\n\n关键实体: " + "、".join(ents)
    except Exception:
        pass
    if not ctx.strip():
        return {"ok": False, "error": "no deepread"}
    try:
        from .llm import memory_card
        from .llm_ops import _extract_json
        raw = memory_card(ctx, "、".join(_all_concepts()))
        obj = _extract_json(raw)
    except Exception as e:
        return {"ok": False, "error": str(e)[:120]}
    if not isinstance(obj, dict):
        return {"ok": False, "error": "bad json"}
    n_alias = 0
    for c in obj.get("concepts", []) or []:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        for a in (c.get("aliases") or []):
            if name and add_alias(str(a), str(name)).get("ok"):
                n_alias += 1
    raw_claims = [cl for cl in (obj.get("claims") or []) if isinstance(cl, dict) and cl.get("text")]
    # 质检门: 优先模型批量验证, 失败回退词法重合度
    verdicts = _verify_claims([str(c["text"]) for c in raw_claims], ctx)
    claims = []
    contested = 0
    for i, cl in enumerate(raw_claims):
        stance = str(cl.get("stance", "supports"))
        ok = verdicts[i] if i < len(verdicts) else True
        status = "active" if ok else "candidate"
        if stance == "contradicts" and ok:
            status = "contested"
            contested += 1
        claims.append({"kind": "claim", "content": str(cl["text"]),
                       "source": "memory_card", "status": status,
                       "rel": {"contradicts": "contradicts",
                               "extends": "extends"}.get(stance, "supports")})
    res = add_notes(paper_id, "memory_card", claims) if claims else {"added": 0, "linked": 0}
    # 矛盾闭环: 与新主张相冲突的旧笔记也标 contested (排除自身, 找真正的对方)
    if contested:
        for cl in claims:
            if cl.get("status") == "contested":
                other = _find_conflict(cl["content"])
                if other:
                    _set_status(other["id"], "contested")
    return {"ok": True, "aliases_added": n_alias, "claims": res,
            "verified": sum(1 for v in verdicts if v), "contested": contested,
            "candidates": sum(1 for c in claims if c["status"] == "candidate")}


def _verify_claims(texts: List[str], ctx: str) -> List[bool]:
    """主张验证: 有模型→批量 1 次调用; 否则词法重合。返回与 texts 等长的布尔列表。
    兼容模型返回 1-based 序号 (否则判定整体错位); 结构完整时即使全 False 也采信模型。"""
    if not texts:
        return []
    try:
        from .llm import verify_claims
        from .llm_ops import _extract_json
        raw = verify_claims(texts, ctx)
        obj = _extract_json(raw)
        if isinstance(obj, list):
            idx: Dict[int, bool] = {}
            for o in obj:
                if isinstance(o, dict):
                    try:
                        idx[int(o.get("i", -1))] = bool(o.get("ok"))
                    except Exception:
                        continue
            offset = 0
            n = len(texts)
            if all(i in idx for i in range(n)):
                pass  # 0-based 完整
            elif all((i + 1) in idx for i in range(n)):
                offset = 1  # 1-based 完整
            else:
                idx = {}  # 不完整: 不做偏移推断 (缺失条目可能被错误左移), 回退词法
            if idx:
                return [idx.get(i + offset, False) for i in range(n)]
    except Exception:
        pass
    # 词法门: 主张关键词在原文的重合率
    return [_lexical_support(t, ctx) for t in texts]


def _lexical_support(text: str, ctx: str) -> bool:
    kws = set(k.lower() for k in _keywords(text, 8))
    if not kws:
        return False  # 无可核对关键词 → 保守判不通过 (不无条件放行短断言)
    low = (ctx or "").lower()
    hit = sum(1 for k in kws if k in low)
    return hit / max(1, len(kws)) >= 0.34


def _set_status(note_id: int, status: str) -> None:
    con = _conn()
    try:
        con.execute("UPDATE notes SET status=? WHERE id=?", (status, note_id))
        if status not in RETRIEVABLE:
            con.execute("DELETE FROM vectors WHERE owner_type='note' AND owner_id=?", (note_id,))
        con.commit()
    except Exception:
        pass
    con.close()
    if status not in RETRIEVABLE:
        try:
            from .vector_store import delete_ids
            delete_ids([note_id])
        except Exception:
            pass


# ------------------------------------------------------------ 向量与画像
def embed_notes(paper_id: Optional[str] = None, batch: int = 32) -> Dict:
    """把笔记批量向量化入库 (内容哈希幂等)。失败返回 0, 不阻塞。"""
    from . import embeddings
    if not embeddings.available():
        return {"embedded": 0, "reason": "embed 未配置"}
    con = _conn()
    con.row_factory = sqlite3.Row
    if paper_id:
        rows = con.execute(
            f"SELECT id, content, keywords FROM notes WHERE paper_id=? "
            f"AND kind!='entity' AND {_status_sql('notes')}",
                           (paper_id,)).fetchall()
    else:
        rows = con.execute(
            f"SELECT id, content, keywords FROM notes WHERE kind!='entity' "
            f"AND {_status_sql('notes')}").fetchall()
    m = embeddings.storage_model()
    existing = {}
    for r in con.execute(
            "SELECT owner_id, hash, dim, vec FROM vectors "
            "WHERE owner_type='note' AND model=?", (m,)):
        existing[r[0]] = (r[1], int(r[2]), r[3])
    try:
        from . import vector_store
        vss_ready = vector_store.available()
        indexed_hashes = vector_store.content_hashes(m) if vss_ready else {}
    except Exception:
        vector_store, vss_ready, indexed_hashes = None, False, {}
    todo, texts, hashes = [], [], []
    index_rows = []
    for r in rows:
        try:
            kws = json.loads(r["keywords"] or "[]")
        except Exception:
            kws = []
        text = (r["content"] or "") + " " + " ".join(str(k) for k in kws)
        h = embeddings.content_hash(text)
        cached = existing.get(r["id"])
        if cached and cached[0] == h:
            if vss_ready and indexed_hashes.get((int(r["id"]), cached[1])) != h:
                try:
                    vector = embeddings.from_blob(cached[2])
                    if len(vector) == cached[1]:
                        index_rows.append((int(r["id"]), h, vector))
                except Exception:
                    pass
            continue
        todo.append(r["id"])
        texts.append(text)
        hashes.append(h)
    if not todo:
        con.close()
        if index_rows and vector_store is not None:
            indexed = vector_store.upsert(m, index_rows)
            return {"embedded": 0, "pending": 0, "reason": "缓存已最新，已修复向量索引",
                    "vector_index": indexed}
        return {"embedded": 0, "reason": "已最新"}
    vecs = embeddings.embed_texts(texts, batch=batch)
    n = 0
    for nid, h, v in zip(todo, hashes, vecs):
        if not v:
            continue
        try:
            con.execute("REPLACE INTO vectors VALUES(?,?,?,?,?,?)",
                        ("note", nid, m, len(v), h, embeddings.to_blob(v)))
            index_rows.append((nid, h, v))
            n += 1
        except Exception:
            pass
    con.commit()
    con.close()
    try:
        from . import vector_store
        indexed = vector_store.upsert(m, index_rows)
    except Exception as exc:
        indexed = {"ok": False, "indexed": 0, "reason": str(exc)[:120]}
    return {"embedded": n, "pending": len(todo) - n, "vector_index": indexed}


def bump_preference(term: str, delta: float = 0.3, cap: float = 3.0) -> Dict:
    t = _clean(str(term))[:60]
    if not t or len(t) < 2:
        return {"ok": False}
    con = _conn()
    row = con.execute("SELECT weight FROM preferences WHERE term=?", (t,)).fetchone()
    w = min(cap, (row[0] if row else 1.0) + delta)
    con.execute("REPLACE INTO preferences VALUES(?,?,?)", (t, w, time.time()))
    con.commit()
    con.close()
    return {"ok": True, "term": t, "weight": round(w, 2)}


def list_preferences() -> List[Dict]:
    con = _conn()
    rows = con.execute("SELECT term, weight, updated FROM preferences ORDER BY weight DESC").fetchall()
    con.close()
    return [{"term": r[0], "weight": round(r[1], 2), "updated": r[2]} for r in rows]


def delete_preference(term: str) -> Dict:
    con = _conn()
    con.execute("DELETE FROM preferences WHERE term=?", (_clean(term),))
    con.commit()
    con.close()
    return {"ok": True}


def decay_preferences(half_life_days: float = 30.0) -> Dict:
    now = time.time()
    con = _conn()
    n = 0
    for term, w, upd in con.execute("SELECT term, weight, updated FROM preferences").fetchall():
        days = max(0.0, (now - (upd or now)) / 86400.0)
        w2 = w * (0.5 ** (days / half_life_days))
        con.execute("UPDATE preferences SET weight=? WHERE term=?", (w2, term))
        n += 1
    con.commit()
    con.close()
    return {"ok": True, "decayed": n}


def pending_decisions(kind: str = "candidate") -> List[Dict]:
    """待裁决队列: candidate(未验证) 或 contested(有争议)。"""
    con = _conn()
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM notes WHERE status=? ORDER BY created_at DESC LIMIT 100",
                       (kind,)).fetchall()
    con.close()
    return [_note_dict(r) for r in rows]


def decide_note(note_id: int, accept: bool) -> Dict:
    _set_status(note_id, "active" if accept else "rejected")
    return {"ok": True, "id": note_id, "status": "active" if accept else "rejected"}
