"""批量处理多篇文献 (v5.0): 串行 + 断点续跑 + 逐篇沉淀记忆 + 台账。

为什么串行: 语义调用走本机 opencode 会话 (全局复用) 与远端 embedding, 并发会互相污染/打满;
需要并发时用 workers>1 仅对"不调模型"的离线档 (fast + use_llm=False) 安全。

对外接口:
- run_batch(items, out_root, ...)  核心; items=[{path, paper_id?, name?}]
- collect_files(paths=[], dirs=[], patterns=...)  从路径/目录收集输入
"""
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import config


MARKER_SCHEMA = "paperbrain-task-v5.1"
_CODE_FINGERPRINT: Optional[str] = None
_CODE_FINGERPRINT_KEY = None


def _sha256_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return ""


def _pipeline_fingerprint() -> str:
    """Hash executable pipeline sources and calibration records.

    Resume is a result cache. Accepting artifacts produced by different code or
    thresholds silently bypasses current gates, so the cache key must bind both.
    """
    global _CODE_FINGERPRINT, _CODE_FINGERPRINT_KEY
    root = Path(__file__).resolve().parent.parent
    files = sorted((root / "paperbrain").glob("*.py"))
    files += [p for p in (root / "thresholds.json", root / "thresholds_nli.json",
                          config.graph_policy_file())
              if p.is_file()]
    # Long-running server processes must notice policy/threshold/code changes without
    # re-hashing every source for every paper in a 50-paper batch.
    fingerprint_key = tuple(
        (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size) for path in files)
    if _CODE_FINGERPRINT is not None and _CODE_FINGERPRINT_KEY == fingerprint_key:
        return _CODE_FINGERPRINT
    h = hashlib.sha256()
    for path in files:
        try:
            identity = str(path.relative_to(root))
        except ValueError:
            identity = str(path.resolve())
        h.update(identity.encode("utf-8"))
        h.update(bytes.fromhex(_sha256_file(path)))
    _CODE_FINGERPRINT = h.hexdigest()
    _CODE_FINGERPRINT_KEY = fingerprint_key
    return _CODE_FINGERPRINT


def _run_signature(source_sha: str, task: str, depth: str, focus: str,
                   vision: bool, scorer_threshold: float, use_llm: bool) -> str:
    """Build a secret-free signature for every option that can change output."""
    payload = {
        "schema": MARKER_SCHEMA,
        "source_sha256": source_sha,
        "pipeline_sha256": _pipeline_fingerprint(),
        "task": task,
        "depth": depth,
        "focus": focus,
        "vision": bool(vision),
        "use_llm": bool(use_llm),
        "scorer_threshold": float(scorer_threshold),
        "provider": config.provider() if use_llm else "off",
        "model": config.model() if use_llm else "off",
        "vl_model": config.vl_model() if vision else "off",
        "all_model": config.all_model() if use_llm else False,
        "max_images": config.max_images(),
        "context_chars": config.context_chars(),
        "n_questions": config.n_questions(),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _artifact_signature(out_dir: Path) -> str:
    """Hash all per-paper artifacts except the self-referential completion marker."""
    if not out_dir.is_dir():
        return ""
    h = hashlib.sha256()
    files = sorted(p for p in out_dir.rglob("*")
                   if p.is_file() and p.name != "task_done.json")
    if not files:
        return ""
    for path in files:
        h.update(str(path.relative_to(out_dir)).encode("utf-8"))
        digest = _sha256_file(path)
        if not digest:
            return ""
        h.update(bytes.fromhex(digest))
    return h.hexdigest()


def derive_paper_id(path: str, fallback: str = "paper") -> str:
    """文件名 -> PaperID (安全化: 剔除 .. 与首尾点, 防目录穿越)。"""
    stem = Path(path).stem or fallback
    pid = re.sub(r"[^\w\-.]+", "_", stem)
    while ".." in pid:
        pid = pid.replace("..", "_")
    pid = pid.strip("._")
    return (pid or fallback)[:64]


def collect_files(paths: Optional[List[str]] = None,
                  dirs: Optional[List[str]] = None,
                  pattern: str = "*.pdf") -> List[str]:
    """从显式路径 + 目录 glob 收集输入 (去重保序)。"""
    out: List[str] = []
    for p in (paths or []):
        p = str(p).strip()
        if p and Path(p).expanduser().is_file():
            out.append(str(Path(p).expanduser()))
    for d in (dirs or []):
        d = Path(str(d)).expanduser()
        if d.is_dir():
            for f in sorted(d.glob(pattern)):
                if f.is_file():
                    out.append(str(f))
    seen, uniq = set(), []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


def _already_done(out_dir: Path, task: str = "", depth: str = "",
                  source_sha: str = "", run_signature: str = "") -> bool:
    """Only reuse an artifact created by the exact current input/code/options.

    Unversioned legacy outputs are deliberately stale: accepting them can reuse
    pre-calibration CLEAN results and bypass current production gates.
    """
    marker = out_dir / "task_done.json"
    if marker.exists():
        try:
            d = json.loads(marker.read_text(encoding="utf-8"))
            return bool(
                d.get("marker_schema") == MARKER_SCHEMA and source_sha and run_signature and
                d.get("source_sha256") == source_sha and
                d.get("run_signature") == run_signature and
                d.get("artifact_signature") == _artifact_signature(out_dir) and
                (not task or (d.get("task") == task and d.get("depth") == depth))
            )
        except Exception:
            return False
    return False


def _read_marker(out_dir: Path, task: str, depth: str) -> Dict:
    marker = out_dir / "task_done.json"
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("task") == task and data.get("depth") == depth:
            return data
    except Exception:
        pass
    return {}


def _write_marker(out_dir: Path, task: str, depth: str, source_sha: str,
                  run_signature: str, result: Optional[Dict] = None) -> None:
    try:
        result = result or {}
        artifact_signature = _artifact_signature(out_dir)
        if not artifact_signature:
            return
        (out_dir / "task_done.json").write_text(
            json.dumps({"marker_schema": MARKER_SCHEMA,
                        "source_sha256": source_sha,
                        "run_signature": run_signature,
                        "artifact_signature": artifact_signature,
                        "task": task, "depth": depth, "at": time.time(),
                        "verify": result.get("verify"),
                        "is_clean": result.get("is_clean"),
                        "budget_ok": result.get("budget_ok"),
                        "quality_ok": result.get("quality_ok"),
                        "ledger": result.get("ledger") or {}},
                       ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def run_batch(items: List[Dict], out_root: str,
              task: str = "full_read", depth: str = "standard",
              focus: str = "", vision: bool = False,
              scorer_threshold: float = 0.82, resume: bool = True,
              use_llm: bool = True,
              on_progress: Optional[Callable[[int, int, Dict], None]] = None) -> Dict:
    """逐篇执行 run_task, 并 finalize_memory 沉淀 (notes/concepts/embeddings/跨论文边)。

    items: [{path: str, paper_id?: str, name?: str}]
    返回 {total, done, failed, skipped, results:[{file,paper_id,verify,notes,concepts,elapsed}]}
    并写 out_root/batch_ledger.csv。
    """
    from .tasks import run_task
    from .memory_store import finalize_memory, suppress_field_reflection, reflect_global

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    normalized_items = [it if isinstance(it, dict) else {"path": str(it)} for it in items]
    base_ids = [str(it.get("paper_id") or "").strip() or derive_paper_id(str(it.get("path", "")))
                for it in normalized_items]
    counts: Dict[str, int] = {}
    for pid in base_ids:
        counts[pid] = counts.get(pid, 0) + 1
    resolved_ids: List[str] = []
    used = set()
    for idx, (it, base_pid) in enumerate(zip(normalized_items, base_ids)):
        pid = base_pid
        if counts[base_pid] > 1:
            src_key = str(Path(str(it.get("path", ""))).expanduser().resolve(strict=False))
            suffix = hashlib.sha256(src_key.encode("utf-8")).hexdigest()[:8]
            pid = f"{base_pid[:55]}_{suffix}"
        if pid in used:
            pid = f"{pid[:58]}_{idx + 1:05d}"
        used.add(pid)
        resolved_ids.append(pid)
    items = normalized_items
    total = len(items)
    results: List[Dict] = []
    privacy_ledger: List[Dict] = []
    done = failed = skipped = 0
    quality_passed = quality_failed = quality_unverified = 0
    suppress_field_reflection(True)  # 批量中段不触发领域反思, 收尾统一一次

    try:
        for i, it in enumerate(items):
            src = str(it.get("path", ""))
            pid = resolved_ids[i]
            name = Path(src).name or pid
            od = root / pid
            source_sha = _sha256_file(Path(src).expanduser())
            run_signature = _run_signature(
                source_sha, task, depth, focus, vision, scorer_threshold, use_llm)
            t0 = time.time()
            row = {"file": name, "paper_id": pid, "verify": "", "notes": 0, "concepts": 0,
                   "elapsed": 0.0, "error": "", "is_clean": None,
                   "budget_ok": None, "quality_ok": None}
            if resume and od.exists() and _already_done(
                    od, task, depth, source_sha, run_signature):
                marker = _read_marker(od, task, depth)
                row.update({"verify": "SKIPPED", "elapsed": 0.0,
                            "is_clean": marker.get("is_clean"),
                            "budget_ok": marker.get("budget_ok"),
                            "quality_ok": marker.get("quality_ok")})
                skipped += 1
                if row["quality_ok"] is True:
                    quality_passed += 1
                elif row["quality_ok"] is False:
                    quality_failed += 1
                else:
                    quality_unverified += 1
                results.append(row)
                cached_usage = marker.get("ledger") if isinstance(marker.get("ledger"), dict) else {}
                privacy_ledger.append(_privacy_ledger_row(source_sha, run_signature, cached_usage))
                if on_progress:
                    on_progress(i + 1, total, row)
                continue
            # 处理中: 先上报 RUNNING, 让界面在长任务期间也有反馈
            if on_progress:
                try:
                    on_progress(i + 1, total, dict(row, verify="RUNNING"))
                except Exception:
                    pass
            try:
                r = run_task(task, src, pid, str(od), use_llm=use_llm, vision=vision,
                             scorer_threshold=scorer_threshold, focus=focus, depth=depth)
                row["verify"] = str(r.get("verify") or r.get("task") or "OK")
                row["is_clean"] = r.get("is_clean")
                row["budget_ok"] = r.get("budget_ok")
                row_ledger = r.get("ledger") if isinstance(r.get("ledger"), dict) else {}
                if row["is_clean"] is None:
                    row["quality_ok"] = None
                    quality_unverified += 1
                else:
                    row["quality_ok"] = bool(row["is_clean"] and row["budget_ok"])
                    if row["quality_ok"]:
                        quality_passed += 1
                    else:
                        quality_failed += 1
                try:
                    fin = finalize_memory(str(od), pid, task)
                    row["notes"] = int((fin.get("notes") or {}).get("added", 0) or 0)
                    row["concepts"] = int((fin.get("concepts") or {}).get("concepts", 0) or 0)
                    if not fin.get("memory") and not row["notes"]:
                        row["error"] = "记忆沉淀部分失败"
                except Exception as e:
                    row["error"] = f"记忆沉淀失败: {e}"[:120]
                done += 1
            except Exception as e:
                row["verify"] = "ERROR"
                row["error"] = str(e)[:200]
                failed += 1
            row["elapsed"] = round(time.time() - t0, 1)
            if row["verify"] != "ERROR":
                marker_row = dict(row, ledger=row_ledger if 'row_ledger' in locals() else {})
                _write_marker(od, task, depth, source_sha, run_signature, marker_row)
                # 连同质量状态和完整缓存身份记录，续跑不得丢失或绕过门禁结果。
            results.append(row)
            privacy_ledger.append(_privacy_ledger_row(
                source_sha, run_signature, row_ledger if 'row_ledger' in locals() else {}))
            if 'row_ledger' in locals():
                del row_ledger
            if on_progress:
                try:
                    on_progress(i + 1, total, row)
                except Exception:
                    pass

    finally:
        suppress_field_reflection(False)
    field_refl: Dict = {}
    try:
        if config.reflect_enabled() and done:
            field_refl = reflect_global()  # 全库收尾统一反思 (≥2 篇且 ≥4 条才真正执行)
    except Exception as e:
        field_refl = {"ok": False, "reason": str(e)[:120]}

    ledger = root / "batch_ledger.csv"
    with open(ledger, "w", newline="", encoding="utf-8") as f:
        # v5 隐私约束：持久台账只包含不可逆身份哈希、token 与费用，不存文件名/原文/错误文本。
        fields = ["source_sha256", "run_signature", "text_tokens", "vision_tokens",
                  "total", "input_tokens", "output_tokens", "llm_calls", "cost"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(privacy_ledger)
    return {"total": total, "done": done, "failed": failed, "skipped": skipped,
            "quality_passed": quality_passed, "quality_failed": quality_failed,
            "quality_unverified": quality_unverified,
            "results": results, "ledger": str(ledger), "field_reflection": field_refl}


def _privacy_ledger_row(source_sha: str, run_signature: str, usage: Dict) -> Dict:
    """Normalize persisted batch accounting without names, content, or diagnostics."""
    def number(name: str) -> int:
        try:
            return max(0, int(usage.get(name, 0) or 0))
        except (TypeError, ValueError):
            return 0
    text_tokens = number("text_tokens")
    vision_tokens = number("vision_tokens")
    total = number("total") or text_tokens + vision_tokens
    return {"source_sha256": source_sha, "run_signature": run_signature,
            "text_tokens": text_tokens, "vision_tokens": vision_tokens, "total": total,
            "input_tokens": number("input_tokens"), "output_tokens": number("output_tokens"),
            "llm_calls": number("llm_calls"), "cost": str(usage.get("cost", "n/a"))[:40]}
