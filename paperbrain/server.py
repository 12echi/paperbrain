"""本地操控台 (stdlib only, 零依赖).

启动: PYTHONPATH=. python3 -m paperbrain.server --port 8000
打开: http://127.0.0.1:8000
功能: 粘贴论文txt -> 一键运行最小闭环 -> 看 report/draft/verify/ledger/outline.
"""
import argparse
import hashlib
import json
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

from . import config

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "out" / "web"

# 路径白名单: 只允许读取 用户目录/临时目录/挂载卷, 防任意文件读取
_ALLOWED_ROOTS = [Path.home(), Path("/tmp"), Path("/private/tmp"), Path("/Volumes")]


def _path_allowed(p: Path) -> bool:
    try:
        rp = p.resolve()
    except Exception:
        return False
    for root in _ALLOWED_ROOTS:
        try:
            rp.relative_to(root.resolve())
            return True
        except Exception:
            continue
    return False


def _safe_pid(pid: str) -> str:
    """PaperID 安全化: 只留单词字符/短横/点, 剔除 .. 与首尾点号 (防路径穿越)。"""
    p = re.sub(r"[^\w\-.]", "_", str(pid or "").strip())[:64]
    while ".." in p:
        p = p.replace("..", "_")
    p = p.strip("._")
    return p or "paper"


def _under(child: Path, root: Path) -> bool:
    """纵深防御: 解析后必须仍在 root 之内。"""
    try:
        child.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False

# 异步任务表: job_id -> {stage, done, ok, result/error, paper_id, t0}
JOBS: dict = {}
JOBS_LOCK = threading.Lock()
_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=config.job_workers(),
                                   thread_name_prefix="paperbrain-job")
_SUBMIT_LOCK = threading.Lock()
_SUBMITTED = 0


def _submit_job(jid: str, fn, *args) -> bool:
    """有界提交：运行中+排队任务超过上限立即拒绝，避免每请求创建一个线程。"""
    global _SUBMITTED
    with _SUBMIT_LOCK:
        if _SUBMITTED >= config.job_queue_limit():
            _job_set(jid, stage="拒绝", done=True, ok=False, error="任务队列已满")
            return False
        _SUBMITTED += 1

    def wrapped():
        global _SUBMITTED
        try:
            fn(*args)
        finally:
            with _SUBMIT_LOCK:
                _SUBMITTED = max(0, _SUBMITTED - 1)

    try:
        _JOB_EXECUTOR.submit(wrapped)
        return True
    except Exception:
        with _SUBMIT_LOCK:
            _SUBMITTED = max(0, _SUBMITTED - 1)
        _job_set(jid, stage="失败", done=True, ok=False, error="任务提交失败")
        return False


def _job_new(pid: str) -> str:
    _jobs_cleanup()
    jid = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[jid] = {"stage": "排队", "done": False, "ok": False,
                     "paper_id": pid, "t0": time.time()}
    return jid


def _jobs_cleanup(ttl_done: float = 3600.0, max_jobs: int = 200) -> int:
    """JOB 表自清理: 完成超 1 小时或超量时淘汰最旧 (防内存无界增长)。"""
    now = time.time()
    dropped = 0
    with JOBS_LOCK:
        for k, v in list(JOBS.items()):
            if v.get("done") and (now - v.get("t0", now)) > ttl_done:
                JOBS.pop(k, None)
                dropped += 1
        if len(JOBS) > max_jobs:
            for k in sorted(JOBS, key=lambda x: JOBS[x].get("t0", 0))[:len(JOBS) - max_jobs]:
                JOBS.pop(k, None)
                dropped += 1
    return dropped


def _job_set(jid: str, **kw):
    with JOBS_LOCK:
        if jid in JOBS:
            JOBS[jid].update(kw)


def _job_dir(pid: str, jid: Optional[str] = None) -> str:
    """隔离每个任务的工作目录: 包含 jid 时写入 OUT_ROOT / f"{pid}_{jid}", 杜绝多会话/并发任务碰撞."""
    if jid:
        d = OUT_ROOT / f"{pid}_{jid}"
    else:
        d = OUT_ROOT / pid
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _run_full_job(jid: str, src: str, pid: str, th: float, vision: bool):
    """一键全跑任务: outline 阶段 -> 生成验真, 分阶段上报进度."""
    import sys as _sys
    import time as _t
    import traceback as _tb
    from .pipeline import run_outline, generate_from_outline, set_trace
    t0 = _t.time()
    worker_id = threading.get_ident()
    job_dir = _job_dir(pid, jid)
    _job_set(jid, out_dir=job_dir)

    def _watchdog():
        _t.sleep(45)
        with JOBS_LOCK:
            j = JOBS.get(jid)
            if not j or j.get("done"):
                return
            fr = _sys._current_frames().get(worker_id)
            if fr is not None:
                j["watchdog"] = "".join(_tb.format_stack(fr)[-12:])

    threading.Thread(target=_watchdog, daemon=True).start()

    def _tr(name: str):
        with JOBS_LOCK:
            if jid in JOBS:
                JOBS[jid].setdefault("trace", []).append(
                    {"step": name, "at": round(_t.time() - t0, 1)})

    set_trace(_tr)
    try:
        _job_set(jid, stage="预处理·分章·纪要·图谱·大纲")
        o = run_outline(src, pid, job_dir, vision=vision, trace_fn=_tr)
        _job_set(jid, stage="草稿生成·一致性·事实门禁")
        r = generate_from_outline(job_dir, pid, o["outline"], scorer_threshold=th, trace_fn=_tr)
        mem = {}
        try:  # 与其他入口一致: 每次运行都沉淀记忆 (修复 /api/run 不写知识库)
            from .memory_store import finalize_memory
            mem = finalize_memory(job_dir, pid, "full_read")
        except Exception:
            pass
        # 兼容旧路径读取: 若 pid 非 demo_paper, 同步产物到 OUT_ROOT / pid
        if pid != "demo_paper":
            try:
                legacy_d = OUT_ROOT / pid
                legacy_d.mkdir(parents=True, exist_ok=True)
                for f in Path(job_dir).iterdir():
                    if f.is_file():
                        shutil.copy2(f, legacy_d / f.name)
            except Exception:
                pass
        quality_ok = bool(r.get("is_clean")) and bool(r.get("budget_ok"))
        _job_set(jid, stage="完成" if quality_ok else "完成·质量门禁未通过",
                 done=True, ok=True, quality_ok=quality_ok, result={
            "paper_id": pid, "verify": r["verify"], "is_clean": r["is_clean"],
            "budget_ok": r["budget_ok"], "ledger": r["ledger"],
            "model": r.get("model", "?"), "memory_saved": bool(mem),
            "notes": mem.get("notes"), "concepts": mem.get("concepts")})
    except Exception as e:
        traceback.print_exc()
        _job_set(jid, stage="失败", done=True, ok=False, error=str(e)[:300])
    finally:
        try:
            from .pipeline import set_trace as _st
            _st(None)
        except Exception:
            pass


def _run_batch_job(jid: str, items: List[dict], task: str, depth: str, focus: str,
                   vision: bool, th: float, resume: bool):
    """批量任务: 串行逐篇 run_task + finalize_memory, 逐篇上报进度。"""
    from .batch import run_batch
    total = len(items)

    def prog(i: int, n: int, row: dict):
        with JOBS_LOCK:
            j = JOBS.get(jid)
            if j is None:
                return
            files = j.setdefault("files", [])
            while len(files) < i:
                files.append({})
            files[i - 1] = dict(row)
            j["stage"] = f"{i}/{n} {row.get('paper_id','')} {row.get('verify','')}".strip()
            j["progress"] = {"done": i, "total": n}

    try:
        _job_set(jid, stage=f"0/{total} 开始", files=[], progress={"done": 0, "total": total})
        res = run_batch(items, str(OUT_ROOT), task=task, depth=depth, focus=focus,
                        vision=vision, scorer_threshold=th, resume=resume,
                        on_progress=prog)
        quality_ok = res["quality_failed"] == 0 and res["quality_unverified"] == 0
        if res["failed"]:
            stage = f"完成 {res['done']} 执行 / {res['failed']} 失败 / {res['skipped']} 跳过"
        elif quality_ok:
            stage = f"完成 {res['quality_passed']} 篇，质量门禁全过"
        else:
            stage = (f"完成·质量门禁未通过 {res['quality_failed']} 篇 / "
                     f"未运行门禁 {res['quality_unverified']} 篇")
        _job_set(jid, stage=stage, done=True, ok=res["failed"] == 0,
                 quality_ok=quality_ok, result=res)
    except Exception as e:
        traceback.print_exc()
        _job_set(jid, stage="失败", done=True, ok=False, error=str(e)[:300])


def _run_generate_job(jid: str, pid: str, th: float, outline=None, draft=None,
                      use_session_draft=False, run_deepread: bool = False,
                      focus: str = ""):
    """只生成任务: 用已确认大纲/外部草稿验真."""
    from .pipeline import generate_from_outline
    job_dir = _job_dir(pid, jid)
    _job_set(jid, out_dir=job_dir)
    legacy_d = OUT_ROOT / pid
    if legacy_d.exists():
        version_files = list(legacy_d.glob("outline_v*.json")) if legacy_d.exists() else []
        for req_f in ("state.json", "memory.json", "draft_session.md", "outline_history.json"):
            src_f = legacy_d / req_f
            dst_f = Path(job_dir) / req_f
            if src_f.exists() and not dst_f.exists():
                try:
                    shutil.copy2(src_f, dst_f)
                except Exception:
                    pass
        for src_f in version_files:
            dst_f = Path(job_dir) / src_f.name
            if not dst_f.exists():
                try:
                    shutil.copy2(src_f, dst_f)
                except Exception:
                    pass
    try:
        deep_result = {}
        if run_deepread:
            _job_set(jid, stage="确认后深读·多视角拷问")
            from .deepread import build as build_deepread
            deep_result = build_deepread(job_dir, pid, use_llm=True, focus=focus)
        _job_set(jid, stage="草稿生成·一致性·事实门禁")
        if use_session_draft and draft is None:
            fp = Path(job_dir) / "draft_session.md"
            if not fp.exists():
                raise RuntimeError("会话草稿不存在: 请先在对话中让我写好草稿")
            draft = fp.read_text(encoding="utf-8")
        r = generate_from_outline(job_dir, pid, outline,
                                  scorer_threshold=th, draft_override=draft)
        mem = {}
        try:  # 与其他入口一致: 每次运行都沉淀记忆
            from .memory_store import finalize_memory
            mem = finalize_memory(job_dir, pid, "full_read")
        except Exception:
            pass
        if pid != "demo_paper":
            try:
                for f in Path(job_dir).iterdir():
                    if f.is_file():
                        shutil.copy2(f, legacy_d / f.name)
            except Exception:
                pass
        quality_ok = bool(r.get("is_clean")) and bool(r.get("budget_ok"))
        markdown = ""
        try:
            markdown = (Path(job_dir) / "draft.md").read_text(encoding="utf-8")
        except Exception:
            markdown = deep_result.get("markdown", "")
        _job_set(jid, stage="完成" if quality_ok else "完成·质量门禁未通过",
                 done=True, ok=True, quality_ok=quality_ok, result={
            "paper_id": pid, "verify": r["verify"], "is_clean": r["is_clean"],
            "budget_ok": r["budget_ok"], "ledger": r["ledger"],
            "markdown": markdown,
            "model": r.get("model", "?"), "memory_saved": bool(mem),
            "notes": mem.get("notes"), "concepts": mem.get("concepts")})
    except Exception as e:
        traceback.print_exc()
        _job_set(jid, stage="失败", done=True, ok=False, error=str(e)[:300])


def _run_task_job(jid: str, task: str, src: str, pid: str, th: float, vision: bool,
                  focus: str = "", depth: str = "standard"):
    """按解读类别执行, 完成后自动沉淀学习记忆."""
    from .tasks import run_task
    job_dir = _job_dir(pid, jid)
    _job_set(jid, out_dir=job_dir)
    try:
        _job_set(jid, stage="开始")
        res = run_task(task, src, pid, job_dir, use_llm=True, vision=vision,
                       scorer_threshold=th, focus=focus, depth=depth,
                       progress=lambda s: _job_set(jid, stage=s))
        try:
            from .memory_store import finalize_memory
            fin = finalize_memory(job_dir, pid, task)
            res["memory_saved"] = "memory_error" not in fin
            res.update({"notes": fin.get("notes"), "concepts": fin.get("concepts"),
                        "paper_edges": fin.get("paper_edges")})
            if "memory_error" in fin:
                res["memory_error"] = fin["memory_error"]
        except Exception as e:
            res["memory_saved"] = False
            res["memory_error"] = str(e)[:120]
        if pid != "demo_paper":
            try:
                legacy_d = OUT_ROOT / pid
                legacy_d.mkdir(parents=True, exist_ok=True)
                for f in Path(job_dir).iterdir():
                    if f.is_file():
                        shutil.copy2(f, legacy_d / f.name)
            except Exception:
                pass
        if res.get("is_clean") is None:
            quality_ok = None
            final_stage = "完成·质量门禁未运行"
        else:
            quality_ok = bool(res.get("is_clean")) and bool(res.get("budget_ok"))
            final_stage = "完成" if quality_ok else "完成·质量门禁未通过"
        _job_set(jid, stage=final_stage, done=True, ok=True,
                 quality_ok=quality_ok, result=res)
    except Exception as e:
        traceback.print_exc()
        _job_set(jid, stage="失败", done=True, ok=False, error=str(e)[:300])

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PaperBrain · 文献精读工作台</title>
<style>
:root{
  --bg:#f5f7fb; --card:#ffffff; --ink:#0b1220; --ink2:#334155; --mut:#64748b; --mut2:#94a3b8;
  --line:#e5eaf1; --line2:#eef2f6; --acc:#2563eb; --acc-soft:#eff6ff; --acc-ring:rgba(37,99,235,.14);
  --ok:#047857; --ok-soft:#ecfdf5; --warn:#b45309; --warn-soft:#fffbeb; --bad:#b91c1c; --bad-soft:#fef2f2;
  --r:16px; --r-sm:10px;
  --sh:0 1px 2px rgba(16,24,40,.04), 0 10px 28px -14px rgba(16,24,40,.12);
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font-size:14px;line-height:1.62;
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
body{background-image:radial-gradient(circle at 12% -8%,rgba(37,99,235,.08),transparent 28%),radial-gradient(circle at 90% 8%,rgba(14,165,233,.06),transparent 24%)}
a{color:var(--acc);text-decoration:none}
.top{position:sticky;top:0;z-index:30;background:rgba(255,255,255,.82);backdrop-filter:saturate(180%) blur(12px);border-bottom:1px solid var(--line)}
.topin{max-width:1280px;margin:0 auto;padding:13px 28px;display:flex;align-items:center;gap:12px}
.brand{display:flex;align-items:center;gap:11px}
.logo{width:32px;height:32px;border-radius:9px;background:var(--ink);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;letter-spacing:.3px}
.brand h1{margin:0;font-size:15px;font-weight:650;letter-spacing:-.01em}
.brand span{color:var(--mut);font-weight:400;font-size:12px;margin-left:2px}
.sp{flex:1}
.chip{font-size:12px;padding:5px 11px;border-radius:999px;border:1px solid var(--line);background:#fff;color:var(--mut);white-space:nowrap}
.chip.on{color:var(--acc);border-color:#dcd9fb;background:var(--acc-soft)}
.chip.warn{color:var(--warn);border-color:#fde9c8;background:var(--warn-soft)}
.wrap{max-width:1280px;margin:24px auto 72px;padding:0 28px;display:grid;grid-template-columns:minmax(340px,400px) 1fr;gap:20px;align-items:start}
@media(max-width:1000px){.wrap{grid-template-columns:1fr}}
.col{display:flex;flex-direction:column;gap:18px}
.card{background:rgba(255,255,255,.96);border:1px solid var(--line);border-radius:18px;padding:20px;box-shadow:var(--sh)}
.card h2{margin:0 0 3px;font-size:14.5px;font-weight:650;letter-spacing:-.01em;display:flex;align-items:center;gap:9px}
.step{width:20px;height:20px;border-radius:6px;background:var(--acc-soft);color:var(--acc);border:1px solid #e0e2fb;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700}
.card .sub{color:var(--mut);font-size:12.5px;margin:0 0 15px;line-height:1.55}
label.f{display:block;font-size:12px;color:var(--ink2);margin:13px 0 6px;font-weight:550;letter-spacing:.01em}
input[type=text],input[type=password],select{width:100%;padding:9px 12px;font-size:13.5px;background:#fff;border:1px solid #e2e6ec;border-radius:var(--r-sm);color:var(--ink);outline:none;transition:border-color .15s,box-shadow .15s}
input::placeholder{color:var(--mut2)}
input:focus{border-color:var(--acc);box-shadow:0 0 0 3px var(--acc-ring)}
.btn{font-size:13.5px;border-radius:var(--r-sm);padding:9px 14px;cursor:pointer;border:1px solid #e2e6ec;background:#fff;color:var(--ink);transition:.15s;font-weight:550}
.btn:hover{border-color:#cfd6df;background:#fcfcfd}
.btn.primary{background:var(--ink);border-color:var(--ink);color:#fff;font-weight:600}
.btn.primary:hover{background:#1e293b;border-color:#1e293b}
.btn.ghost{background:#fff}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btnrow{display:flex;gap:9px;flex-wrap:wrap;margin-top:14px}
.btnrow .btn{flex:1}
.drop{border:1.5px dashed #cbd5e1;border-radius:14px;padding:24px 16px;text-align:center;color:var(--mut);cursor:pointer;transition:.18s;background:linear-gradient(145deg,#fbfdff,#f8fafc)}
.drop:hover,.drop.hot{border-color:var(--acc);color:var(--acc);background:var(--acc-soft)}
.drop .big{font-size:13.5px;color:var(--ink);margin-bottom:2px;font-weight:600}
.filetag{display:none;margin-top:11px;padding:9px 12px;border:1px solid #e2e6ec;border-radius:var(--r-sm);font-size:12.5px;color:var(--ink2);background:#fbfcfe;word-break:break-all}
.tasks{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media(max-width:520px){.tasks{grid-template-columns:1fr}}
.task{border:1px solid #e6e9ee;border-radius:12px;padding:12px 13px;cursor:pointer;background:#fff;transition:.15s}
.task:hover{border-color:#cfd6df;box-shadow:0 2px 8px -4px rgba(16,24,40,.12)}
.task.sel{border-color:var(--acc);background:var(--acc-soft);box-shadow:0 0 0 1px var(--acc) inset}
.task .t{font-weight:600;font-size:13px;display:flex;align-items:center;gap:8px;color:var(--ink)}
.task .d{color:var(--mut);font-size:11.5px;margin-top:4px;line-height:1.5}
.dot{width:8px;height:8px;border-radius:50%;border:2px solid #d3d9e2;flex:none}
.task.sel .dot{background:var(--acc);border-color:var(--acc)}
.badge{display:inline-block;padding:3px 12px;border-radius:999px;font-size:12px;font-weight:650;letter-spacing:.01em}
.clean{background:var(--ok-soft);color:var(--ok)}
.dirty{background:var(--bad-soft);color:var(--bad)}
.review{background:var(--warn-soft);color:var(--warn)}
.neutral{background:var(--acc-soft);color:var(--acc)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0 4px}
@media(max-width:640px){.stats{grid-template-columns:repeat(2,1fr)}}
.stat{border:1px solid var(--line);border-radius:12px;padding:11px 12px;background:#fbfcfe}
.stat .k{font-size:11px;color:var(--mut)}
.stat .v{font-size:16px;font-weight:700;margin-top:2px;letter-spacing:-.01em}
.tabs{display:flex;gap:4px;flex-wrap:wrap;margin-top:15px;padding:3px;background:#f3f4f6;border-radius:11px;width:fit-content}
.tab{padding:6px 12px;font-size:12.5px;border-radius:8px;border:none;background:transparent;color:var(--mut);cursor:pointer;font-weight:550;transition:.15s}
.tab:hover{color:var(--ink)}
.tab.on{background:#fff;color:var(--ink);box-shadow:0 1px 2px rgba(16,24,40,.08)}
.md{margin-top:12px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;max-height:600px;overflow:auto;font-size:13.5px;line-height:1.75}
.md h1{font-size:17px;margin:4px 0 10px;font-weight:700}
.md h2{font-size:14.5px;margin:16px 0 7px;font-weight:650;color:var(--ink)}
.md h3{font-size:13.5px;margin:12px 0 5px;font-weight:650;color:var(--ink2)}
.md ul{margin:6px 0;padding-left:20px}
.md li{margin:3px 0;color:var(--ink2)}
.md b{color:var(--ink)}
.md p{margin:7px 0;color:var(--ink2)}
pre{margin:12px 0 0;background:#fbfcfe;border:1px solid var(--line);border-radius:12px;padding:15px;overflow:auto;max-height:600px;font-size:12.5px;line-height:1.7;white-space:pre-wrap;word-break:break-word;color:var(--ink2)}
.mut{color:var(--mut)}
.empty{border:1.5px dashed #d3d9e2;border-radius:12px;padding:28px;text-align:center;color:var(--mut);font-size:13px;background:#fbfcfe}
.prog{display:none;margin-top:14px}
.bar{height:6px;background:#eef0f3;border-radius:999px;overflow:hidden}
.bar > i{display:block;height:100%;width:6%;background:var(--acc);border-radius:999px;transition:width .5s}
.mems{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:13px}
@media(max-width:760px){.mems{grid-template-columns:1fr}}
.mem{border:1px solid var(--line);border-radius:12px;padding:14px;background:#fff;transition:.15s}
.mem:hover{border-color:#d7dbe2}
.mem .h{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}
.mem .ttl{font-weight:600;font-size:13px;color:var(--ink)}
.mem .meta{color:var(--mut2);font-size:11px;margin-top:2px}
.mem .bd{color:var(--ink2);font-size:12.5px;margin-top:8px;max-height:130px;overflow:auto}
.mem .x{cursor:pointer;color:var(--mut2);font-size:12px}
.mem .x:hover{color:var(--bad)}
.spin{display:inline-block;width:11px;height:11px;border:2px solid #d7dbe2;border-top-color:var(--acc);border-radius:50%;animation:sp .8s linear infinite;vertical-align:-1px}
@keyframes sp{to{transform:rotate(360deg)}}
.hint{font-size:11.5px;color:var(--mut);margin-top:8px}
.sep{height:1px;background:var(--line);margin:16px 0}
.row{display:flex;gap:9px}
.row > *{min-width:0}
.panel-summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:11px;margin:-2px;padding:2px}
.panel-summary::-webkit-details-marker{display:none}
.panel-summary .meta{margin-left:auto;color:var(--mut);font-size:12px;white-space:nowrap}
.panel-summary:after{content:"›";font-size:22px;color:var(--mut2);transform:rotate(90deg);transition:.18s}
details[open]>.panel-summary:after{transform:rotate(-90deg)}
.advanced-body{padding-top:16px;margin-top:14px;border-top:1px solid var(--line)}
.advanced{margin-top:14px;border:1px solid var(--line);border-radius:12px;background:#fbfcfe}
.advanced>summary{list-style:none;cursor:pointer;padding:11px 13px;color:var(--ink2);font-size:12.5px;font-weight:600;display:flex;align-items:center;justify-content:space-between}
.advanced>summary::-webkit-details-marker{display:none}
.advanced>summary:after{content:"＋";color:var(--mut)}
.advanced[open]>summary:after{content:"−"}
.advanced .inside{padding:0 13px 14px;border-top:1px solid var(--line)}
.section-label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--mut2);margin:0 0 7px}
.hero-drop{padding:30px 16px}
.hero-drop .big{font-size:15px}
.import-row{display:grid;grid-template-columns:1fr 112px;gap:10px;margin-top:12px;align-items:end}
.select-task{appearance:none;background-image:linear-gradient(45deg,transparent 50%,var(--mut) 50%),linear-gradient(135deg,var(--mut) 50%,transparent 50%);background-position:calc(100% - 16px) 50%,calc(100% - 11px) 50%;background-size:5px 5px,5px 5px;background-repeat:no-repeat;padding-right:30px}
.trust-panel{border:1px solid #dbeafe;background:linear-gradient(145deg,#f8fbff,#eff6ff);border-radius:14px;padding:15px}
.trust-head{display:flex;align-items:center;gap:9px;font-weight:650}
.pulse{width:9px;height:9px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 4px rgba(4,120,87,.10)}
.trust-copy{font-size:12px;color:var(--mut);margin-top:5px}
.auto-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:13px}
.auto-stat{background:rgba(255,255,255,.8);border:1px solid rgba(219,234,254,.9);border-radius:10px;padding:10px}
.auto-stat b{display:block;font-size:17px;letter-spacing:-.02em}
.auto-stat span{font-size:10.5px;color:var(--mut)}
.flow{display:flex;align-items:center;gap:5px;margin-top:13px;color:var(--mut);font-size:11px;overflow:hidden}
.flow span{white-space:nowrap}
.flow i{height:1px;background:#bfdbfe;flex:1;min-width:8px}
.askbox{margin-top:16px}
.askbox .row{background:#f8fafc;border:1px solid var(--line);border-radius:12px;padding:5px}
.askbox input{border:0;background:transparent;box-shadow:none}
.artifact-select{width:auto;min-width:118px;padding:6px 28px 6px 10px;font-size:12.5px;border-radius:8px;background-color:transparent}
.result-head{display:flex;align-items:center;justify-content:space-between;gap:12px}
@media(max-width:520px){.import-row{grid-template-columns:1fr}.auto-grid{grid-template-columns:1fr 1fr}.flow{display:none}.chip:nth-last-child(2){display:none}}
</style></head><body>

<header class="top"><div class="topin">
  <div class="brand"><div class="logo">PB</div><div><h1>PaperBrain <span>文献精读工作台</span></h1></div></div>
  <div class="sp"></div>
  <span class="chip" id="cLLM">模型 · 检测中</span>
  <span class="chip" id="cTools">工具 · 检测中</span>
  <span class="chip" id="cMem">记忆 · 0</span>
</div></header>

<main class="wrap">
  <div class="col">

    <details class="card" id="modelPanel">
      <summary class="panel-summary"><span class="step">1</span><strong>模型设置</strong><span class="meta" id="mCompact">检测配置中</span></summary>
      <div class="advanced-body">
      <p class="sub">支持本机 opencode CLI 或 OpenAI 兼容接口。论文内容只有在你明确授权后才会发送。</p>
      <div class="btnrow" style="margin-top:0">
        <button class="btn primary" id="bImport">接入 opencode-go 模型</button>
      </div>
      <label class="f">API Key（可留空，复用本机保存的 opencode 远端服务凭据）</label>
      <input type="password" id="mk" placeholder="sk-..." autocomplete="off"/>
      <label class="f">Base URL</label>
      <input type="text" id="mb" placeholder="https://api.openai.com/v1"/>
      <div class="row">
        <div style="flex:1"><label class="f">文本模型</label><input type="text" id="mm" placeholder="muse-spark-1.3-contributor"/></div>
        <div style="flex:1"><label class="f">视觉模型</label><input type="text" id="mv" placeholder="deepseek-v4-flash-vision-exp"/></div>
      </div>
      <label class="f" style="display:flex;gap:8px;align-items:center;margin-top:13px;font-weight:500">
        <input type="checkbox" id="cloudAllowed" style="width:auto"/> 明确授权向已配置模型端点发送论文文本切块及所选配图
      </label>
      <div class="btnrow">
        <button class="btn" id="bSave">保存配置</button>
        <button class="btn ghost" id="bTest">测试连通</button>
        <button class="btn ghost" id="bClear">清除</button>
      </div>
      <label class="f"><input type="checkbox" id="webVerify"/> 允许将待核查主张发送至 Crossref 学术检索补充证据</label>
      <div class="hint" id="mSt">读取中…</div>
      </div>
    </details>

    <div class="card">
      <h2><span class="step">2</span>文件导入</h2>
      <p class="sub">拖入一篇或多篇文献，系统自动识别单篇与批量任务。</p>
      <div class="drop hero-drop" id="drop">
        <div class="big">拖入文献，开始构建可信知识</div>
        <div>PDF / TXT / Markdown · 支持多选 · 单文件 ≤ 20MB</div>
      </div>
      <input type="file" id="file" accept=".pdf,.txt,.md" multiple style="display:none"/>
      <div class="filetag" id="ftag"></div>
      <div class="import-row">
        <div><label class="f">论文标识</label><input type="text" id="pid" placeholder="自动根据文件名生成"/></div>
        <div class="hint" style="margin:0 0 9px">可手动修改</div>
      </div>
      <details class="advanced">
        <summary>高级导入</summary>
        <div class="inside">
          <label class="f">直接读取本机绝对路径</label>
          <input type="text" id="path" placeholder="/Users/you/Downloads/paper.pdf"/>
          <label class="f">批量文件与文件夹</label>
          <button class="btn" id="dropMany" type="button">选择多篇文献</button>
          <input type="file" id="files" accept=".pdf,.txt,.md" multiple style="display:none"/>
          <div class="filetag" id="ftagMany"></div>
          <textarea id="bpaths" style="height:70px;margin-top:9px;font-family:ui-monospace,Menlo,monospace;font-size:12px" placeholder="每行一个文件路径或文件夹"></textarea>
          <label class="f" style="display:flex;gap:8px;align-items:center;margin-top:13px;font-weight:500">
            <input type="checkbox" id="vis" style="width:auto"/> 启用视觉模型读图（最多 5 张）
          </label>
        </div>
      </details>
    </div>

    <div class="card">
      <h2><span class="step">3</span>解读任务</h2>
      <p class="sub">默认执行全文精读、来源核验与自动学习。</p>
      <div id="tasks"><select class="select-task" id="taskSelect"><option value="full_read">全文精读</option></select></div>
      <details class="advanced">
        <summary>目标与深度</summary>
        <div class="inside">
          <label class="f">学习关注点</label>
          <input type="text" id="focus" placeholder="例如：方法为何有效？复现的关键障碍是什么？"/>
          <label class="f">解读深度</label>
          <select id="depth">
            <option value="fast">快速 · 纪要与大纲</option>
            <option value="standard" selected>标准 · 穿透式深读</option>
            <option value="deep">深度 · 确认大纲后生成与验真</option>
          </select>
        </div>
      </details>
      <div class="btnrow" style="margin-top:16px">
        <button class="btn primary" id="bRun">开始智能解读</button>
        <button class="btn" id="bBatch" title="批量串行处理所选文件/文件夹">批量运行</button>
      </div>
      <div id="outlineConfirm" style="display:none;margin-top:14px">
        <label class="f">待确认大纲（可编辑 JSON；确认后才生成深度草稿）</label>
        <textarea id="outlineEdit" style="height:240px;font-family:ui-monospace,Menlo,monospace;font-size:12px"></textarea>
        <div class="btnrow"><button class="btn primary" id="bConfirmOutline">确认大纲并生成</button></div>
      </div>
      <div class="prog" id="prog"><div class="bar"><i id="pbar"></i></div><div class="hint" id="ptxt"></div></div>
      <div class="mems" id="bfiles"></div>
      <div class="hint" id="runSt"></div>
    </div>
  </div>

  <div class="col">
    <div class="card">
      <div class="result-head"><h2>解读结果</h2><span class="badge neutral">来源可追溯</span></div>
      <div id="sum"><div class="empty">尚未运行 · 左侧完成三步后点“开始解读”</div></div>
      <div class="stats" id="stats" style="display:none">
        <div class="stat"><div class="k">文本 tokens</div><div class="v" id="sTok">–</div></div>
        <div class="stat"><div class="k">预算门禁</div><div class="v" id="sBud">–</div></div>
        <div class="stat"><div class="k">引用数</div><div class="v" id="sCit">–</div></div>
        <div class="stat"><div class="k">图谱实体</div><div class="v" id="sEnt">–</div></div>
      </div>
      <div class="tabs" id="tabs">
        <button class="tab on" data-f="result">精读成果</button>
        <button class="tab" data-f="deepread_full.md">完整分析</button>
        <button class="tab" data-f="verify.json">事实核验</button>
        <select class="artifact-select" id="artifactSelect">
          <option value="">更多产物</option>
          <option value="deepread.md">精读简报</option>
          <option value="report.md">运行报告</option>
          <option value="draft.md">论文草稿</option>
          <option value="outline_v1.json">结构大纲</option>
          <option value="memory.json">知识图谱</option>
        </select>
      </div>
      <div class="md" id="view"><span class="mut">运行后在此查看成果。</span></div>
    </div>

    <div class="card">
      <h2><span class="step">4</span>自动学习记忆</h2>
      <p class="sub">每次解读后自动提取、来源绑定、逐条验真和分层入库。未通过内容会被隔离，不进入可信回答。</p>
      <div class="trust-panel">
        <div class="trust-head"><span class="pulse"></span><span id="autoTrust">自动校验已启用</span></div>
        <div class="trust-copy">可信记忆只接收通过产物级与主张级双重校验的内容</div>
        <div class="auto-grid">
          <div class="auto-stat"><b id="mActive">–</b><span>可信笔记</span></div>
          <div class="auto-stat"><b id="mReview">–</b><span>隔离待复核</span></div>
          <div class="auto-stat"><b id="mPapers">–</b><span>已学习论文</span></div>
        </div>
        <div class="flow"><span>自动提取</span><i></i><span>来源绑定</span><i></i><span>主张验真</span><i></i><span>可信入库</span></div>
      </div>
      <div class="askbox">
        <p class="section-label">向可信记忆提问</p>
        <div class="row">
          <input type="text" id="aq" placeholder="输入问题，回答将标注 [M#] 记忆来源"/>
          <button class="btn primary" id="bAsk" style="white-space:nowrap">提问</button>
        </div>
        <div class="md" id="ans" style="max-height:240px"><span class="mut">回答只使用已验证记忆；证据不足会明确拒答。</span></div>
      </div>
      <details class="advanced" id="memoryAdvanced">
        <summary>高级记忆管理</summary>
        <div class="inside">
          <div class="btnrow">
            <button class="btn" id="bReflect" title="自动流程已默认执行；此按钮用于手动重算">重算本篇洞见</button>
            <button class="btn" id="bField" title="自动流程每新增三篇会执行；此按钮用于手动重算">重算领域洞见</button>
          </div>
          <div class="hint" id="prefs"></div>
          <div class="sep"></div>
          <p class="sub" style="margin:0 0 8px">隔离区：未验证或存在争议的内容</p>
          <div class="mems" id="decisions"><div class="empty" style="grid-column:1/-1">暂无待复核内容</div></div>
          <div class="sep"></div>
          <div class="row"><input type="text" id="mq" placeholder="检索论文记忆与实体"/><button class="btn" id="bSearch">检索</button></div>
          <div class="mems" id="mems"><div class="empty" style="grid-column:1/-1">暂无记忆</div></div>
          <div class="sep"></div>
          <div class="row"><input type="text" id="nq" placeholder="召回原子笔记与关联"/><button class="btn" id="bNotes">召回</button></div>
          <div class="mems" id="notes"><div class="empty" style="grid-column:1/-1">暂无笔记</div></div>
          <div class="sep"></div>
          <p class="sub" style="margin:0 0 8px">跨论文领域网络</p>
          <div id="field"><div class="empty">积累同方向论文后自动形成关联</div></div>
        </div>
      </details>
    </div>
  </div>
</main>

<script>
var API=function(p,o){return fetch(p,o).then(function(r){return r.json();});};
var TASKS=[], lastPid="", lastResult=null, pickedFile=null, pendingFocus="", pidTouched=false;

/* ---------- 1. 模型 ---------- */
function refreshModel(){
  return API("/api/model").then(function(m){
    var c=document.getElementById("cLLM");
    if(m.has_key&&m.cloud_allowed){c.className="chip on";c.textContent="模型 · "+(m.provider==="opencode-cli"?"opencode-go":(m.model||"已接入"));}
    else{c.className="chip warn";c.textContent=m.has_key?"模型 · 已配置但未授权":"模型 · 未接入(离线规则)";}
    document.getElementById("mCompact").textContent=m.has_key?(m.cloud_allowed?"已接入并授权":"已配置 · 未授权"):"离线模式";
    document.getElementById("mSt").textContent=(m.has_key?("已配置 · "+(m.model||"")+" @ "+(m.base_url||"")+(m.cloud_allowed?" · 已授权":" · 云端发送未授权")):"未接入，将使用离线规则模式")+(m.opencode_available?" · opencode可用":"");
    var mb=document.getElementById("mb"),mm=document.getElementById("mm"),mv=document.getElementById("mv");
    document.getElementById("cloudAllowed").checked=!!m.cloud_allowed;
    document.getElementById("webVerify").checked=!!m.web_verify_allowed;
    if(!mb.value)mb.value=m.base_url||""; if(!mm.value)mm.value=m.model||""; if(!mv.value)mv.value=m.vl_model||"";
  });
}
document.getElementById("bImport").onclick=function(){
  var s=document.getElementById("mm").value.trim()||"muse-spark-1.3-contributor";
  API("/api/model/import-opencode",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model:s})})
   .then(function(r){ if(!r.ok){alert(r.error||"导入失败");return;} return refreshModel().then(function(){alert("已接入 opencode-go 远端模型服务："+(r.model||s)+"。尚未授权发送论文内容，请按需单独勾选云端授权。");}); });
};
document.getElementById("bSave").onclick=function(){
  API("/api/model",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
    PAPERBRAIN_API_KEY:document.getElementById("mk").value,
    PAPERBRAIN_BASE_URL:document.getElementById("mb").value,
    PAPERBRAIN_MODEL:document.getElementById("mm").value,
    PAPERBRAIN_VL_MODEL:document.getElementById("mv").value,
    PAPERBRAIN_CLOUD_ALLOWED:document.getElementById("cloudAllowed").checked?"1":"0",
    PAPERBRAIN_WEB_VERIFY:document.getElementById("webVerify").checked?"1":"0"})})
   .then(function(r){document.getElementById("mk").value="";return refreshModel();}).then(function(){alert("已保存");});
};
document.getElementById("bTest").onclick=function(){
  document.getElementById("mSt").textContent="测试中…";
  API("/api/model/test",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"})
   .then(function(r){document.getElementById("mSt").textContent=r.ok?("连通正常，模型回包："+r.reply):("连通失败："+(r.error||""));});
};
document.getElementById("bClear").onclick=function(){
  if(!confirm("清除本机保存的模型配置？"))return;
  fetch("/api/model",{method:"DELETE"}).then(function(){return refreshModel();});
};

/* ---------- 2. 文件 ---------- */
function makePaperId(name){
  var base=(name||"paper").replace(/\.[^.]+$/,"").trim();
  return base.replace(/[^a-zA-Z0-9\u4e00-\u9fff._-]+/g,"_").replace(/^[_\.]+|[_\.]+$/g,"").slice(0,64)||"paper";
}
function suggestPaperId(name){if(!pidTouched)document.getElementById("pid").value=makePaperId(name);}
function setFile(f){
  pickedFile=f||null;
  window.__manyFiles=[];
  var tag=document.getElementById("ftag");
  if(f){tag.style.display="block";tag.textContent="已选择："+f.name+" （"+(f.size/1024/1024).toFixed(2)+" MB）";suggestPaperId(f.name);}
  else{tag.style.display="none";}
}
var drop=document.getElementById("drop"), fileEl=document.getElementById("file");
drop.onclick=function(){fileEl.click();};
drop.ondragover=function(e){e.preventDefault();drop.classList.add("hot");};
drop.ondragleave=function(){drop.classList.remove("hot");};
drop.ondrop=function(e){e.preventDefault();drop.classList.remove("hot");if(e.dataTransfer.files.length>1)setMany(e.dataTransfer.files);else if(e.dataTransfer.files.length)setFile(e.dataTransfer.files[0]);};
fileEl.onchange=function(){if(fileEl.files.length>1)setMany(fileEl.files);else if(fileEl.files.length)setFile(fileEl.files[0]);};
document.getElementById("pid").addEventListener("input",function(){pidTouched=!!this.value.trim();});
document.getElementById("path").addEventListener("change",function(){if(this.value.trim())suggestPaperId(this.value.trim().split("/").pop());});
var manyEl=document.getElementById("files"), dropMany=document.getElementById("dropMany");
dropMany.onclick=function(){manyEl.click();};
dropMany.ondragover=function(e){e.preventDefault();dropMany.classList.add("hot");};
dropMany.ondragleave=function(){dropMany.classList.remove("hot");};
dropMany.ondrop=function(e){e.preventDefault();dropMany.classList.remove("hot");if(e.dataTransfer.files.length)setMany(e.dataTransfer.files);};
manyEl.onchange=function(){if(manyEl.files.length)setMany(manyEl.files);};
function setMany(list){
  pickedFile=null;
  window.__manyFiles=Array.prototype.slice.call(list);
  var msg="已选 "+window.__manyFiles.length+" 篇："+window.__manyFiles.map(function(f){return f.name;}).slice(0,4).join("、")+(window.__manyFiles.length>4?" …":"");
  ["ftag","ftagMany"].forEach(function(id){var t=document.getElementById(id);t.style.display="block";t.textContent=msg;});
}
function readB64(f){return new Promise(function(res,rej){var r=new FileReader();r.onload=function(){res((r.result||"").split(",",2)[1]||"");};r.onerror=rej;r.readAsDataURL(f);});}
function batchPaths(){
  var raw=document.getElementById("bpaths").value||"";
  return raw.split(String.fromCharCode(10)).map(function(s){return s.trim();}).filter(function(s){return s.length>0;});
}
function submitBatch(btn){
  if(!btn)btn=document.getElementById("bBatch");
  var paths=batchPaths();
  var files=window.__manyFiles||[];
  if(!paths.length && !files.length){alert("请选择多个文件，或填写路径/文件夹（每行一个）");return;}
  btn.disabled=true;
  document.getElementById("runSt").innerHTML='<span class="spin"></span> 提交批量任务…';
  var payload={task:selTask(),depth:document.getElementById("depth").value,
    focus:document.getElementById("focus").value.trim(),
    vision:document.getElementById("vis").checked,paths:paths};
  var prep=Promise.resolve(payload);
  if(files.length){
    prep=Promise.all(files.map(function(f){return readB64(f).then(function(b64){return {filename:f.name,data_b64:b64};});}))
      .then(function(arr){payload.files=arr;return payload;});
  }
  prep.then(function(body){return API("/api/batch",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});})
    .then(function(r){ if(!r.ok){throw new Error(r.error||"提交失败");}
      document.getElementById("runSt").textContent="批量任务已提交："+r.total+" 篇";
      return waitJob(r.job); })
    .then(function(res){ lastResult=null; loadMemory(); loadMemoryStats(); loadNotes(""); loadField();
      document.getElementById("sum").innerHTML='<span class="badge neutral">批量完成</span> <span class="mut">成功 '+(res.done||0)+' / 跳过 '+(res.skipped||0)+' / 失败 '+(res.failed||0)+'</span>'; })
    .catch(function(e){ document.getElementById("runSt").textContent=""; alert(e.message); })
    .then(function(){ btn.disabled=false; });
}
document.getElementById("bBatch").onclick=function(){submitBatch(this);};

/* ---------- 3. 类别 ---------- */
function loadTasks(){
  API("/api/tasks").then(function(r){
    TASKS=r.tasks||[];
    var select=document.getElementById("taskSelect");select.innerHTML="";
    TASKS.forEach(function(t){
      var o=document.createElement("option");o.value=t.id;o.textContent=t.label+" · "+t.desc;select.appendChild(o);
    });
  });
}
function selTask(){var e=document.getElementById("taskSelect");return e?e.value:"full_read";}

/* ---------- 4. 运行 ---------- */
function readFileB64(f){return new Promise(function(res,rej){var r=new FileReader();r.onload=function(){res((r.result||"").split(",",2)[1]||"");};r.onerror=rej;r.readAsDataURL(f);});}
function renderBatch(files, prog){
  var box=document.getElementById("bfiles");
  if(!files||!files.length){box.innerHTML="";return;}
  var doneN=(prog&&prog.done)||files.filter(Boolean).length, totalN=(prog&&prog.total)||files.length;
  var head='<div class="hint" style="grid-column:1/-1">批量进度 '+doneN+'/'+totalN+'（串行）</div>';
  box.innerHTML=head+files.map(function(f){
    if(!f||!f.paper_id)return "";
    var cls=(f.verify==="CLEAN")?"clean":(f.verify==="ERROR"?"dirty":"review");
    return '<div class="mem"><div class="h"><div><div class="ttl">'+esc(f.paper_id)+'</div>'+
      '<div class="meta">'+esc(f.file||"")+(f.notes?(' · 笔记 '+f.notes):'')+(f.elapsed?(' · '+f.elapsed+'s'):'')+'</div></div>'+
      '<span class="badge '+cls+'">'+esc(f.verify||"…")+'</span></div>'+
      (f.error?('<div class="hint" style="color:var(--bad)">'+esc(f.error)+'</div>'):'')+'</div>';
  }).join("");
}
function waitJob(job){
  document.getElementById("prog").style.display="block";
  return new Promise(function(resolve,reject){
    (function loop(){
      API("/api/job?id="+job).then(function(j){
        var pct=j.done?100:Math.min(92,8+(j.elapsed||0)*2);
        document.getElementById("pbar").style.width=pct+"%";
        document.getElementById("ptxt").textContent="阶段："+(j.stage||"…")+" · "+(j.elapsed||0)+"s";
        if(j.files)renderBatch(j.files, j.progress);
        if(j.done){document.getElementById("prog").style.display="none";if(!j.ok){reject(new Error(j.error||"任务失败"));return;}resolve(j.result);return;}
        setTimeout(loop,900);
      }).catch(reject);
    })();
  });
}
document.getElementById("bRun").onclick=function(){
  var path=document.getElementById("path").value.trim();
  var many=(window.__manyFiles||[]).length, bpaths=batchPaths().length;
  // 选了多个文件 / 填了多行路径 -> 自动走批量 (修复: 以前只认单文件, 误弹"请先选择")
  if(!pickedFile && !path && (many||bpaths)){ submitBatch(this); return; }
  if(!pickedFile && !path){alert("请先选择文件：单篇用上方文件框，多篇用「批量」多选框/拖拽，或在路径框/路径列表填写本机路径");return;}
  var btn=document.getElementById("bRun");btn.disabled=true;
  document.getElementById("runSt").innerHTML='<span class="spin"></span> 提交任务…';
  var payload={task:selTask(),paper_id:document.getElementById("pid").value.trim()||"paper",
    vision:document.getElementById("vis").checked,
    depth:document.getElementById("depth").value,
    focus:document.getElementById("focus").value.trim()};
  var prep=path?Promise.resolve(Object.assign(payload,{path:path})):readFileB64(pickedFile).then(function(b64){return Object.assign(payload,{filename:pickedFile.name,data_b64:b64});});
  var needsConfirm=(payload.task==="full_read"&&payload.depth==="deep");
  prep.then(function(body){return API(needsConfirm?"/api/outline":"/api/task",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});})
   .then(function(r){
     if(!r.ok){throw new Error(r.error||"提交失败");}
     lastPid=r.paper_id;
     if(needsConfirm){
       pendingFocus=payload.focus||"";
       document.getElementById("outlineEdit").value=JSON.stringify(r.outline,null,2);
       document.getElementById("outlineConfirm").style.display="block";
       lastResult={paper_id:r.paper_id,verify:"AWAITING_CONFIRMATION",is_clean:false,
         markdown:"大纲已生成。请检查并编辑左侧 JSON，明确确认后才会生成深度草稿。"};
       renderResult(lastResult);showFile("outline_v1.json");return null;
     }
     document.getElementById("outlineConfirm").style.display="none";
     return waitJob(r.job);
   })
   .then(function(res){ if(!res)return; lastResult=res; renderResult(res); loadMemory(); loadMemoryStats(); loadNotes(""); loadField(); })
   .catch(function(e){ document.getElementById("sum").innerHTML='<div class="empty">'+e.message+'</div>'; })
   .then(function(){ btn.disabled=false; document.getElementById("runSt").textContent=""; });
};
document.getElementById("bConfirmOutline").onclick=function(){
  var btn=this, outline;
  try{outline=JSON.parse(document.getElementById("outlineEdit").value);}catch(e){alert("大纲 JSON 无效："+e.message);return;}
  btn.disabled=true;document.getElementById("runSt").innerHTML='<span class="spin"></span> 已确认，开始深读与生成…';
  API("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
    paper_id:lastPid,outline:outline,deepread:true,focus:pendingFocus})})
   .then(function(r){if(!r.ok)throw new Error(r.error||"生成提交失败");return waitJob(r.job);})
   .then(function(res){lastResult=res;renderResult(res);document.getElementById("outlineConfirm").style.display="none";loadMemory();loadMemoryStats();loadNotes("");loadField();})
   .catch(function(e){document.getElementById("sum").innerHTML='<div class="empty">'+esc(e.message)+'</div>';})
   .then(function(){btn.disabled=false;document.getElementById("runSt").textContent="";});
};

function esc(s){return (s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function mdToHtml(md){
  if(!md)return '<span class="mut">无内容</span>';
  var out=esc(md);
  out=out.replace(/^### (.*)$/gm,"<h3>$1</h3>").replace(/^## (.*)$/gm,"<h2>$1</h2>").replace(/^# (.*)$/gm,"<h1>$1</h1>");
  out=out.replace(/\\*\\*(.+?)\\*\\*/g,"<b>$1</b>");
  out=out.replace(/^\\- (.*)$/gm,"<li>$1</li>");
  out=out.replace(/(<li>[\\s\\S]*?<\\/li>)/g,function(x){return "<ul>"+x+"</ul>";});
  out=out.replace(/([^>])\\n/g,"$1<br/>");
  return out;
}
function renderResult(res){
  var md=res.markdown||"";
  var cls="neutral",txt=res.verify||res.task||"完成";
  if(res.verify==="CLEAN")cls="clean"; else if(res.verify==="DIRTY")cls="dirty"; else if(res.verify==="NEEDS_REVIEW")cls="review";
  document.getElementById("sum").innerHTML='<span class="badge '+cls+'">'+txt+'</span> <span class="mut">'+(res.paper_id||lastPid)+' · 模型:'+(res.model||"opencode-go")+'</span>'+(res.memory_saved?' <span class="mut">· 已写入记忆</span>':'');
  var st=document.getElementById("stats"); st.style.display="grid";
  var led=res.ledger||{};
  document.getElementById("sTok").textContent=(led.total!=null?led.total:"–");
  document.getElementById("sBud").textContent=(res.budget_ok===undefined?"–":(res.budget_ok?"通过":"超限"));
  document.getElementById("sBud").style.color=res.budget_ok===false?"var(--bad)":(res.budget_ok?"var(--ok)":"");
  document.getElementById("sCit").textContent="-";
  document.getElementById("sEnt").textContent="-";
  if(lastPid){
    API("/api/out?paper_id="+encodeURIComponent(lastPid)+"&file=verify.json").then(function(v){
      try{ document.getElementById("sCit").textContent=JSON.parse(v.content).report.length; }catch(e){}
    }).catch(function(){});
    API("/api/out?paper_id="+encodeURIComponent(lastPid)+"&file=memory.json").then(function(m){
      try{ document.getElementById("sEnt").textContent=JSON.parse(m.content).entities.length; }catch(e){}
    }).catch(function(){});
  }
  showFile("result");
}
function showFile(f){
  document.querySelectorAll(".tab").forEach(function(b){b.classList.toggle("on",b.dataset.f===f);});
  var view=document.getElementById("view");
  if(f==="result"){ view.innerHTML=mdToHtml(lastResult&&lastResult.markdown); return; }
  if(!lastPid){view.innerHTML='<span class="mut">请先运行</span>';return;}
  API("/api/out?paper_id="+encodeURIComponent(lastPid)+"&file="+f).then(function(q){
    if(f==="verify.json"){ try{var v=JSON.parse(q.content);view.innerHTML=mdToHtml("**状态** "+v.status+"  **可放行** "+v.is_clean+"\\n\\n"+(v.report||[]).map(function(x){return "- "+x.citation+" → "+x.status+(x.score!=null?(" ("+x.score+")"):"")+" "+(x.reason||"");}).join("\\n"));return;}catch(e){} }
    if(f.endsWith(".json")){view.textContent=q.content||"";return;}
    view.innerHTML=mdToHtml(q.content||"");
  });
}
document.querySelectorAll(".tab").forEach(function(b){b.onclick=function(){showFile(b.dataset.f);};});
document.getElementById("artifactSelect").onchange=function(){if(this.value){showFile(this.value);this.value="";}};

/* ---------- 4. 记忆 ---------- */
function loadMemoryStats(){
  API("/api/memory/stats").then(function(s){
    var review=(s.candidate_notes||0)+(s.contested_notes||0);
    document.getElementById("mActive").textContent=s.active_notes||0;
    document.getElementById("mReview").textContent=review;
    document.getElementById("mPapers").textContent=s.papers||0;
    document.getElementById("cMem").textContent="可信记忆 · "+(s.active_notes||0);
    API("/api/memory/review").then(function(r){
      document.getElementById("autoTrust").textContent=r.running?"正在回查原文并复核记忆":
        (r.reason||r.error||("本轮复核 "+r.processed+" 条 · 吸收 "+r.accepted+" · 移除 "+r.rejected));
    });
  });
}
function loadMemory(q){
  API("/api/memory"+(q?("?q="+encodeURIComponent(q)):"")).then(function(r){
    var items=r.items||[];
    var box=document.getElementById("mems");
    if(!items.length){box.innerHTML='<div class="empty" style="grid-column:1/-1">暂无记忆，跑一次解读即自动写入</div>';return;}
    box.innerHTML=items.map(function(m){
      var ents=(m.entities||[]).map(function(e){return (e.name||e);}).slice(0,8).join("、");
      return '<div class="mem"><div class="h"><div><div class="ttl">'+esc(m.title||m.paper_id)+'</div><div class="meta">'+esc(m.paper_id)+' · '+esc(m.task)+'</div></div><span class="x" onclick="delMem('+m.id+')">删除</span></div>'+
             '<div class="bd">'+esc((m.summary||"").slice(0,300))+(ents?('<br/><span class="mut">实体：'+esc(ents)+'</span>'):'')+'</div></div>';
    }).join("");
  });
}
function delMem(id){ if(!confirm("删除该条记忆？"))return; fetch("/api/memory?id="+id,{method:"DELETE"}).then(function(){loadMemory(document.getElementById("mq").value.trim());}); }
document.getElementById("bSearch").onclick=function(){loadMemory(document.getElementById("mq").value.trim());};
document.getElementById("mq").addEventListener("keydown",function(e){if(e.key==="Enter")loadMemory(this.value.trim());});

/* 知识网络: 原子笔记 + 链接 */
function noteCard(n,tag){
  var isEnt=(n.kind==="entity");
  var kws=isEnt?[]:(n.keywords||[]).filter(function(k){return k.length>1;}).slice(0,6).join("、");
  var lnk=(n.links||[]).length;
  return '<div class="mem"><div class="h"><div><div class="ttl">'+esc(tag||n.kind||"note")+'</div><div class="meta">'+
    esc(n.paper_id)+(lnk?(' · 链'+lnk):'')+'</div></div></div><div class="bd">'+esc((n.content||"").slice(0,280))+
    (kws?('<br/><span class="mut">关键词：'+esc(kws)+'</span>'):'')+'</div></div>';
}
function loadNotes(q){
  API("/api/notes"+(q?("?q="+encodeURIComponent(q)):"")).then(function(r){
    var all=(r.seeds||[]), nbrs=(r.neighbors||[]);
    var seeds=all.filter(function(n){return n.kind!=="entity";});   // 实体笔记另见「图谱」, 不铺满网络
    var hidden=all.length-seeds.length;
    var box=document.getElementById("notes");
    if(!seeds.length&&!nbrs.length){box.innerHTML='<div class="empty" style="grid-column:1/-1">暂无笔记，跑一次“全文精读”即自动生成</div>';return;}
    var html=seeds.map(function(n){return noteCard(n,n.kind);}).join("");
    if(nbrs.length){ html+='<div class="empty" style="grid-column:1/-1;padding:10px">沿链接扩展：</div>'+nbrs.map(function(n){return noteCard(n,"关联 · "+(n.kind||""));}).join(""); }
    if(hidden>0){ html+='<div class="hint" style="grid-column:1/-1">已折叠 '+hidden+' 条实体笔记（详见「图谱」页签）</div>'; }
    box.innerHTML=html;
  });
}
document.getElementById("bNotes").onclick=function(){loadNotes(document.getElementById("nq").value.trim());};
document.getElementById("nq").addEventListener("keydown",function(e){if(e.key==="Enter")loadNotes(this.value.trim());});
loadNotes("");

/* 领域地图: 跨论文共享概念 */
function loadField(){
  API("/api/field").then(function(f){
    var box=document.getElementById("field");
    var sc=(f&&f.shared_concepts)||[], ed=(f&&f.edges)||[], cl=(f&&f.clusters)||[];
    if(!sc.length&&!ed.length){box.innerHTML='<div class="empty">暂无跨论文关联，多跑几篇同方向论文即自动成网</div>';return;}
    var h='<div class="mems">';
    sc.slice(0,12).forEach(function(c){
      h+='<div class="mem"><div class="h"><div><div class="ttl">'+esc(c.term)+'</div><div class="meta">'+c.papers.length+' 篇共享</div></div></div>'+
         '<div class="bd">'+esc((c.papers||[]).join("、"))+'</div></div>';
    });
    h+='</div>';
    if(cl.length){h+='<div class="hint">连通簇：'+cl.map(function(c){return '['+esc(c.join(" · "))+']';}).join(" ")+'</div>';}
    box.innerHTML=h;
  });
}
loadField();

/* 问记忆: 检索 + 合成回答 */
function askMem(){
  var q=document.getElementById("aq").value.trim();
  if(!q){alert("请输入问题");return;}
  var box=document.getElementById("ans");
  box.innerHTML='<span class="spin"></span> 检索并合成…';
  API("/api/ask?q="+encodeURIComponent(q)).then(function(r){
    var html='<div>'+mdToHtml(r.answer||"")+'</div>';
    if(r.used&&r.used.length){html+='<div class="hint">依据记忆编号：'+r.used.slice(0,8).join(", ")+'</div>';}
    box.innerHTML=html;
  }).catch(function(){box.innerHTML='<span class="mut">检索失败</span>';});
}
document.getElementById("bAsk").onclick=askMem;
document.getElementById("aq").addEventListener("keydown",function(e){if(e.key==="Enter")askMem();});
async function doReflect(scope,btn){
  btn.disabled=true;
  var st=document.getElementById("runSt");
  st.innerHTML='<span class="spin"></span> '+((scope==="field")?"跨论文综合领域洞见":"综合高层洞见")+'中（需模型，约30-60秒）…';
  try{
    var payload = scope==="field" ? {scope:"field"} : {paper_id:lastPid||""};
    var r=await API("/api/reflect",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(payload)});
    st.textContent = r.ok ? ("已生成 "+r.insights+" 条"+(scope==="field"?"领域":"")+"反思（已验证 "
      +(r.verified||0)+" 条），可在“知识网络”查看") : ("未生成："+(r.reason||""));
    if(r.ok){loadNotes("");}
  }catch(e){st.textContent="反思失败";}
  finally{btn.disabled=false;}
}
document.getElementById("bReflect").onclick=function(){doReflect("paper",this);};
document.getElementById("bField").onclick=function(){doReflect("field",this);};

/* 待裁决 + 偏好 */
function loadDecisions(){
  API("/api/decisions?kind=candidate").then(function(a){
    API("/api/decisions?kind=contested").then(function(b){
      var items=(a.items||[]).concat(b.items||[]);
      var box=document.getElementById("decisions");
      if(!items.length){box.innerHTML='<div class="empty" style="grid-column:1/-1">暂无待裁决项</div>';return;}
      box.innerHTML=items.map(function(n){
        return '<div class="mem"><div class="h"><div><div class="ttl">'+esc(n.status)+' · '+esc(n.kind)+'</div>'+
          '<div class="meta">'+esc(n.paper_id)+'</div></div></div>'+
          '<div class="bd">'+esc((n.content||"").slice(0,200))+'</div>'+
          '<div class="hint"><a href="#" onclick="decide('+n.id+',true);return false;">采纳</a> · '+
          '<a href="#" onclick="decide('+n.id+',false);return false;">驳回</a></div></div>';
      }).join("");
    });
  });
}
function decide(id,accept){ API("/api/decisions",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:id,accept:accept})}).then(function(){loadDecisions();loadMemoryStats();}); }
function loadPrefs(){
  API("/api/preferences").then(function(r){
    var items=r.items||[];
    var el=document.getElementById("prefs");
    if(!items.length){el.textContent="偏好：暂无（提问命中已知概念后自动积累）";return;}
    el.innerHTML="偏好（点 × 删除）："+items.slice(0,12).map(function(p){
      return '<span class="chip">'+esc(p.term)+' '+p.weight+' <a href="#" onclick="delPref(\\''+esc(p.term)+'\\');return false;">×</a></span>';
    }).join(" ");
  });
}
function delPref(term){ fetch("/api/preferences?term="+encodeURIComponent(term),{method:"DELETE"}).then(loadPrefs); }
loadDecisions(); loadPrefs();

/* init */
fetch("/api/health").then(function(r){return r.json();}).then(function(h){
  var c=document.getElementById("cTools");
  c.textContent="工具 · PDF"+(h.pdf?"✓":"✗")+" OCR"+(h.ocr?"✓":"✗")+" 图"+(h.cv2?"✓":"✗");
});
refreshModel(); loadTasks(); loadMemory(""); loadMemoryStats();
</script>
</body></html>

"""


class Server(ThreadingHTTPServer):
    request_queue_size = 32

    def process_request(self, request, client_address):
        request.settimeout(180)  # 单连接读写上限, 防半开连接卡死服务
        super().process_request(request, client_address)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            b = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if u.path == "/api/demo":
            p = ROOT / "demo" / "sample_paper.txt"
            self._json({"text": p.read_text(encoding="utf-8") if p.exists() else ""})
            return
        if u.path == "/api/health":
            import shutil
            has = lambda c: shutil.which(c) is not None
            try:
                import fitz  # noqa
                pdf = True
            except Exception:
                pdf = False
            try:
                import cv2  # noqa
                cv = True
            except Exception:
                cv = False
            try:
                from .llm import has_key
                llm = has_key()
            except Exception:
                llm = False
            try:
                from .ocr import has_ocr
                ocr = has_ocr()
            except Exception:
                ocr = False
            try:
                from .formulas import checker_health
                formula_checkers = checker_health()
            except Exception:
                formula_checkers = {"katex": False, "sympy": False,
                                    "any": False, "dual": False}
            try:
                from .llm_ops import enabled as _am
                all_model = _am()
            except Exception:
                all_model = False
            try:
                from .vector_store import health as _vss_health
                vector_index = _vss_health()
            except Exception as exc:
                vector_index = {"ok": False, "backend": "duckdb-vss",
                                "reason": str(exc)[:120]}
            self._json({"llm": llm, "pdf": pdf, "ocr": ocr, "cv2": cv,
                        "all_model": all_model, "formula_checkers": formula_checkers,
                        "vector_index": vector_index,
                        "node": has("node"), "tesseract": has("tesseract")})
            return
        if u.path == "/api/model":
            from .modelconf import status
            self._json(status())
            return
        if u.path == "/api/job":
            q = parse_qs(u.query)
            jid = (q.get("id", [""])[0] or "").strip()
            with JOBS_LOCK:
                j = dict(JOBS.get(jid, {}))
            if not j:
                self._json({"error": "未知任务"}, 404)
                return
            j["elapsed"] = round(time.time() - j.get("t0", time.time()), 1)
            self._json(j)
            return
        if u.path == "/api/tasks":
            from .tasks import task_labels
            self._json({"tasks": task_labels()})
            return
        if u.path in ("/api/outline/history", "/api/outline/diff"):
            q = parse_qs(u.query)
            pid = _safe_pid(q.get("paper_id", [""])[0])
            out_dir = OUT_ROOT / pid
            from .outline import list_outline_versions, diff_outline_versions
            try:
                if u.path.endswith("/history"):
                    self._json({"paper_id": pid,
                                "versions": list_outline_versions(str(out_dir))})
                else:
                    old = str(q.get("from", [""])[0])
                    new = str(q.get("to", [""])[0])
                    self._json({"paper_id": pid, "from": old, "to": new,
                                "diff": diff_outline_versions(str(out_dir), old, new)})
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
            return
        if u.path == "/api/memory":
            from .memory_store import search
            q = parse_qs(u.query)
            kw = (q.get("q", [""])[0] or "").strip()
            self._json({"items": search(kw)})
            return
        if u.path == "/api/notes":
            from .memory_store import search_notes, recall
            q = parse_qs(u.query)
            kw = (q.get("q", [""])[0] or "").strip()
            if kw:
                self._json(recall(kw, k=6))
            else:
                # 默认只看最近一篇的笔记, 避免把历次所有笔记铺满 (防"满屏卡片")
                alln = search_notes("", limit=500)
                pid = str(q.get("paper_id", [""])[0] or "").strip()
                if not pid and alln:
                    pid = alln[0]["paper_id"]
                seeds = [n for n in alln if (not pid or n["paper_id"] == pid)]
                self._json({"seeds": seeds, "neighbors": [], "paper": pid})
            return
        if u.path == "/api/field":
            from .memory_store import field_map
            self._json(field_map())
            return
        if u.path == "/api/memory/stats":
            from .memory_store import memory_stats
            self._json(memory_stats())
            return
        if u.path == "/api/memory/review":
            from .adjudication import status
            self._json(status())
            return
        if u.path == "/api/ask":
            q = parse_qs(u.query)
            kw = (q.get("q", [""])[0] or "").strip()
            if not kw:
                self._json({"error": "缺少问题 q"}, 400)
                return
            compose = (q.get("compose", ["1"])[0] or "1") != "0"
            from .memory_store import ask_memory
            self._json(ask_memory(kw, compose=compose))
            return
        if u.path == "/api/aliases":
            from .memory_store import list_aliases
            self._json({"items": list_aliases()})
            return
        if u.path == "/api/embed/health":
            from .embeddings import health
            self._json(health())
            return
        if u.path == "/api/preferences":
            from .memory_store import list_preferences
            self._json({"items": list_preferences()})
            return
        if u.path == "/api/decisions":
            q = parse_qs(u.query)
            kind = (q.get("kind", ["candidate"])[0] or "candidate").strip()
            from .memory_store import pending_decisions
            self._json({"kind": kind, "items": pending_decisions(kind)})
            return
        if u.path == "/api/out":
            q = parse_qs(u.query)
            pid = _safe_pid((q.get("paper_id", [""])[0] or "").strip())
            fn = (q.get("file", [""])[0] or "").strip().replace("/", "")
            if not fn or fn in (".", "..") or not re.match(r"^[\w\-.]+$", fn):
                self._json({"error": "bad params"}, 400)
                return
            jid_param = (q.get("job") or q.get("job_id") or [""])[0].strip()
            fp = None
            if jid_param:
                cand = OUT_ROOT / f"{pid}_{jid_param}" / fn
                if cand.is_file():
                    fp = cand
            if fp is None:
                cand = OUT_ROOT / pid / fn
                if cand.is_file():
                    fp = cand
            if fp is None:
                # 检查最新运行的隔离目录
                matches = sorted(OUT_ROOT.glob(f"{pid}_*"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                for m in matches:
                    if (m / fn).is_file():
                        fp = m / fn
                        break
            if fp is None or not _under(fp, OUT_ROOT) or not fp.is_file():
                self._json({"error": f"不存在: {fn}"}, 404)
                return
            if q.get("raw", ["0"])[0] == "1":
                b = fp.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            self._json({"content": fp.read_text(encoding="utf-8", errors="replace")[:20000]})
            return
        self._json({"error": "not found"}, 404)

    def do_DELETE(self):
        u = urlparse(self.path)
        if u.path == "/api/memory":
            q = parse_qs(u.query)
            from .memory_store import delete, clear
            if q.get("id", [""])[0].strip().isdigit():
                self._json(delete(int(q["id"][0])))
            else:
                self._json(clear())
            return
        if u.path == "/api/notes":
            q = parse_qs(u.query)
            from .memory_store import delete_note, clear_notes
            if q.get("id", [""])[0].strip().isdigit():
                self._json(delete_note(int(q["id"][0])))
            else:
                self._json(clear_notes())
            return
        if u.path == "/api/field":
            from .memory_store import clear_concepts
            self._json(clear_concepts())
            return
        if u.path == "/api/preferences":
            q = parse_qs(u.query)
            from .memory_store import delete_preference
            self._json(delete_preference((q.get("term", [""])[0] or "").strip()))
            return
        if u.path != "/api/model":
            self._json({"error": "not found"}, 404)
            return
        from .modelconf import clear
        self._json({"ok": True, **clear()})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path not in ("/api/run", "/api/upload", "/api/outline", "/api/generate",
                           "/api/outline/rollback",
                           "/api/task", "/api/batch", "/api/model", "/api/model/test",
                           "/api/model/import-opencode", "/api/embed/backfill",
                           "/api/preferences", "/api/decisions", "/api/reflect",
                           "/api/invalidate"):
            self._json({"error": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 30 * 1024 * 1024:
                self._json({"ok": False, "error": "请求体超过 30MB 上限"})
                return
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            if u.path == "/api/model":
                from .modelconf import save
                fields = {k: body.get(k, "") for k in
                          ("PAPERBRAIN_API_KEY", "PAPERBRAIN_BASE_URL",
                           "PAPERBRAIN_MODEL", "PAPERBRAIN_VL_MODEL",
                           "PAPERBRAIN_PROVIDER", "PAPERBRAIN_CLOUD_ALLOWED", "PAPERBRAIN_WEB_VERIFY")}
                self._json({"ok": True, **save(fields)})
                from .adjudication import start
                start()
                return
            if u.path == "/api/model/test":
                from .modelconf import test_connection
                self._json(test_connection())
                return
            if u.path == "/api/model/import-opencode":
                from .modelconf import import_opencode
                self._json(import_opencode(str(body.get("model", ""))))
                return
            if u.path == "/api/embed/backfill":
                from .memory_store import embed_notes
                self._json(embed_notes(body.get("paper_id") or None))
                return
            if u.path == "/api/preferences":
                from .memory_store import bump_preference, decay_preferences
                if body.get("decay"):
                    self._json(decay_preferences())
                    return
                self._json(bump_preference(str(body.get("term", "")),
                                           float(body.get("weight", 0.3))))
                return
            if u.path == "/api/decisions":
                from .memory_store import decide_note
                self._json(decide_note(int(body.get("id", 0)), bool(body.get("accept", True))))
                return
            if u.path == "/api/reflect":
                from .memory_store import reflect, reflect_global
                if str(body.get("scope") or "") == "field":
                    self._json(reflect_global())
                else:
                    pid = _safe_pid(body.get("paper_id") or "") if body.get("paper_id") else None
                    self._json(reflect(pid))
                return
            if u.path == "/api/invalidate":
                from .memory_store import invalidate_note
                self._json(invalidate_note(int(body.get("id", 0))))
                return
            pid = str(body.get("paper_id", "")).strip() or "demo_paper"
            pid = _safe_pid(pid)
            th = float(body.get("threshold", 0.82))
            d = OUT_ROOT / pid
            d.mkdir(parents=True, exist_ok=True)
            from .pipeline import run_paper, run_outline, generate_from_outline
            if u.path == "/api/outline/rollback":
                from .outline import rollback_outline
                try:
                    rolled = rollback_outline(str(d), str(body.get("version", "")))
                    self._json({"ok": True, "paper_id": pid, "outline": rolled})
                except ValueError as exc:
                    self._json({"ok": False, "error": str(exc)}, 400)
                return
            if u.path == "/api/task":
                task = str(body.get("task", "full_read")).strip() or "full_read"
                jid = _job_new(pid)
                job_dir = Path(_job_dir(pid, jid))
                src = self._save_input(body, job_dir, upload=True)
                if isinstance(src, dict):
                    self._json(src)
                    return
                if src is None:
                    self._json({"ok": False, "error": "请选择本地文件或输入文件路径"})
                    return
                if not _submit_job(jid, _run_task_job, jid, task, str(src), pid, th,
                                   bool(body.get("vision")), str(body.get("focus", "")),
                                   str(body.get("depth", "standard"))):
                    self._json({"ok": False, "error": "任务队列已满"}, 429)
                    return
                self._json({"ok": True, "job": jid, "paper_id": pid, "task": task})
                return
            if u.path == "/api/batch":
                import base64
                task = str(body.get("task", "full_read")).strip() or "full_read"
                depth = str(body.get("depth", "standard")).strip() or "standard"
                focus = str(body.get("focus", ""))
                resume = body.get("resume", True) is not False
                items: List[dict] = []
                pat = str(body.get("pattern", "*.pdf"))
                for p in (body.get("paths") or []):
                    sp = Path(str(p).strip()).expanduser()
                    if not str(p).strip():
                        continue
                    if not _path_allowed(sp):
                        self._json({"ok": False, "error": f"路径不在允许范围: {p}"})
                        return
                    if sp.is_file():
                        items.append({"path": str(sp)})
                    elif sp.is_dir():
                        for f in sorted(sp.glob(pat)):
                            if f.is_file():
                                items.append({"path": str(f)})
                for dd in (body.get("dirs") or []):
                    sd = Path(str(dd).strip()).expanduser()
                    if not sd.is_dir():
                        self._json({"ok": False, "error": f"目录不存在: {dd}"})
                        return
                    if not _path_allowed(sd):
                        self._json({"ok": False, "error": f"目录不在允许范围: {dd}"})
                        return
                    for f in sorted(sd.glob(pat)):
                        if f.is_file():
                            items.append({"path": str(f)})
                updir = OUT_ROOT / "__uploads__"
                for fobj in (body.get("files") or []):
                    fn = str(fobj.get("filename", "")).strip().lower()
                    if fn and not fn.endswith((".pdf", ".txt", ".md")):
                        continue
                    sfx = ".pdf" if fn.endswith(".pdf") else (".md" if fn.endswith(".md") else ".txt")
                    try:
                        raw = base64.b64decode(fobj.get("data_b64", ""), validate=True)
                    except Exception:
                        continue
                    if not (10 <= len(raw) <= 20 * 1024 * 1024):
                        continue
                    updir.mkdir(parents=True, exist_ok=True)
                    stem = re.sub(r"[^\w\-.]+", "_", Path(fn or "paper").stem)[:48] or "paper"
                    digest = hashlib.sha256(raw).hexdigest()[:10]
                    dst = updir / f"{stem}_{digest}{sfx}"
                    dst.write_bytes(raw)
                    items.append({"path": str(dst)})
                # 去重保序
                seen, uniq = set(), []
                for it in items:
                    if it["path"] not in seen:
                        seen.add(it["path"])
                        uniq.append(it)
                if not uniq:
                    self._json({"ok": False, "error": "没有可处理的文件（提供 paths/dirs/files）"})
                    return
                jid = _job_new("batch")
                if not _submit_job(jid, _run_batch_job, jid, uniq, task, depth, focus,
                                   bool(body.get("vision")), th, resume):
                    self._json({"ok": False, "error": "任务队列已满"}, 429)
                    return
                self._json({"ok": True, "job": jid, "total": len(uniq)})
                return
            if u.path == "/api/generate":
                try:
                    outline = body.get("outline") or None
                    if isinstance(outline, str):
                        outline = json.loads(outline)
                except Exception:
                    self._json({"ok": False, "error": "outline JSON 解析失败"})
                    return
                if not (d / "state.json").exists():
                    self._json({"ok": False, "error": "先跑步骤1出大纲"})
                    return
                draft = body.get("draft") or None
                if body.get("use_session_draft") and not draft:
                    fp = d / "draft_session.md"
                    if not fp.exists():
                        self._json({"ok": False, "error": "会话草稿不存在：请先在对话中让我写好草稿"})
                        return
                    draft = fp.read_text(encoding="utf-8")
                jid = _job_new(pid)
                if not _submit_job(jid, _run_generate_job, jid, pid, th, outline, draft, False,
                                   bool(body.get("deepread")), str(body.get("focus", ""))):
                    self._json({"ok": False, "error": "任务队列已满"}, 429)
                    return
                self._json({"ok": True, "job": jid, "paper_id": pid})
                return
            if u.path == "/api/outline":
                jid = _job_new(pid)
                job_dir = Path(_job_dir(pid, jid))
                src = self._save_input(body, job_dir)
                if isinstance(src, dict):
                    self._json(src)
                    return
                if src is None:
                    text = str(body.get("text", ""))
                    if len(text.strip()) < 50:
                        self._json({"ok": False, "error": f"正文太短 (实收{len(text.strip())}字<50字). 请确认粘到左侧“论文正文”框 (不是大纲框), 或选文件/点填入demo"})
                        return
                    src = job_dir / "input.txt"
                    src.write_text(text, encoding="utf-8")
                o = run_outline(src, pid, str(job_dir), vision=bool(body.get("vision")))
                if pid != "demo_paper":
                    try:
                        for f in job_dir.iterdir():
                            if f.is_file():
                                shutil.copy2(f, d / f.name)
                    except Exception:
                        pass
                result = {"paper_id": pid, "outline": o["outline"],
                          "route": o["route"], "warnings": o["warnings"]}
                _job_set(jid, stage="完成·等待确认大纲", done=True, ok=True,
                         quality_ok=None, result=result)
                self._json({"ok": True, "job": jid, **result})
                return
            jid = _job_new(pid)
            job_dir = Path(_job_dir(pid, jid))
            if u.path == "/api/upload":
                src = self._save_input(body, job_dir, upload=True)
                if isinstance(src, dict):
                    self._json(src)
                    return
            else:
                text = str(body.get("text", ""))
                if len(text.strip()) < 50:
                    self._json({"ok": False, "error": f"正文太短 (实收{len(text.strip())}字<50字). 请确认粘到左侧“论文正文”框 (不是大纲框), 或选文件/点填入demo"})
                    return
                src = job_dir / "input.txt"
                src.write_text(text, encoding="utf-8")
            if not _submit_job(jid, _run_full_job, jid, str(src), pid, th,
                               bool(body.get("vision"))):
                self._json({"ok": False, "error": "任务队列已满"}, 429)
                return
            self._json({"ok": True, "job": jid, "paper_id": pid})
        except Exception as e:
            traceback.print_exc()
            self._json({"ok": False, "error": f"{e}"})

    @staticmethod
    def _save_input(body: dict, d, upload: bool = False):
        import base64
        # 直接读取本地路径 (本机服务, 无需上传, 最省空间)
        pth = str(body.get("path", "")).strip()
        if pth:
            import shutil as _sh
            sp = Path(pth).expanduser()
            if not sp.exists():
                return {"ok": False, "error": f"路径不存在: {pth}"}
            if not _path_allowed(sp):
                return {"ok": False, "error": "路径不在允许范围 (仅限 用户目录/tmp/Volumes)"}
            sfx = sp.suffix.lower()
            if sfx not in (".pdf", ".txt", ".md"):
                return {"ok": False, "error": "仅支持 .pdf / .txt / .md"}
            dst = d / f"source{sfx}"
            try:
                _sh.copy2(str(sp), str(dst))
            except Exception:
                return {"ok": False, "error": "复制失败(检查权限)"}
            return str(dst)
        if not upload and "data_b64" not in body:
            return None
        if upload or "data_b64" in body:
            fn = str(body.get("filename", "")).strip().lower()
            sfx = ".pdf" if fn.endswith(".pdf") else (".md" if fn.endswith(".md") else ".txt")
            if fn and not fn.endswith((".txt", ".md", ".pdf")):
                return {"ok": False, "error": "仅支持 .txt / .md / .pdf"}
            try:
                raw = base64.b64decode(body.get("data_b64", ""), validate=True)
            except Exception:
                return {"ok": False, "error": "文件 base64 解码失败"}
            if len(raw) > 20 * 1024 * 1024 or len(raw) < 10:
                return {"ok": False, "error": "文件大小非法 (需 10B~20MB)"}
            dst = d / f"upload{sfx}"
            dst.write_bytes(raw)
            return str(dst)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    try:
        from .modelconf import load as _load_conf
        _load_conf()
    except Exception:
        pass
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    from .adjudication import start
    start()
    print(f"PaperBrain 控制台: http://{a.host}:{a.port}  (Ctrl+C 退出)")
    Server((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
