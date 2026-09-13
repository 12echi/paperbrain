"""可复现性能基准 (offline, 不调模型/不联网). 用法: python3 tools/bench.py [rounds]

覆盖: 记忆写入/检索/召回/问记忆/统计/领域图, 验真器, 上下文构建, 命令行端到端。
输出各操作中位数毫秒, 便于改动前后对比。
"""
import json
import atexit
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 隔离 DB + 关闭网络依赖 (注意: 必须显式置空, 否则 config 会退回读 env.json)
TMP = tempfile.mkdtemp(prefix="pb_bench_")
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
os.environ["PAPERBRAIN_MEMORY_DB"] = os.path.join(TMP, "mem.sqlite")
os.environ["PAPERBRAIN_EMBED_BASE_URL"] = ""
os.environ["PAPERBRAIN_EMBED_API_KEY"] = ""
os.environ["PAPERBRAIN_PROVIDER"] = "https"
os.environ["PAPERBRAIN_API_KEY"] = ""


def timed(fn, rounds: int):
    xs = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(xs), min(xs)


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    from paperbrain import memory_store as ms
    from paperbrain.retrieval import make_scorer
    from paperbrain.verifier import CitationVerifierV5

    out = {}
    # ---- 记忆写入: 500 notes ----
    t0 = time.perf_counter()
    for i in range(500):
        ms.add_notes(f"P{i % 10}", "bench", [
            {"kind": "claim", "content": f"笔记 {i}: DBSCAN 与 DNA 损伤的束流品质评估 QoB LET 实验证据 {i}"}])
    out["ingest_500_notes"] = (time.perf_counter() - t0) * 1000.0

    q = "DBSCAN 束流品质 LET 损伤"
    out["search_notes"] = timed(lambda: ms.search_notes(q, limit=10), rounds)
    out["recall"] = timed(lambda: ms.recall(q, k=5), rounds)
    out["ask_offline"] = timed(lambda: ms.ask_memory(q, k=5, compose=False, expand=False, rerank=False), max(5, rounds // 2))
    out["memory_stats"] = timed(ms.memory_stats, rounds)
    out["field_map"] = timed(ms.field_map, max(5, rounds // 2))
    out["list_preferences"] = timed(ms.list_preferences, rounds)

    # ---- 验真器: 200 chunks + 30 引用 ----
    gt = {f"PA_Sec{i % 8}_C{i:03d}": (f"Chunk {i} discusses DBSCAN clustering for DNA damage and LET based beam quality." * 4) for i in range(200)}
    gt.update({f"PA_{s}": " ".join(gt[f"PA_Sec{i % 8}_C{i:03d}"] for i in range(s, 200, 8)) for s in range(8)})
    section_chunks = {f"PA_{s}": [gt[f"PA_Sec{i % 8}_C{i:03d}"] for i in range(s, 200, 8)] for s in range(8)}
    draft = "\n".join(
        f"本节结论 number {i}: DBSCAN clustering improves DNA damage assessment [Ref: PA, Sec {i % 8}]。"
        for i in range(30))
    v = CitationVerifierV5(gt, {}, semantic_scorer=make_scorer(gt),
                           threshold=0.6, review_threshold=0.5, section_chunks=section_chunks)
    out["verify_30_cites"] = timed(lambda: v.verify_draft(draft), max(5, rounds // 2))

    # ---- 端到端 (offline) ----
    from paperbrain.pipeline import run_paper
    def one_run():
        d = tempfile.mkdtemp(dir=TMP)
        run_paper(str(ROOT / "demo" / "sample_paper.txt"), "B01", d, use_llm=False,
                  confirm_outline=True)
    out["run_paper_offline"] = timed(one_run, 5)

    print(json.dumps({k: (round(v[0], 2) if isinstance(v, tuple) else round(v, 2))
                      for k, v in out.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
