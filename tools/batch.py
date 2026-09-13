"""批量跑多篇 CLI (v5.0): 委托 paperbrain.batch 核心, 支持断点续跑/任务/深度.

用法:
  PYTHONPATH=. python3 tools/batch.py <目录或文件...> <输出目录> \
      [--pattern "*.pdf"] [--task full_read] [--depth standard] [--force]
已有同输入/代码/选项且产物哈希完整的论文自动跳过。full_read/deep 只产出待确认
大纲，不会绕过逐篇确认直接生成草稿。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="输入文件/目录 (目录按 --pattern 展开)")
    ap.add_argument("--pattern", default="*.pdf")
    ap.add_argument("--task", default="full_read")
    ap.add_argument("--depth", default="standard", choices=["fast", "standard", "deep"])
    ap.add_argument("--focus", default="")
    ap.add_argument("--out", default="out/batch")
    ap.add_argument("--force", action="store_true", help="忽略已有产物, 全部重跑")
    a = ap.parse_args()

    from paperbrain.batch import collect_files, run_batch
    files = []
    for item in a.inputs:
        p = Path(item)
        if p.is_dir():
            files += collect_files(dirs=[str(p)], pattern=a.pattern)
        elif p.is_file():
            files.append(str(p))
    if not files:
        print("no input files (pattern=%s)" % a.pattern)
        return 2
    print("batch: %d files -> %s (task=%s depth=%s resume=%s)"
          % (len(files), a.out, a.task, a.depth, not a.force))

    def on_prog(i, n, row):
        print("[%d/%d] %s -> %s%s (%.1fs)%s"
              % (i, n, row["paper_id"], row["verify"], 
                 "" if not row.get("notes") else f" notes={row['notes']}", 
                 row.get("elapsed", 0),
                 ("  ERR: " + row["error"]) if row.get("error") else ""))

    r = run_batch([{"path": f} for f in files], a.out, task=a.task, depth=a.depth,
                  focus=a.focus, resume=not a.force, on_progress=on_prog)
    print(f"done: {r['done']} done, {r['skipped']} skipped, {r['failed']} failed "
          f"-> {r['ledger']}")
    return 0 if r["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
