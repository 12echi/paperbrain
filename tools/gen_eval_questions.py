#!/usr/bin/env python3
"""模糊评测问题生成 (一次性, 结果缓存复用): 依据笔记让模型写口语化自然问题。

- 每个笔记生成 zh (中文改写) 与可选 en (英文跨语种) 问题; 正例 = 该笔记 + 图邻居。
- 增量写入 out/eval_questions.json (中断可续跑, 已有 note_id 跳过)。
- 生成后由 `tools/eval_retrieval.py --questions out/eval_questions.json` 使用 (离线复跑)。

用法: PYTHONPATH=. python3 tools/gen_eval_questions.py --n 20 [--db out/eval_snapshot.sqlite] [--en] [--out out/eval_questions.json]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_json(path: Path, data) -> None:
    """原子写: 防中断产生截断 JSON 导致缓存全丢。"""
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="生成问题的笔记条数")
    ap.add_argument("--db", default="out/eval_snapshot.sqlite")
    ap.add_argument("--out", default="out/eval_questions.json")
    ap.add_argument("--en", action="store_true", help="额外生成英文问题 (跨语种)")
    a = ap.parse_args()
    if a.db:
        os.environ["PAPERBRAIN_MEMORY_DB"] = a.db

    from paperbrain import llm
    from paperbrain import memory_store as ms
    from paperbrain import modelconf
    from tools.eval_retrieval import _build_cases

    modelconf.load()  # 与 server 一致: 注入 env.json 配置 (provider/model/embed)
    if not llm.has_key() and llm.provider() != "opencode-cli":
        print("无可用模型 (需要 PAPERBRAIN_API_KEY 或 opencode-cli), 拒绝生成")
        return 1

    out_path = Path(a.out)
    data = []
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"问题缓存损坏 ({e}), 拒绝覆盖: 请人工检查 {out_path} (可改名后重跑)")
            return 1
    have = {(d["note_id"], d["lang"]) for d in data}

    con = ms._conn()
    con.row_factory = __import__("sqlite3").Row
    cases = _build_cases(con, a.n)
    by_id = {c["id"]: c for c in cases}
    notes = {}
    for c in cases:
        row = con.execute("SELECT content FROM notes WHERE id=?", (c["id"],)).fetchone()
        notes[c["id"]] = (row["content"] if row else "")
    con.close()

    langs = ["zh"] + (["en"] if a.en else [])
    todo = [(nid, lg) for nid in by_id for lg in langs if (nid, lg) not in have]
    print(f"计划生成 {len(todo)} 个问题 (已有 {len(data)} 条缓存)")
    for i, (nid, lg) in enumerate(todo):
        t0 = time.time()
        try:
            q = llm.gen_eval_question(notes.get(nid, ""), lg)
        except Exception as e:
            print(f"  [{i+1}/{len(todo)}] note {nid}/{lg} 失败: {str(e)[:80]}")
            continue
        q = (q or "").strip().strip('"').split("\n")[0][:300]
        if len(q) < 6:
            print(f"  [{i+1}/{len(todo)}] note {nid}/{lg} 生成过短, 跳过")
            continue
        c = by_id[nid]
        data.append({"note_id": nid, "paper_id": c["paper_id"], "lang": lg,
                     "q": q, "pos": sorted(c["pos"])})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(out_path, data)
        print(f"  [{i+1}/{len(todo)}] note {nid}/{lg} ({time.time()-t0:.1f}s): {q[:60]}")
    print(f"完成: {len(data)} 条 → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
