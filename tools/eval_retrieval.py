#!/usr/bin/env python3
"""检索评测 (离线, 确定性): 给"混合×三因子×图多跳"做 A/B 量化, 避免凭感觉优化。

构造: 从记忆库抽 N 条笔记, 以其关键词为查询, 正例 = 该笔记 + 图邻居(互链/supports/extends)。
变体: bm25 → hybrid(RRF+向量) → threefactor(+重要性/近因/偏好) → ppr(+图多跳)。
指标: Recall@k / MRR / nDCG@k。

用法: PYTHONPATH=. python3 tools/eval_retrieval.py [--n 30] [--k 5] [--db out/memory.sqlite] [--json out/eval_retrieval.json]
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_cases(con, n: int, seed: int = 42):
    rows = [dict(r) for r in con.execute(
        "SELECT id, paper_id, kind, content, keywords, links FROM notes "
        "WHERE kind!='entity' AND COALESCE(status,'active') IN ('active','contested') "
        "AND invalid_at IS NULL AND keywords IS NOT NULL AND keywords!='[]'")]
    valid = {r["id"] for r in rows}
    typed = {}
    for s, d, rel in con.execute("SELECT src,dst,rel FROM note_links"):
        if rel == "contradicts":
            continue
        typed.setdefault(int(s), set()).add(int(d))
        typed.setdefault(int(d), set()).add(int(s))
    cases = []
    for r in rows:
        try:
            kws = [str(k) for k in json.loads(r["keywords"]) if len(str(k)) >= 2]
        except Exception:
            kws = []
        if len(kws) < 3 or len(r["content"] or "") < 20:
            continue
        q = " ".join(kws[:6])  # 关键词查询: 测"从模糊查询召回原笔记及邻域"
        pos = {r["id"]} | (typed.get(int(r["id"]), set()) & valid)  # 只保留可命中正例 (防幽灵 id)
        try:
            pos |= {int(i) for i in json.loads(r["links"] or "[]") if int(i) in valid}
        except Exception:
            pass
        cases.append({"id": r["id"], "paper_id": r["paper_id"], "q": q, "pos": pos})
    random.Random(seed).shuffle(cases)
    return cases[:n]


def _rank_bm25(con, q, pool=50):
    from paperbrain.memory_store import _bm25_ids
    return _bm25_ids(con, q, pool)


def _rank_hybrid(con, q, pool=50, w=0.1):
    from paperbrain.memory_store import _bm25_ids, _vector_ids, _rrf_scores
    rr = _rrf_scores([_bm25_ids(con, q, pool), _vector_ids(con, q, pool)], weights=[1.0, w])
    return [i for i, _ in sorted(rr.items(), key=lambda x: -x[1])]


def _rank_threefactor(con, q, pool=12):
    # 与 ppr 变体同池 (12): 隔离图扩展的净贡献, 避免 min-max 跨度差异造成伪差异
    from paperbrain.memory_store import search_notes
    return [h["id"] for h in search_notes(q, limit=pool)]


def _rank_rrf_w1(con, q, pool=12):
    """纯加权 RRF (BM25+向量, 权重 1:1), 池与生产一致: 隔离三因子/min-max 的净贡献。"""
    from paperbrain.memory_store import _bm25_ids, _vector_ids, _rrf_scores
    bm = _bm25_ids(con, q, pool)
    vec = _vector_ids(con, q, pool)
    lists, ws = [], []
    if bm:
        lists.append(bm)
        ws.append(1.0)
    if vec:
        lists.append(vec)
        ws.append(1.0)
    rr = _rrf_scores(lists, weights=ws)
    return [i for i, _ in sorted(rr.items(), key=lambda x: -x[1])]


def _rank_ppr(con, q, pool=50):
    from paperbrain.memory_store import search_notes, graph_expand
    hits = search_notes(q, limit=12)
    if not hits:
        return []
    base = {h["id"]: float(h.get("score") or 0.1) for h in hits}
    ppr = graph_expand(base, steps=2)
    have = set(base)
    extra = sorted(((i, s) for i, s in ppr.items() if i not in have), key=lambda x: -x[1])
    return [h["id"] for h in hits] + [i for i, _ in extra][:8]


def _load_questions(path: str, n: int, langs: str = ""):
    """载入模型生成的模糊问题 (正例随问题给出)。langs 过滤: 'zh,en'; n=0 表示不截断。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    want = {x.strip() for x in langs.split(",") if x.strip()}
    if want:
        data = [d for d in data if d.get("lang") in want]
    if n:
        data = data[:n]
    return [{"id": d["note_id"], "paper_id": d.get("paper_id", ""), "q": d["q"],
             "pos": set(d.get("pos") or []), "lang": d.get("lang", "")}
            for d in data if d.get("q") and d.get("pos")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="用例数 (0=关键词模式取40/问题模式全量)")
    ap.add_argument("--k", type=int, default=5, help="Recall/nDCG 的 k")
    ap.add_argument("--db", default="", help="记忆库路径 (默认 out/memory.sqlite)")
    ap.add_argument("--questions", default="", help="模型生成的模糊问题 JSON (gen_eval_questions.py)")
    ap.add_argument("--langs", default="", help="只评某语言子集: zh,en")
    ap.add_argument("--json", default="", help="结果写 JSON")
    a = ap.parse_args()
    if a.db:
        os.environ["PAPERBRAIN_MEMORY_DB"] = a.db
    from paperbrain import memory_store as ms
    from paperbrain.retrieval import recall_at_k, mrr, ndcg_at_k
    from paperbrain import config

    con = ms._conn()
    con.row_factory = __import__("sqlite3").Row
    if a.questions:
        cases = _load_questions(a.questions, a.n, a.langs)
        mode = "模糊问题"
    else:
        cases = _build_cases(con, a.n or 40)
        mode = "关键词自检索"
    if not cases:
        print("库内可用笔记不足, 无法评测")
        return 1
    # 正例校验: 过滤幽灵 id (已删除) 与不可检索状态 — 历史库 links 常含死 id, 否则指标被系统性稀释
    valid = {int(r[0]) for r in con.execute(
        "SELECT id FROM notes WHERE kind!='entity' AND COALESCE(status,'active')"
        " IN ('active','contested') AND invalid_at IS NULL")}
    clean = []
    for c in cases:
        c = dict(c)
        c["pos"] = set(c["pos"]) & valid
        if c["pos"]:
            clean.append(c)
    dropped = len(cases) - len(clean)
    cases = clean
    if dropped:
        print(f"[info] 正例校验: 丢弃 {dropped} 个无可命中正例的用例, 余 {len(cases)}")
    if not cases:
        print("校验后无用例 (快照与问题集可能不匹配)")
        return 1
    # 向量预检: 端点不可达时立即降级, 避免逐查询超时 (隧道断开会表现为"卡住")
    embed = False
    try:
        embed = bool(config.embed_enabled())
    except Exception:
        pass
    if embed:
        try:
            from paperbrain.embeddings import embed_texts
            probe = embed_texts(["vector-probe"])
            if not probe or probe[0] is None:
                print("[warn] embedding 端点不可达, 已降级 (恢复隧道: bash tools/embed_tunnel.sh start)")
                os.environ["PAPERBRAIN_EMBED_BASE_URL"] = ""
                embed = False
        except Exception:
            embed = False
    variants = [("bm25", _rank_bm25), ("hybrid", _rank_hybrid), ("rrf_w1", _rank_rrf_w1),
                ("threefactor", _rank_threefactor), ("ppr", _rank_ppr)]
    results = {}
    for name, fn in variants:
        rec = mmr_ = nd = 0.0
        for c in cases:
            ranked = fn(con, c["q"])
            rec += recall_at_k(ranked, c["pos"], a.k)
            mmr_ += mrr(ranked, c["pos"])
            nd += ndcg_at_k(ranked, c["pos"], a.k)
        n = len(cases)
        results[name] = {"recall@%d" % a.k: round(rec / n, 4),
                         "mrr": round(mmr_ / n, 4),
                         "ndcg@%d" % a.k: round(nd / n, 4)}
    from paperbrain.memory_store import _bm25_ids
    lex = sum(len(_bm25_ids(con, c["q"], 40)) for c in cases) / max(1, len(cases))
    con.close()
    print("=" * 66)
    print(f"检索评测[{mode}]: {len(cases)} 用例, k={a.k}, 向量={'开' if embed else '关(hybrid==bm25)'}, "
          f"平均词面命中 {lex:.1f}/40")
    print(f"{'变体':<14}{'Recall@'+str(a.k):<12}{'MRR':<12}{'nDCG@'+str(a.k):<12}")
    for name, _ in variants:
        r = results[name]
        print(f"{name:<14}{r['recall@%d' % a.k]:<12}{r['mrr']:<12}{r['ndcg@%d' % a.k]:<12}")
    print("=" * 66)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"cases": len(cases), "k": a.k, "embed": embed, "results": results},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print("已写", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
