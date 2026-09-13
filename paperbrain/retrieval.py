"""离线检索引擎 (v5.0 生产版): TF-IDF chunk 召回 + 余弦语义打分.

替代 pipeline 里的 Jaccard 桩. stdlib only, 无需 embedding API.
- build_index(chunks) -> index {terms, df, vecs, norms}
- retrieve(query, index, top_k) -> [(chunk_id, score)]
- tfidf_scorer(claim, source) -> [0,1], 可直接注入 CitationVerifierV5
生产路径: scorer 可换 embedding cos/NLI；跨笔记召回由 memory_store 的 DuckDB VSS 路径执行。
"""
import hashlib
import math
import re
from functools import lru_cache
from typing import Dict, List, Tuple

_TOK = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+(?:[-_][A-Za-z0-9\u4e00-\u9fff]+)*")


@lru_cache(maxsize=8192)
def tokenize(s: str) -> Tuple[str, ...]:
    """分词 (带 LRU 缓存; 返回 tuple 防调用方误改)。验真器重复调用热点。"""
    return tuple(t.lower() for t in _TOK.findall(s or ""))


def build_index(chunks: Dict[str, str]) -> Dict:
    df: Dict[str, int] = {}
    tf: Dict[str, Dict[str, int]] = {}
    for cid, text in chunks.items():
        seen = set()
        vec: Dict[str, int] = {}
        for t in tokenize(text):
            vec[t] = vec.get(t, 0) + 1
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)
        tf[cid] = vec
    n = max(1, len(chunks))
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    vecs, norms = {}, {}
    for cid, vec in tf.items():
        w = {t: (1 + math.log(c)) * idf[t] for t, c in vec.items()}
        vecs[cid] = w
        norms[cid] = math.sqrt(sum(v * v for v in w.values())) or 1.0
    return {"idf": idf, "vecs": vecs, "norms": norms, "n": n}


def _qvec(query: str, idf: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    tf: Dict[str, int] = {}
    for t in tokenize(query):
        tf[t] = tf.get(t, 0) + 1
    w = {t: (1 + math.log(c)) * idf.get(t, math.log(2) + 1.0) for t, c in tf.items()}
    return w, math.sqrt(sum(v * v for v in w.values())) or 1.0


def retrieve(query: str, index: Dict, top_k: int = 3) -> List[Tuple[str, float]]:
    qv, qn = _qvec(query, index["idf"])
    out = []
    for cid, wv in index["vecs"].items():
        dot = sum(qv.get(t, 0.0) * w for t, w in wv.items())
        out.append((cid, dot / (qn * index["norms"][cid])))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]


def recall_at_k(ranked: List, positives, k: int) -> float:
    """前 k 命中正例的比例 (二值)。positives 为空 → 1.0 (无可召回, 不计罚)。"""
    ps = set(positives)
    if not ps:
        return 1.0
    top = list(ranked)[:k]
    return len([x for x in top if x in ps]) / len(ps)


def mrr(ranked: List, positives) -> float:
    """首个正例的倒数排名; 无命中 → 0。"""
    ps = set(positives)
    for i, x in enumerate(ranked):
        if x in ps:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: List, positives, k: int) -> float:
    """二值 nDCG@k (同分序稳定; 无正例 → 1.0)。"""
    ps = set(positives)
    if not ps:
        return 1.0
    top = list(ranked)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, x in enumerate(top) if x in ps)
    ideal = [1.0] * min(len(ps), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(ideal)))
    return dcg / idcg if idcg else 0.0


@lru_cache(maxsize=2048)
def _index_of(source: str) -> Dict:
    """单文本微型索引缓存 (验真 Stage-1 会对同一 chunk 反复建索引, 实测 3-5× 提速)。"""
    return build_index({"s": source})


def tfidf_scorer(claim: str, source: str) -> float:
    """claim vs source 的蕴含近似: 以 source 为库建微型索引查 claim."""
    idx = _index_of(source)
    qv, qn = _qvec(claim, idx["idf"])
    wv = idx["vecs"]["s"]
    dot = sum(qv.get(t, 0.0) * w for t, w in wv.items())
    cos = dot / (qn * idx["norms"]["s"])
    # 关键词覆盖加成: claim 实词在 source 中的覆盖率
    ct = set(tokenize(claim))
    st = set(tokenize(source))
    cover = len(ct & st) / max(1, len(ct))
    return min(1.0, 0.65 * cos + 0.35 * cover)


def _blend(qv, qn, sv, sn, ct, st, idf) -> float:
    dot = sum(qv.get(t, 0.0) * w for t, w in sv.items())
    cos = dot / (qn * sn) if qn * sn else 0.0
    # IDF 加权包含度: claim 实词在 source 中的权重占比 (逐字引用应≈1)
    tot = sum(idf.get(t, 1.0) for t in ct) or 1.0
    cover_w = sum(idf.get(t, 1.0) for t in ct if t in st) / tot
    s = 0.25 * cos + 0.75 * cover_w
    if len(ct) < 6:  # 过短断言封顶 (防 "效果很好" 类空话刷分)
        s *= 0.7
    return min(1.0, s)


def make_scorer(corpus: Dict[str, str]):
    """用全库 chunk 做 IDF 的打分器闭包 (比单篇 tfidf_scorer 更准)。
    接口与 SemanticScorer 一致, 可直接注入 CitationVerifierV5。
    内部缓存 (文本 hash -> 加权向量/词集): 验真器同一 claim/source 会被打分数次。
    """
    idx = build_index(corpus)
    cache: Dict[bytes, tuple] = {}

    def _cached(text: str):
        # 完整内容摘要；旧的 (前200字符, 长度) 会把不同长文本错误复用为同一向量。
        key = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        hit = cache.get(key)
        if hit is None:
            qv, qn = _qvec(text, idx["idf"])
            hit = (qv, qn, frozenset(tokenize(text)))
            if len(cache) > 4096:
                cache.clear()
            cache[key] = hit
        return hit

    def score(claim: str, source: str) -> float:
        qv, qn, ct = _cached(claim or "")
        sv, sn, st = _cached(source or "")
        return _blend(qv, qn, sv, sn, ct, st, idx["idf"])

    return score
