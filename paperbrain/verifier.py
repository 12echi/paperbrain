"""三阶段引文门禁 V5 (生产重构版 - M3 收敛与缓存).

核心特性 (F09, F10, F11):
1. F09 两阶段粗精排匹配:
   - 阶段一 (词法粗排): 废除针对多达 12 分块的全量笛卡尔积双调用 (原至多 24 次/引用),
     利用本地零开销 tfidf_scorer 对候选分块进行相关度排序并严格收敛至 Top-2.
   - 阶段二 (语义精排): 仅针对 Top-2 分块调用语义打分器 (semantic_scorer),
     严格约束单引用 semantic_scorer 实际调用次数 <= 2 次.
2. F10 5元组精确结果缓存层:
   - 基于 (citation_text, context, model_tag, prompt_version, chunk_hash) 的 SHA-256 5 元组缓存键.
   - 线程安全锁保护 (threading.Lock).
   - 重复验证调用 100% 缓存命中, 增量网络/模型调用严格为 0 (call delta == 0).
   - 上下文/Claim 变更时缓存安全失效, 正确触发新评估.
3. F11 生产级三阶段门禁状态机:
   - 全/半角括号及逗号兼容.
   - Fig/Tab 编号精确校验与全局清单反查.
   - 单引用多图表严格拒绝 (UNVERIFIED).
   - 语义蕴含阈值判定 (PASS >= 0.82, NEEDS_REVIEW [0.35, 0.82), UNVERIFIED < 0.35).
   - 全局状态机: CLEAN / DIRTY / NEEDS_REVIEW / NO_CITATION.
"""
import hashlib
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ids import parse_loc, make_cache_key
from .text import split_spans
from .retrieval import tfidf_scorer

SemanticScorer = Callable[[str, str], float]

# 兼容 [] 【】, Ref 大小写, 全/半角冒号逗号
CIT_PAT = re.compile(r"[\[【]\s*ref\s*[:：]\s*([^,\]】，]+?)\s*[,，]\s*([^\]】]+?)\s*[\]】]", re.I)


def _uncited_long_paragraphs(text: str, min_chars: int = 200) -> List[str]:
    """M5 门禁：超过 min_chars 个有效字符的正文段落必须至少含一条规范引用。"""
    bad: List[str] = []
    for block in re.split(r"\n\s*\n", text or ""):
        if CIT_PAT.search(block):
            continue
        lines = [line for line in block.splitlines()
                 if line.strip() and not line.lstrip().startswith(("#", "```"))]
        plain = re.sub(r"\s+", "", re.sub(r"[*_>`~-]", "", "\n".join(lines)))
        if len(plain) > min_chars:
            bad.append(plain[:160])
    return bad


def extract_claim(text: str, pos: int, end: int, window: int = 500) -> Dict:
    """取引用所在句为 claim 上下文, 避免固定字符窗切断语义."""
    start = max(0, pos - window)
    stop = min(len(text), end + window)
    frag = text[start:stop]
    # 按句切带原文跨度, 引用位置精确命中所在句 (防空白漂移串句)
    spans = split_spans(frag)
    if not spans:  # 空片段兜底, 防 IndexError
        return {"sentence": frag.strip(), "window": frag.strip()}
    sents = [s for s, _, _ in spans]
    rel = pos - start
    hit = 0
    for i, (_, a, b) in enumerate(spans):
        if a <= rel < b or (i == len(spans) - 1 and rel >= a):
            hit = i
            break
    lo = max(0, hit - 1)
    hi = min(len(sents), hit + 2)
    window_txt = "".join(sents[lo:hi]).strip()
    # 引用独占一句时 (如 "...settings. [Ref:X]。"), 取证以前一句实质句为准
    sent = sents[hit].strip()
    bare = CIT_PAT.sub("", sent).strip(" 。.!?；;:\n\t")
    if len(bare) < 20 and hit > 0:
        sent = (sents[hit - 1].strip() + " " + sent).strip()
    return {"sentence": sent, "window": window_txt or frag.strip()}


class CitationVerifierV5:
    def __init__(self, ground_truth_chunks: Dict[str, str],
                 fig_index: Optional[Dict[str, Dict[str, List[str]]]] = None,
                 semantic_scorer: Optional[SemanticScorer] = None,
                 threshold: float = 0.82, review_threshold: float = 0.35,
                 section_chunks: Optional[Dict[str, List[str]]] = None,
                 model_tag: str = "v5_semantic_nli",
                 prompt_version: str = "v5.0",
                 cache: Optional[Dict[str, float]] = None):
        # threshold: PASS 线 (embedding/NLI 用 0.82, 词法打分器用标定值见 tools/calibrate.py)
        # review_threshold: 低于此直接 UNVERIFIED, 两线之间 NEEDS_REVIEW (人工/强模型复核)
        # section_chunks: seckey -> 该节各 chunk 全文; 有则节引用取节内最强块 (防长节稀释)
        # 精确键 + 小写映射 (用于 case-drift 二次查)
        self.gt = {k.strip(): v for k, v in ground_truth_chunks.items()}
        self.gt_lower = {k.strip().lower(): (k, v) for k, v in ground_truth_chunks.items()}
        self.fig_index = fig_index or {}
        # 预建全局图表清单: PaperID(lower) -> set(fig/tab)
        self.global_figs: Dict[str, set] = {}
        for k, v in self.fig_index.items():
            # k 形如 PaperID_Sec
            pid = k.rsplit("_", 1)[0] if "_" in k else k
            s = self.global_figs.setdefault(pid.lower(), set())
            s.update([x.lower() for x in v.get("figs", [])])
            s.update([x.lower() for x in v.get("tabs", [])])
        self.scorer = semantic_scorer
        self.th = threshold
        self.review_th = review_threshold
        self.section_chunks = section_chunks or {}
        self.model_tag = model_tag
        self.prompt_version = prompt_version
        self.cache: Dict[str, float] = cache if cache is not None else {}
        self._lock = threading.Lock()

    def clear_cache(self) -> None:
        """清空验真结果缓存."""
        with self._lock:
            self.cache.clear()

    def _lookup(self, key: str):
        if key in self.gt:
            return self.gt[key], False
        hit = self.gt_lower.get(key.lower())
        if hit:
            return hit[1], True
        return None, False

    def verify_draft(self, draft_text: str) -> Dict:
        reports: List[Dict] = []
        seen: Dict[str, int] = {}
        for m in CIT_PAT.finditer(draft_text):
            raw_ref = m.group(1).strip()
            raw_loc = m.group(2).strip()
            cit_raw = m.group(0)
            seen[cit_raw] = seen.get(cit_raw, 0) + 1

            sec_norm, figs, tabs, has_multi = parse_loc(raw_loc)
            all_ft = figs + tabs

            # P0: 单引用多图表直接打回
            if has_multi:
                reports.append({"citation": cit_raw, "status": "UNVERIFIED",
                                "reason": f"单引用含多图表 {all_ft}, 必须拆分为单图引用"})
                continue

            fig_norm = all_ft[0] if all_ft else None

            # 存在性: Sec+Fig -> 查 Sec 键; Fig-only -> 查 Fig 键; Sec-only -> 查 Sec 键
            if sec_norm:
                key = f"{raw_ref}_{sec_norm}"
            elif fig_norm:
                key = f"{raw_ref}_{fig_norm}"
            else:
                reports.append({"citation": cit_raw, "status": "UNVERIFIED",
                                "reason": f"无法解析 loc: {raw_loc}"})
                continue

            src, case_drift = self._lookup(key)
            if src is None:
                reports.append({"citation": cit_raw, "status": "UNVERIFIED",
                                "reason": f"{key} 不在有效片段中"})
                continue
            if case_drift:
                reports.append({"citation": cit_raw, "status": "NEEDS_REVIEW",
                                "reason": f"PaperID大小写漂移: {raw_ref}, 期望精确匹配",
                                "key": key})
                continue

            # 编号精确: Sec+Fig 查节内清单; Fig-only 查全局清单
            if sec_norm and fig_norm:
                idx = self.fig_index.get(f"{raw_ref}_{sec_norm}") or \
                    self.fig_index.get(f"{raw_ref.lower()}_{sec_norm.lower()}", {})
                allowed = set([x.lower() for x in idx.get("figs", [])] +
                              [x.lower() for x in idx.get("tabs", [])])
                if fig_norm.lower() not in allowed:
                    reports.append({"citation": cit_raw, "status": "UNVERIFIED",
                                    "reason": f"{fig_norm} 不在 {raw_ref} Sec {sec_norm} 图表清单"})
                    continue
            elif fig_norm and not sec_norm:
                g = self.global_figs.get(raw_ref.lower(), set())
                if g and fig_norm.lower() not in g:
                    reports.append({"citation": cit_raw, "status": "UNVERIFIED",
                                    "reason": f"{fig_norm} 不在 {raw_ref} 全局图表清单"})
                    continue
                # g 为空 (无清单) 则降级为存在性通过 + 标记, 不直接 PASS
                if not g:
                    reports.append({"citation": cit_raw, "status": "NEEDS_REVIEW",
                                    "reason": f"{raw_ref} 缺图表清单, Fig-only 无法精确核验",
                                    "key": key})
                    continue

            # 语义蕴含: 无 scorer -> NEEDS_REVIEW (P0 修复, 不再 PASS)
            if self.scorer is None:
                reports.append({"citation": cit_raw, "status": "NEEDS_REVIEW",
                                "reason": "缺语义打分器, 仅通过存在性",
                                "source_excerpt": src[:150] + ("..." if len(src) > 150 else "")})
                continue

            claim = extract_claim(draft_text, m.start(), m.end())
            s_txt = CIT_PAT.sub("", claim["sentence"] or claim["window"]).strip()
            w_txt = CIT_PAT.sub("", claim["window"]).strip()
            claim_text = s_txt if len(s_txt) >= 20 else (w_txt or s_txt)
            if not claim_text:
                claim_text = cit_raw

            # 候选分块收集与保序去重
            srcs = self.section_chunks.get(key, [src]) or [src]
            seen_chunks = set()
            unique_srcs: List[str] = []
            for s in srcs:
                if s not in seen_chunks:
                    seen_chunks.add(s)
                    unique_srcs.append(s)

            # Stage 1: 词法粗排 (Lexical Coarse Retrieval) -> 严格收敛至 Top-2
            if len(unique_srcs) <= 2:
                top2 = unique_srcs
            else:
                coarse_scores: List[Tuple[float, str]] = []
                for chunk in unique_srcs:
                    try:
                        c_score = tfidf_scorer(claim_text, chunk)
                    except Exception:
                        c_score = 0.0
                    coarse_scores.append((c_score, chunk))
                coarse_scores.sort(key=lambda x: x[0], reverse=True)
                top2 = [chunk for _, chunk in coarse_scores[:2]]

            # Stage 2: 精确语义蕴含判定 (Semantic Verification)
            # 严格约束: semantic_scorer 调用次数每条引文严格 <= 2 次
            # 集成 5 元组 SHA-256 结果缓存层 (线程安全)
            best_score = 0.0
            best_chunk = top2[0] if top2 else src
            try:
                for chunk in top2:
                    chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
                    cache_key = make_cache_key(
                        cit_raw,
                        claim_text,
                        self.model_tag,
                        self.prompt_version,
                        chunk_hash
                    )
                    with self._lock:
                        cached_score = self.cache.get(cache_key)

                    if cached_score is not None:
                        score = cached_score
                    else:
                        score = float(self.scorer(claim_text, chunk))
                        with self._lock:
                            self.cache[cache_key] = score

                    if score > best_score:
                        best_score = score
                        best_chunk = chunk

                score = best_score
            except Exception as e:
                reports.append({"citation": cit_raw, "status": "NEEDS_REVIEW",
                                "reason": f"语义打分异常: {e}"})
                continue

            if score < self.review_th:
                reports.append({"citation": cit_raw, "status": "UNVERIFIED",
                                "reason": f"语义蕴含 {score:.2f} < {self.review_th} (复核线)",
                                "score": round(score, 3),
                                "source_excerpt": best_chunk[:150] + ("..." if len(best_chunk) > 150 else "")})
            elif score < self.th:
                reports.append({"citation": cit_raw, "status": "NEEDS_REVIEW",
                                "reason": f"语义蕴含 {score:.2f} 在复核带 [{self.review_th},{self.th})，需人工/强模型复核",
                                "score": round(score, 3),
                                "source_excerpt": best_chunk[:150] + ("..." if len(best_chunk) > 150 else "")})
            else:
                reports.append({"citation": cit_raw, "status": "PASS",
                                "score": round(score, 3),
                                "source_excerpt": best_chunk[:150] + ("..." if len(best_chunk) > 150 else "")})

        for excerpt in _uncited_long_paragraphs(draft_text):
            reports.append({"citation": "[UNCITED_LONG_PARAGRAPH]", "status": "UNVERIFIED",
                            "reason": ">200字正文段落缺少规范引用", "source_excerpt": excerpt})

        if not reports:
            return {"status": "NO_CITATION", "is_clean": False, "report": []}
        # 去重提示 (不改变状态, 供上游去重)
        dup = {k: c for k, c in seen.items() if c > 1}
        if any(r["status"] == "UNVERIFIED" for r in reports):
            st = "DIRTY"
        elif any(r["status"] == "NEEDS_REVIEW" for r in reports):
            st = "NEEDS_REVIEW"
        else:
            st = "CLEAN"
        out: Dict = {"status": st, "is_clean": st == "CLEAN", "report": reports}
        if dup:
            out["duplicates"] = dup
        return out
