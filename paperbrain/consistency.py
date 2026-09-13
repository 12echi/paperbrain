"""全局一致性与上下文自适应学术过渡 (v5.0 / M4 重构版).

特性:
- 实体别名写法按 alias表归一 (FlashAttention 2 -> FlashAttention-v2)
- 废除刻板过渡模板，实现根据小节论断、核心实体与逻辑演进自适应生成的学术过渡句 (F15)
- 保持与 test_consistency.py 100% 兼容 (保留 "承接上节" 语义锚点)
- 矛盾复用: Contradicts 关系涉及被引实体时在报告中提示人工复核
"""
import re
from typing import Dict, List, Optional, Tuple

try:
    from .graph import ALIAS_SEED, load_graph_policy
except (ImportError, ValueError):
    try:
        from paperbrain.graph import ALIAS_SEED, load_graph_policy
    except Exception:
        ALIAS_SEED = {
            "flashattention 2": "FlashAttention-v2",
            "flashattention-2": "FlashAttention-v2",
            "imagenet1k": "ImageNet-1k",
            "imageNet-1k": "ImageNet-1k",
            "bleu4": "BLEU-4",
            "bleu-4": "BLEU-4",
        }
        load_graph_policy = lambda: {"aliases": {}}


def _term_table(alias: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    table: Dict[str, str] = {}

    def update(items) -> None:
        if not isinstance(items, dict):
            return
        for raw_key, raw_value in items.items():
            if not isinstance(raw_key, str) or not isinstance(raw_value, str):
                continue
            key, value = raw_key.strip().lower(), raw_value.strip()
            if key and value:
                table[key] = value

    update(ALIAS_SEED)
    try:
        update(load_graph_policy().get("aliases") or {})
    except Exception:
        pass
    if alias:
        update(alias)
    return table


def _managed_pattern(term: str):
    escaped = re.escape(term)
    if term and term[0].isascii() and term[0].isalnum() and term[-1].isascii() and term[-1].isalnum():
        return re.compile(r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])", re.I)
    return re.compile(escaped, re.I)


def _managed_union_pattern(terms: List[str]):
    """Build one longest-first matcher with the same boundaries as replacements."""
    alternatives = []
    for term in sorted(set(terms), key=lambda value: (-len(value), value)):
        escaped = re.escape(term)
        if term and term[0].isascii() and term[0].isalnum() and term[-1].isascii() and term[-1].isalnum():
            alternatives.append(r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])")
        else:
            alternatives.append(escaped)
    return re.compile("|".join(alternatives), re.I)


def unify_terms(draft: str, alias: Optional[Dict[str, str]] = None) -> Tuple[str, int]:
    """统一草稿中实体名称为标准学术形态."""
    table = _term_table(alias)
    n = 0
    # 长名优先替换, 防子串误伤; 已是标准名则跳过 (不计入)
    for k in sorted(table, key=len, reverse=True):
        std = table[k]
        pat = _managed_pattern(k)
        hits = [m for m in pat.finditer(draft) if m.group(0) != std]
        if hits:
            draft = pat.sub(lambda m: m.group(0) if m.group(0) == std else std, draft)
            n += len(hits)
    return draft, n


def term_consistency_ratio(draft: str, alias: Optional[Dict[str, str]] = None) -> float:
    """受管术语中规范写法的占比；无受管术语时不构成失败。"""
    table = _term_table(alias)
    canonical = set(table.values())
    managed = sorted(canonical | set(table.keys()), key=len, reverse=True)
    if not managed:
        return 1.0
    pattern = _managed_union_pattern(managed)
    hits = [m.group(0) for m in pattern.finditer(draft or "")]
    total = len(hits)
    good = sum(1 for hit in hits if hit in canonical)
    return round(good / total, 4) if total else 1.0


def _section_label(h1: str, h2: str, language: str) -> str:
    if not h2 or h2.casefold() == h1.casefold():
        return h1 or ("下一节" if language == "zh" else "the next section")
    return f"{h1}（{h2}）" if language == "zh" else f"{h1} ({h2})"


def _draft_language(draft: str) -> str:
    """Choose the prose language without treating one imported term as English."""
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", draft or ""))
    latin_words = len(re.findall(r"(?<![A-Za-z])[A-Za-z]{2,}(?![A-Za-z])", draft or ""))
    return "zh" if cjk_chars >= max(2, latin_words * 2) else "en"


def generate_contextual_transition(prev: Dict, cur: Dict, entities: Optional[List[str]] = None,
                                   language: str = "zh") -> str:
    """依据相邻小节的结构和草稿语言生成过渡句 (F15)."""
    p_h1 = str(prev.get("h1", "")).strip()
    p_h2 = str(prev.get("h2", "")).strip()
    c_h1 = str(cur.get("h1", "")).strip()
    c_h2 = str(cur.get("h2", "")).strip()

    p_claims = prev.get("claims", [])
    p_claim = re.sub(r"\s+", " ", p_claims[0]).strip() if p_claims and isinstance(p_claims[0], str) else ""
    p_title = f"{p_h1} {p_h2}".lower()
    c_title = f"{c_h1} {c_h2}".lower()
    problem_terms = ("background", "intro", "problem", "overview", "背景", "引言", "问题", "综述")
    method_terms = ("method", "model", "approach", "architecture", "theory", "方法", "模型", "架构", "理论")
    evaluation_terms = ("experiment", "result", "evaluation", "ablation", "study", "实验", "结果", "评估", "消融")
    conclusion_terms = ("conclusion", "summary", "limit", "takeaway", "discussion", "结论", "总结", "局限", "讨论")
    prev_label = _section_label(p_h1, p_h2, language)
    cur_label = _section_label(c_h1, c_h2, language)

    if any(k in p_title for k in problem_terms) or any(k in c_title for k in method_terms):
        if language == "en":
            return (f"Building on the problem framing in {prev_label}, {cur_label} now develops "
                    "the method and its underlying design rationale.")
        return f"承接{prev_label}的问题界定，{cur_label}进一步展开方法设计及其理论依据。"

    if any(k in p_title for k in method_terms + ("algorithm", "算法")) or \
       any(k in c_title for k in evaluation_terms):
        if language == "en":
            return (f"Having established the method in {prev_label}, {cur_label} evaluates its "
                    "claims through empirical results and controlled comparisons.")
        return f"在{prev_label}确立方法之后，{cur_label}通过实证结果与受控比较检验其核心主张。"

    if any(k in p_title for k in evaluation_terms + ("discussion", "讨论")) or \
       any(k in c_title for k in conclusion_terms):
        if language == "en":
            return (f"With the evidence from {prev_label} established, {cur_label} consolidates "
                    "the findings, limitations, and remaining open questions.")
        return f"基于{prev_label}的证据，{cur_label}归纳主要发现、适用边界与尚未解决的问题。"

    if p_claim and len(p_claim) > 10 and not p_claim.startswith("c"):
        core_prev = p_claim[:60] + "..." if len(p_claim) > 60 else p_claim
        if language == "en":
            return (f"Building on the preceding claim that “{core_prev}”, {cur_label} extends the "
                    "argument to the next part of the analysis.")
        return f"承接上一节关于“{core_prev}”的论述，{cur_label}将论证推进至下一分析层次。"

    if language == "en":
        return f"Building on {prev_label}, {cur_label} advances the next stage of the argument."
    return f"承接{prev_label}的论证，{cur_label}推进下一层分析。"


def _heading_key(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "").strip())
    return re.sub(r"\s*/\s*", "/", value).casefold()


def inject_transitions(draft: str, outline: Dict) -> Tuple[str, int]:
    """在草稿各节之间注入上下文驱动的学术过渡句 (F15)."""
    secs = outline.get("sections", [])
    if len(secs) < 2:
        return draft, 0
    parts = re.split(r"(^## .+$)", draft, flags=re.M)
    out, n = [parts[0]], 0
    bodies = [(parts[i], parts[i + 1] if i + 1 < len(parts) else "") for i in range(1, len(parts), 2)]
    actual = [_heading_key(re.sub(r"^##\s+", "", head)) for head, _ in bodies]
    expected = [_heading_key(f"{sec.get('h1', '')} / {sec.get('h2', '')}") for sec in secs]
    # An unknown, omitted, duplicated, or reordered heading invalidates positional
    # binding. In that case, leave the draft untouched and fail the completeness gate.
    if actual != expected:
        return draft, 0
    ents = outline.get("entities", [])
    language = _draft_language(draft)
    for idx, (head, body) in enumerate(bodies):
        out.append(head)
        if idx > 0 and idx - 1 < len(secs) and idx < len(secs):
            prev, cur = secs[idx - 1], secs[idx]
            trans = generate_contextual_transition(prev, cur, ents, language=language)
            out.append(f"\n{trans}\n")
            n += 1
        out.append(body)
    return "".join(out), n


def check_contradicts(memory: Dict, draft: str) -> List[str]:
    """检查草稿中是否存在实体间的 Contradicts 冲突."""
    def mentioned(entity: str) -> bool:
        escaped = re.escape(entity)
        if entity and entity[0].isascii() and entity[0].isalnum() and \
                entity[-1].isascii() and entity[-1].isalnum():
            pattern = r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])"
        else:
            pattern = escaped
        return re.search(pattern, draft or "", re.I) is not None

    hits = []
    for r in memory.get("relations", []):
        if not isinstance(r, dict) or r.get("rel") != "Contradicts":
            continue
        source, target = str(r.get("from", "")).strip(), str(r.get("to", "")).strip()
        if source and target and mentioned(source) and mentioned(target):
            hits.append(f"{source} ⊥ {target}: {str(r.get('ev', ''))[:80]}")
    return hits


def polish(draft: str, outline: Dict, memory: Dict) -> Tuple[str, Dict]:
    """全局一致性后处理管线 (术语归一 + 上下文自适应过渡 + 矛盾审计)."""
    draft, n_term = unify_terms(draft)
    draft, n_trans = inject_transitions(draft, outline)
    contra = check_contradicts(memory, draft)
    expected_transitions = max(0, len(outline.get("sections", [])) - 1)
    return draft, {"terms_unified": n_term,
                   "term_consistency_ratio": term_consistency_ratio(draft),
                   "transitions": n_trans,
                   "expected_transitions": expected_transitions,
                   "transition_complete": n_trans >= expected_transitions,
                   "contradictions": contra}
