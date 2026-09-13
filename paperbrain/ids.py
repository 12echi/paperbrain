"""ID / 归一化规范 (v5.0 生产版).

PaperID: {YYYY}_{Venue}_{NN} e.g. 2024_NeurIPS_01, 大小写敏感, 入库不可变.
Sec: 数字节 3.2 / 附录 A.1 / Appendix A.1 / 附录A 统一归一.
Fig/Tab: Fig 2 / Figure 2 / 图2 -> fig2 ; Tab 3 / Table 3 / 表3 -> tab3.
缓存键必须带模型版本, 防脏缓存.
"""
import hashlib
import re
from typing import Optional

_PAPERID_RE = re.compile(r"^\d{4}_[A-Za-z0-9\-]+_\d{2,}$")

# 数字节: 3 / 3.2 / 3.2.1 ; 附录: A / A.1 / B.2
_SEC_NUM_RE = re.compile(r"(\d+(?:\.\d+)*)")
_SEC_APP_RE = re.compile(r"(?:appendix|附录)?\s*([A-Z])\s*\.?\s*(\d+(?:\.\d+)*)?", re.I)
_FIG_RE = re.compile(r"(fig(?:ure)?|图)\s*(\d+)", re.I)
_TAB_RE = re.compile(r"(tab(?:le)?|表)\s*(\d+)", re.I)


def is_valid_paper_id(pid: str) -> bool:
    return bool(_PAPERID_RE.match(pid.strip()))


def norm_sec(s: str) -> str:
    """归一 Sec. 支持: Sec 3.2 / Sec.3.2 / 3.2 / A.1 / Appendix A.1 / 附录A.1
    返回: 3.2 / A.1 ; 无法识别则返回 strip().lower() (调用方视为弱归一).
    """
    t = s.strip()
    # 1) Sec + 附录节号: Sec A.1 / Sec. B.2
    m = re.search(r"sec\.?\s*([A-Z]\s*\.\s*\d+(?:\.\d+)*)", t, re.I)
    if m:
        return re.sub(r"\s+", "", m.group(1).upper())
    # 2) Sec + 数字: Sec 3.2
    m = re.search(r"sec\.?\s*(\d+(?:\.\d+)*)", t, re.I)
    if m:
        return m.group(1)
    # 3) Appendix/附录 + 字母: Appendix A.1 / 附录A.1
    m = re.search(r"(?:appendix|附录)\s*([A-Z])(?:\s*\.\s*(\d+(?:\.\d+)*))?", t, re.I)
    if m:
        letter = m.group(1).upper()
        return f"{letter}.{m.group(2)}" if m.group(2) else letter
    # 4) 裸附录节号: A.1
    m = re.search(r"\b([A-Z]\s*\.\s*\d+(?:\.\d+)*)", t)
    if m:
        return re.sub(r"\s+", "", m.group(1).upper())
    # 5) 纯数字节
    m2 = _SEC_NUM_RE.search(t)
    if m2:
        return m2.group(1)
    return t.lower()


def norm_figtab(s: str) -> str:
    """Fig 2 / Figure 2 / 图2 -> fig2 ; Tab 3 / Table 3 / 表3 -> tab3."""
    t = s.strip()
    m = _FIG_RE.search(t)
    if m:
        return f"fig{m.group(2)}"
    m = _TAB_RE.search(t)
    if m:
        return f"tab{m.group(2)}"
    return t.lower()


def parse_loc(raw_loc: str) -> tuple:
    """解析引用 loc 部分, 返回 (sec_norm|None, figs_list, tabs_list, has_multi).
    loc 按中英文逗号/顿号切分. figs+tabs>=2 即 has_multi=True (生产门禁: 直接打回).
    """
    parts = [p.strip() for p in re.split(r"[,，、;；]", raw_loc) if p.strip()]
    sec_norm = None
    sec_count = 0
    figs: list[str] = []
    tabs: list[str] = []
    for p in parts:
        low = p.lower()
        is_sec = ("sec" in low) or ("附录" in p) or ("appendix" in low) or bool(re.match(r"^[A-Z]?\d*\.?\d*$", p.strip(), re.I) and re.search(r"\d|[A-Z]", p))
        # 含 fig/图 或 tab/表 优先判图表, 避免 "Sec 3.2" 误判
        if re.search(r"fig|图", p, re.I):
            figs.append(norm_figtab(p))
        elif re.search(r"tab|表", p, re.I):
            tabs.append(norm_figtab(p))
        elif is_sec or re.search(r"sec", p, re.I):
            sec_count += 1
            sec_norm = norm_sec(p)
        else:
            # 纯 "3.2" / "A.1" 视为 sec
            if re.search(r"\d|[A-Z]", p):
                sec_count += 1
                sec_norm = norm_sec(p)
    has_multi = (len(figs) + len(tabs) >= 2) or sec_count >= 2
    return sec_norm, figs, tabs, has_multi


def make_cache_key(
    citation_text: Optional[str] = None,
    context: Optional[str] = None,
    model_tag: Optional[str] = None,
    prompt_version: Optional[str] = None,
    chunk_hash: Optional[str] = None,
    *args,
    **kwargs
) -> str:
    """生产 5 元组 SHA-256 缓存键: 缺模型版本/必填字段必脏缓存, 故强制要求 5 元组.

    支持 F10 规范签名:
    make_cache_key(citation_text, context, model_tag, prompt_version, chunk_hash)
    同时兼容旧版签名:
    make_cache_key(pdf_sha, section_hash, prompt_version, model_version, embedding_version)
    """
    p1 = citation_text if citation_text is not None else kwargs.get("pdf_sha")
    p2 = context if context is not None else kwargs.get("section_hash")

    if "model_version" in kwargs and "prompt_version" in kwargs:
        p3 = kwargs.get("model_tag") or kwargs["model_version"]
        p4 = kwargs["prompt_version"]
    else:
        p3 = model_tag if model_tag is not None else kwargs.get("model_version")
        p4 = prompt_version if prompt_version is not None else kwargs.get("prompt_version")

    p5 = chunk_hash if chunk_hash is not None else kwargs.get("embedding_version")

    if args:
        vals = [p1, p2, p3, p4, p5]
        for i, a in enumerate(args):
            idx = 5 - len(args) + i
            if 0 <= idx < 5 and vals[idx] is None:
                vals[idx] = a
        p1, p2, p3, p4, p5 = vals

    if any(v is None or str(v).strip() == "" for v in (p1, p2, p3, p4, p5)):
        raise ValueError("cache key 缺必填版本字段")

    raw = "|".join([str(p1).strip(), str(p2).strip(), str(p3).strip(), str(p4).strip(), str(p5).strip()])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
