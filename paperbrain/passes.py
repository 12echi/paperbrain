"""Pass1-4 透读 (v5.0 最小可跑版, 离线规则摘要 + LLM可注入 + M2 图表双向索引).

默认不用云端LLM即可跑通 (规则式抽取前N句+关键词), 生产替换 summarize_fn 即可.
token 估算: 英文 ~4字符/token, 中文 ~1.5字符/token 的保守混合: tokens≈len/3.5.
F07: 构建图表编号、页面 BBox、图像路径与章节的双向索引 (Sec <-> Fig/Tab).
"""
import re
from typing import Any, Callable, Dict, List, Optional

from .budget import check_budget
from .text import split_sentences


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # 混合估算, 偏保守 (宁可高估不超支)
    return max(1, int(len(text) / 3.5))


SENT_SPLIT = re.compile(r"(?<=[。！？.!?])\s*")


def extractive(text: str, max_chars: int, must_have: Optional[List[str]] = None) -> str:
    sents = split_sentences(text)
    if must_have:
        scored = sorted(sents, key=lambda s: sum(k.lower() in s.lower() for k in must_have), reverse=True)
        # 保留原序取 top
        top = set(scored[:6])
        ordered = [s for s in sents if s in top]
        sents = ordered or sents
    out = ""
    for s in sents:
        if len(out) + len(s) + 1 > max_chars:
            break
        out += (s + " ")
    return out.strip() or text[:max_chars]


def _even_reduce(parts: List[str], max_chars: int) -> str:
    """归约均分: 每窗配额相同, 保序拼接, 防关键词平分时只取头部窗口."""
    parts = [p for p in parts if p and p.strip()]
    max_chars = max(0, int(max_chars))
    if not parts or max_chars == 0:
        return ""
    if len(parts) == 1:
        return parts[0][:max_chars]
    # 至少为每个保留窗口分配 1 字符和 1 个分隔符；窗口极多时等距采样，始终保留首尾。
    capacity = max(1, (max_chars + 1) // 2)
    if len(parts) > capacity:
        if capacity == 1:
            parts = [parts[0]]
        else:
            indices = [round(i * (len(parts) - 1) / (capacity - 1)) for i in range(capacity)]
            parts = [parts[i] for i in indices]
    per = max(1, (max_chars - (len(parts) - 1)) // len(parts))
    trimmed = [extractive(p, per) for p in parts]
    return " ".join(trimmed)[:max_chars]


def digest_section(text: str, max_chars: int,
                   summarize_fn: Optional[Callable[[str, str], str]] = None,
                   kind: str = "passx",
                   must_have: Optional[List[str]] = None) -> str:
    """全量 map-reduce: 长文本按窗口逐块摘要再汇总, 不静默丢弃.
    LLM 窗 6000 字 / 规则窗 3000 字. 输出恒 ≤max_chars."""
    text = text or ""
    if not text:
        return ""
    if summarize_fn:
        # 调用方 (_summ) 已做尺寸控制, 这里只调一次; 异常才回退分窗规则
        try:
            return summarize_fn(kind, text)
        except Exception:
            pass
    if len(text) <= max_chars * 2:
        return extractive(text, max_chars, must_have)
    parts = [extractive(text[i:i + 3000], 500, must_have)
             for i in range(0, len(text), 3000)]
    return _even_reduce(parts, max_chars)


def _build_citation_chain(sections: List[Dict], max_chars: int = 2400) -> str:
    """从真实 ChunkID 构建紧凑引用链，供 Pass4 只读纪要与证据锚点。"""
    rows: List[str] = []
    for section in sections:
        for chunk in section.get("chunks", []):
            cid = str(chunk.get("chunk_id", "")).strip()
            body = " ".join(str(chunk.get("text", "")).split())
            if cid and body:
                rows.append(f"[Source: {cid}] {body[:220]}")
    if not rows:
        return ""
    return _even_reduce(rows, max_chars)


KEYWORDS = {"pass1": ["propose", "contribution", "提出", "贡献", "结论"],
            "pass2": ["model", "loss", "attention", "模型", "损失", "公式"],
            "pass3": ["BLEU", "accuracy", "table", "figure", "消融", "Fig", "Tab"],
            "pass4": []}


# ==============================================================================
# F07: 双向图表索引与引用解析 (Bidirectional Figure/Table Index)
# ==============================================================================

def resolve_figure_citation(citation: str, fig_index: Dict[str, Any],
                           paper_id: str = "") -> Optional[Dict[str, Any]]:
    """将正文引文标签 (如 [Ref: Paper, Fig 1] 或 [Ref: 2024_NeurIPS_01, Tab 2]) 解析为具体图表对象."""
    if not citation or not fig_index:
        return None

    by_id = fig_index.get("_by_id", {})

    m_fig = re.search(r"(?i)\b(?:fig|figure|图)\.?\s*([0-9]+[a-zA-Z]?)", citation)
    m_tab = re.search(r"(?i)\b(?:tab|table|表)\.?\s*([0-9]+[a-zA-Z]?)", citation)

    target_id = None
    if m_fig:
        target_id = f"fig{m_fig.group(1).lower()}"
    elif m_tab:
        target_id = f"tab{m_tab.group(1).lower()}"
    elif citation.strip().lower() in by_id:
        target_id = citation.strip().lower()

    if target_id and target_id in by_id:
        return by_id[target_id]

    for k, v in fig_index.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        meta = v.get("meta", {})
        if target_id and target_id in meta:
            return meta[target_id]

    return None


def get_figure_host_section(fig_id: str, fig_index: Dict[str, Any]) -> Optional[str]:
    """根据图表 ID 反查其所属宿主章节 (如 2024_NeurIPS_01_3.2)."""
    if not fig_id or not fig_index:
        return None
    target = fig_id.strip().lower()
    by_id = fig_index.get("_by_id", {})
    if target in by_id:
        return by_id[target].get("host_section")

    for k, v in fig_index.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        figs = [str(x).lower() for x in v.get("figs", [])]
        tabs = [str(x).lower() for x in v.get("tabs", [])]
        if target in figs or target in tabs:
            return k
    return None


def _is_caption_occurrence(text: str, start: int, caption: str) -> bool:
    """True only when the matched reference is on the stored caption line itself."""
    if not caption:
        return False
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    line = " ".join(text[line_start:line_end].split()).strip()
    expected = " ".join(str(caption).split()).strip()
    return bool(line and expected and line.casefold() == expected.casefold())


def _item_contexts(text: str, item_id: str, caption: str = "",
                   radius: int = 220) -> List[str]:
    kind = "fig" if item_id.startswith("fig") else "tab"
    num = re.escape(item_id[len(kind):])
    prefix = (r"(?:\b(?:figure|fig\.?)|图)" if kind == "fig" else
              r"(?:\b(?:table|tab\.?)|表)")
    contexts: List[str] = []
    pattern = re.compile(prefix + r"\s*" + num + r"\b", re.I)
    offset = 0
    for line in (text or "").splitlines(keepends=True):
        clean = " ".join(line.split())
        for match in pattern.finditer(line):
            if _is_caption_occurrence(text or "", offset + match.start(), caption):
                continue
            if clean and clean not in contexts:
                contexts.append(clean)
        offset += len(line)
    return contexts


def build_bidirectional_fig_index(sections: List[Dict], paper_id: str,
                                  figures: Optional[List[Dict]] = None,
                                  tables: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """构建图表与章节的双向结构化索引 (F07)."""
    fig_index: Dict[str, Any] = {}
    by_id: Dict[str, Dict[str, Any]] = {}

    # 预载入已知 figures
    if figures:
        for f in figures:
            f_id = (f.get("id") or f"fig{f.get('num', '')}").lower()
            by_id[f_id] = {
                "id": f_id,
                "type": "figure",
                "num": str(f.get("num", "")),
                "page": f.get("page", 0),
                "bbox": f.get("bbox", []),
                "caption": f.get("caption", ""),
                "image_path": f.get("image", ""),
                "pixels": f.get("pixels", 0),
                "subfigures": f.get("subfigures", []),
                "manual": f.get("manual", False),
                "metadata_stripped": f.get("metadata_stripped", False),
                "bound_contexts": list(f.get("bound_contexts", []))
            }

    # 预载入已知 tables
    if tables:
        for t in tables:
            t_id = (t.get("table_id") or f"tab{t.get('num', '')}").lower()
            by_id[t_id] = {
                "id": t_id,
                "type": "table",
                "num": str(t.get("num", "")),
                "page": t.get("page", 0),
                "bbox": t.get("bbox", []),
                "caption": t.get("caption", ""),
                "markdown": t.get("markdown", ""),
                "cells": t.get("cells", []),
                "rows": t.get("rows", 0),
                "cols": t.get("cols", 0),
                "bound_contexts": list(t.get("bound_contexts", []))
            }

    for s in sections:
        sec_num = s.get("sec", "0")
        seckey = f"{paper_id}_{sec_num}" if paper_id else str(sec_num)
        text = s.get("text", "")

        figs = re.findall(r"(?i)(?:\b(?:figure|fig\.?)|图)\s*([0-9]+[a-zA-Z]?)", text)
        tabs = re.findall(r"(?i)(?:\b(?:table|tab\.?)|表)\s*([0-9]+[a-zA-Z]?)", text)
        fl = [f"fig{x}".lower() for x in figs]
        tl = [f"tab{x}".lower() for x in tabs]

        meta = {}
        for f_id in fl:
            if f_id not in by_id:
                by_id[f_id] = {
                    "id": f_id,
                    "type": "figure",
                    "num": f_id.replace("fig", ""),
                    "page": 0,
                    "bbox": [],
                    "caption": "",
                    "image_path": "",
                    "host_section": seckey,
                    "sec": sec_num
                }
            else:
                by_id[f_id]["host_section"] = seckey
                by_id[f_id]["sec"] = sec_num
            hosts = by_id[f_id].setdefault("host_sections", [])
            if seckey not in hosts:
                hosts.append(seckey)
            contexts = by_id[f_id].setdefault("bound_contexts", [])
            contexts.extend(x for x in _item_contexts(
                text, f_id, str(by_id[f_id].get("caption", ""))) if x not in contexts)
            meta[f_id] = by_id[f_id]

        for t_id in tl:
            if t_id not in by_id:
                by_id[t_id] = {
                    "id": t_id,
                    "type": "table",
                    "num": t_id.replace("tab", ""),
                    "page": 0,
                    "bbox": [],
                    "caption": "",
                    "markdown": "",
                    "host_section": seckey,
                    "sec": sec_num
                }
            else:
                by_id[t_id]["host_section"] = seckey
                by_id[t_id]["sec"] = sec_num
            hosts = by_id[t_id].setdefault("host_sections", [])
            if seckey not in hosts:
                hosts.append(seckey)
            contexts = by_id[t_id].setdefault("bound_contexts", [])
            contexts.extend(x for x in _item_contexts(
                text, t_id, str(by_id[t_id].get("caption", ""))) if x not in contexts)
            meta[t_id] = by_id[t_id]

        if fl or tl:
            fig_index[seckey] = {
                "figs": fl,
                "tabs": tl,
                "meta": meta
            }

    fig_index["_by_id"] = by_id
    return fig_index


def run_passes(sections: List[Dict], paper_id: str,
               summarize_fn: Optional[Callable[[str, str], str]] = None,
               figures: Optional[List[Dict]] = None,
               tables: Optional[List[Dict]] = None) -> Dict:
    """返回 {pass1..pass4, ground_truth, fig_index, pass_tokens, coverage}.
    全量 map-reduce: 每节全文参与 digest, coverage=输入覆盖率 (降级时 <100%).
    summarize_fn(prompt_kind, text)->summary, 不传则用规则式.
    F07: 构建图表与章节双向索引 fig_index.
    """
    # 同名节 (如 Experiments sec3 + Discussion sec4) 必须合并, 否则 dict 覆盖丢整节
    by: Dict[str, Dict] = {}
    for s in sections:
        n = s["name"]
        if n in by:
            by[n]["text"] += "\n" + s.get("text", "")
            by[n]["chunks"].extend(s.get("chunks", []))
        else:
            by[n] = {"text": s.get("text", ""), "chunks": list(s.get("chunks", []))}
    downgrade = (any(s.get("downgrade_pass1_only") or
                     float(s.get("confidence", 1.0)) < 0.5 for s in sections) or not sections)

    get = lambda n: by.get(n, {}).get("text", "")
    total_chars = sum(len(s.get("text", "")) for s in sections)

    # 名称 -> 出现的所有节号(有序去重): 供 outline 按名归位, 不再硬编码 0/1/2/3/4
    name_sec: Dict[str, List[str]] = {}
    for s in sections:
        lst = name_sec.setdefault(s["name"], [])
        if s.get("sec") not in lst:
            lst.append(s.get("sec"))

    t_abstract = by.get("abstract", {}).get("text", "")
    t_intro = by.get("intro", {}).get("text", "") or by.get("full", {}).get("text", "")
    t_concl = by.get("conclusion", {}).get("text", "")
    p1_src = "\n".join([t_abstract, t_intro, t_concl])
    p1 = digest_section(p1_src, 2500 * 3, summarize_fn, "pass1", KEYWORDS.get("pass1"))
    citation_chain = _build_citation_chain(sections)
    if downgrade:
        # 低置信文档严格只跑 Pass1；Pass4 需要完整纪要+引用链，不能在降级态冒充执行。
        p2 = p3 = p4 = ""
        fed = len(p1_src)  # 只读了 abstract/intro/conclusion
    else:
        p2 = digest_section(get("method"), 3500 * 3, summarize_fn, "pass2", KEYWORDS.get("pass2"))
        p3 = digest_section(get("experiments"), 4000 * 3, summarize_fn, "pass3", KEYWORDS.get("pass3"))
        if not citation_chain:
            raise RuntimeError("Pass4 拒绝执行：缺真实 ChunkID 引用链")
        p4_src = "\n".join([p1, p2, p3, "=== 引用链 ===", citation_chain])
        p4 = digest_section(p4_src, 2000 * 3, summarize_fn, "pass4", KEYWORDS.get("pass4"))
        fed = len(p1_src) + len(get("method")) + len(get("experiments"))
    # coverage: 真正被喂入各 Pass 的原文比例 (不含 Pass 输出回灌, 降级时坦白 <100%)
    coverage = round(min(1.0, fed / total_chars), 4) if total_chars else 1.0

    pass_tokens = {k: estimate_tokens(v) for k, v in
                   {"pass1": p1, "pass2": p2, "pass3_text": p3, "pass4": p4}.items() if v}
    # 预算门禁: 超即抛, 不伪造纪要
    br = check_budget(pass_tokens, num_images=0)
    if not br.ok:
        raise RuntimeError(f"预算超限拒绝执行: {br.reasons}")

    # ground_truth: ChunkID级, key 同时兼容 {Paper}_{Sec} 粗键 (verifier 两级)
    gt: Dict[str, str] = {}
    for s in sections:
        for c in s.get("chunks", []):
            gt[c["chunk_id"]] = c["text"]
        seckey = f"{paper_id}_{s.get('sec','0')}"
        if seckey not in gt:
            gt[seckey] = s.get("text", "")[:2000]

    # 构建结构化双向图表索引
    fig_index = build_bidirectional_fig_index(sections, paper_id, figures=figures, tables=tables)

    return {"pass1": p1, "pass2": p2, "pass3": p3, "pass4": p4,
            "ground_truth": gt, "fig_index": fig_index, "name_sec": name_sec,
            "pass_tokens": pass_tokens, "downgrade_pass1_only": downgrade,
            "citation_chain": citation_chain,
            "coverage": coverage, "total_chars": total_chars}
