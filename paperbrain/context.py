"""全局上下文构建 (v5.0): 所有模型调用的统一"喂料"入口。

问题: 之前各处各自 `text[:8000]` 硬截断, 长论文尾部(方法/实验)被砍, 且不随关注点变化。
方案: 按小节配额切分预算 + 关注点相关句优先 + 保留可引用的节标记, 供下游精确引用。
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from . import config
from .retrieval import tokenize

# 各小节预算占比 (缺失的小节其份额回流到其它节)
QUOTA = {"abstract": 0.14, "intro": 0.18, "method": 0.30,
         "experiments": 0.26, "conclusion": 0.12, "full": 1.0}
ORDER = ["abstract", "intro", "method", "experiments", "conclusion", "full"]


def _load(out: Path, name: str, dflt):
    try:
        return json.loads((out / name).read_text(encoding="utf-8"))
    except Exception:
        return dflt


def _sentences(text: str) -> List[str]:
    from .text import split_sentences
    return [s for s in split_sentences(text or "") if s.strip()]


def _rank(sents: List[str], focus: str) -> List[str]:
    """关注点相关句优先, 其余按原序补足 (稳定、离线、无依赖)。"""
    if not focus or not sents:
        return sents
    ft = set(tokenize(focus))
    if not ft:
        return sents
    scored = []
    for i, s in enumerate(sents):
        st = set(tokenize(s))
        ov = len(ft & st) / (len(ft) + 1)
        scored.append((ov, i))
    top = [sents[i] for ov, i in sorted(scored, key=lambda x: (-x[0], x[1])) if ov > 0]
    picked = set(top)
    rest = [s for i, s in enumerate(sents) if s not in picked]
    return top + rest


def _take(text: str, budget: int, focus: str) -> str:
    sents = list(dict.fromkeys(_rank(_sentences(text), focus)))
    out = ""
    for s in sents:
        if len(out) + len(s) + 1 > budget:
            break
        out += (s + " ")
    return out.strip() or (text or "")[:budget]


def build_paper_context(out_dir: str, budget: Optional[int] = None,
                        focus: str = "") -> str:
    """拼出带节标记、按预算分配的论文上下文。"""
    budget = budget or config.context_chars()
    out = Path(out_dir)
    secs = _load(out, "sections.json", [])
    by: Dict[str, List[Dict]] = {}
    for s in secs:
        by.setdefault(s.get("name", "full"), []).append(s)

    # 有效配额: 仅在存在的小节间按比例重分配
    present = [n for n in ORDER if n in by]
    if not present:
        return ""
    total_q = sum(QUOTA.get(n, 0.2) for n in present) or 1.0
    parts: List[str] = []
    for name in present:
        share = int(budget * QUOTA.get(name, 0.2) / total_q)
        chunks = []
        for s in by[name]:
            sec = s.get("sec", "")
            all_chunk_text = "\n".join(str(c.get("text", "")) for c in s.get("chunks", [])
                                       if isinstance(c, dict) and c.get("text"))
            source_text = "\n".join(x for x in (all_chunk_text, str(s.get("text", ""))) if x)
            body = _take(source_text, share, focus)
            if body:
                chunks.append(f"### Sec {sec} {name}\n{body}")
        if chunks:
            parts.append("\n\n".join(chunks))
    ctx = "\n\n".join(parts)

    ents = _entities(out)
    if ents:
        ctx += "\n\n### 已知关键实体\n" + "、".join(ents[:20])
    return ctx[:budget + 800]  # 允许少量头部标记溢出


def _entities(out: Path) -> List[str]:
    mem = _load(out, "memory.json", {})
    return [e.get("name") for e in (mem.get("entities") or []) if isinstance(e, dict)]
