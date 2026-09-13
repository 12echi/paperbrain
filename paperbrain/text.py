"""文本句子切分 (v5.0): 保护小数/缩写后再按句切, 防 3.2 -> 3. 2 碎裂."""
import re
from typing import List

_PROT = [("DOT", re.compile(r"(?<=\d)\.(?=\d)")),
         ("ABBR", re.compile(r"\b(e\.g|i\.e|Fig|Tab|Sec|Eq|Eqs|vs|al|No|Figs|Tabs)\.", re.I))]


def protect(s: str) -> str:
    """用等长单字符 Unicode 私有区占位符保护小数与缩写中的点, 严格保证 len(protect(s)) == len(s).
    \\ue000 保护小数中的点 (e.g. 3.2 -> 3\\ue0002)
    \\ue001 保护缩写中的点 (e.g. e.g. -> e\\ue001g\\ue001)
    """
    if not s:
        return ""
    s = _PROT[0][1].sub("\ue000", s)
    s = _PROT[1][1].sub(lambda m: m.group(0).replace(".", "\ue001"), s)
    return s


def restore(s: str) -> str:
    """将占位符还原为原始标点 (同时兼容旧版多字符标记)."""
    if not s:
        return ""
    return (s.replace("\ue000", ".")
             .replace("\ue001", ".")
             .replace("<DOT>", ".")
             .replace("<P>", "."))


_SPLIT = re.compile(r"(?<=[。！？.!?])\s*")

_BOUND = re.compile(r"。|！|？|!|\?|\.(?=\s|$)")


def split_spans(text: str):
    """按句切并返回 [(句子, 起, 止)] (起止为原文偏移, 精确无漂移).
    先保护小数/缩写, 边界只认中日韩句号/!?/后跟空白或结尾的点."""
    if not text:
        return []
    prot = protect(text)
    spans = []
    last = 0
    for m in _BOUND.finditer(prot):
        spans.append((restore(prot[last:m.end()]).strip(), last, m.end()))
        last = m.end()
    tail = restore(prot[last:]).strip()
    if tail:
        spans.append((tail, last, len(prot)))
    return [(s, a, b) for s, a, b in spans if s]


def split_sentences(text: str) -> List[str]:
    return [s for s, _, _ in split_spans(text)]


_LIG = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
        "\ufb04": "ffl", "\ufb05": "st", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
        "\u200b": "", "\ufeff": "", "\ufffd": ""}


def sanitize_text(text: str) -> str:
    """去控制符/连字/零宽, 归一空白, 防下游乱码。"""
    if not text:
        return ""
    for k, v in _LIG.items():
        if k in text:
            text = text.replace(k, v)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # 折叠 3+ 连续空行为 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{4,}", "  ", text)
    return text
