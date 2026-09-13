"""章节切分与递归语义分块 (v5.0 / M4 重构版).

特性:
- 废除 1500 字符硬截断，实现基于篇章层级的递归语义分块 (F13)
- 分块层级: 章节标题 (H1/H2) -> 自然段落 (\\n\\n) -> 学术标点 (。！？.!? 及分号从句)
- 目标尺寸: 1000-1200 字符，100-150 字符语义滑动重叠 (Overlap)
- 数学公式 ($...$, $$...$$)、Markdown 表格、完整句子跨块保护
- 保持 ChunkID 规范: {paper_id}_Sec{sec}_C{idx:03d}
- 提供统一顶层分块接口 chunk_sections 与导出规范
"""
import re
from typing import Any, Dict, List, Optional, Tuple


HEADERS = [
    ("abstract", r"^\s*(abstract|摘要)\s*$", "0"),
    ("intro", r"^\s*(\d+\s*[\.\)]?\s*)?(introduction|background|引言|绪论|前言)\s*$", "1"),
    # 方法/实验: 有编号标题 (2. Methodology) 或 裸标题行 (Methods/Results/Discussion).
    # 裸标题必须整行精确匹配, 防 "Model Formulation: ..." 误切.
    # 注意: 必须含 "Methods"(复数) 与 "Discussion"(编号), 否则超常见标题整节丢失.
    ("method", r"^\s*\d+\s*[\.\)]\s*(methods?|methodology|models?|方法|模型)\b.*$", "2"),
    ("method_bare", r"^\s*(methods?|methodology|materials?\s+and\s+methods?)\s*$", "2"),
    ("experiments", r"^\s*\d+\s*[\.\)]\s*(experiments?|evaluations?|results?|discussions?|实验|结果|评估|讨论)\b.*$", "3"),
    ("exp_bare", r"^\s*(results?|discussions?|结果|讨论)\s*$", "3"),
    ("conclusion", r"^\s*(\d+\s*[\.\)]?\s*)?(conclusions?|总结|结论)\s*$", "4"),
]

_CANON = {"method_bare": "method", "exp_bare": "experiments"}

# 文末元数据节: 命中即截断丢弃 (防 References 污染结论节), 记录 dropped_tail
TAIL_PAT = re.compile(
    r"^\s*(references|参考文献|acknowledgements?|致谢|data availability|"
    r"author contributions?|funding|declarations?|competing interests|"
    r"additional information|作者贡献|利益冲突|补充信息)\s*$",
    re.I,
)

SEC_NUM = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")

# 保护块模式: Markdown 表格, 显示公式, 行内公式, 代码块
_RE_MD_TABLE = re.compile(
    r"(?:(?<=\n)|^)[ \t]*\|[^\n]+\|[ \t]*\r?\n[ \t]*\|[-:\s|]+\|[ \t]*\r?\n(?:[ \t]*\|[^\n]+\|[ \t]*\r?\n?)*",
    re.MULTILINE,
)
_RE_DISPLAY_MATH = re.compile(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]")
_RE_INLINE_MATH = re.compile(r"(?<!\\)\$(?:\\\$|[^\$\n])+\$|\\\([\s\S]*?\\\)")
_RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```")

# 学术标点分句正则 (保护小数与学术缩写)
# 排除如 3.14, e.g., i.e., Fig., Tab., Sec., Eq., al., vs.
_RE_SENT_SPLIT = re.compile(
    r"(?<=[。！？\?!])\s*|"
    r"(?<=\.)(?!\d)\s+"
)
_ABBREV_SUFFIX_RE = re.compile(
    r"\b(?:e\.g|i\.e|Fig|Tab|Sec|Eq|Eqs|vs|al|No|Figs|Tabs|ref|Ref|Dr|Prof|pp|vol|dept|univ)\.\s*$",
    re.IGNORECASE,
)

# 二级分句/分号边界
_RE_CLAUSE_SPLIT = re.compile(r"(?<=[；;])\s*")


def get_protected_spans(text: str) -> List[Tuple[int, int]]:
    """获取所有受保护区块 (Markdown 表格、数学公式、代码块) 的起止范围，防止分块时横切截断."""
    spans: List[Tuple[int, int]] = []
    for pattern in (_RE_MD_TABLE, _RE_DISPLAY_MATH, _RE_INLINE_MATH, _RE_CODE_BLOCK):
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
    if not spans:
        return []
    spans.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int]] = [spans[0]]
    for cur in spans[1:]:
        prev_start, prev_end = merged[-1]
        if cur[0] < prev_end:
            merged[-1] = (prev_start, max(prev_end, cur[1]))
        else:
            merged.append(cur)
    return merged


def _is_inside_protected(idx: int, protected_spans: List[Tuple[int, int]]) -> bool:
    """判定给定字符索引是否处于保护块内部."""
    for s, e in protected_spans:
        if s < idx < e:
            return True
        if idx < s:
            break
    return False


def split_semantic_units(text: str) -> List[str]:
    """层次化切分文本为最小语义单元 (完整句子、表格、独立公式块).

    切分层级:
    1. 小节标题 / Markdown 标题 (###, 1.1)
    2. 自然段落 (\\n\\n)
    3. 学术标点 (。！？.!?，保护小数和学术缩写)
    4. 从句边界 (；;)
    """
    if not text:
        return []

    def _split_with_regex(block: str, pattern: re.Pattern) -> List[str]:
        spans = get_protected_spans(block)
        cuts = [0]
        for m in pattern.finditer(block):
            if pattern is _RE_SENT_SPLIT and _ABBREV_SUFFIX_RE.search(block[:m.end()]):
                continue
            pos = m.end()
            if not _is_inside_protected(pos, spans):
                cuts.append(pos)
        cuts.append(len(block))
        cuts = sorted(list(set(cuts)))
        parts = []
        for i in range(len(cuts) - 1):
            seg = block[cuts[i]:cuts[i + 1]]
            if seg.strip():
                parts.append(seg.strip())
        return parts

    # 第 1 级: 章节标题 / Markdown 子标题 (###, 1.1 等)
    heading_pat = re.compile(r"\n\n+(?=#{1,4}\s|\d+\.\d+\s+[A-Za-z\u4e00-\u9fff])")
    stage1 = _split_with_regex(text, heading_pat) or [text.strip()]

    # 第 2 级: 自然段落 (\n\n)
    stage2 = []
    para_pat = re.compile(r"\n\n+")
    for b1 in stage1:
        if len(b1) > 1200:
            stage2.extend(_split_with_regex(b1, para_pat))
        else:
            stage2.append(b1)

    # 第 3 级: 学术句子标点
    stage3 = []
    for b2 in stage2:
        if len(b2) > 1200:
            stage3.extend(_split_with_regex(b2, _RE_SENT_SPLIT))
        else:
            stage3.append(b2)

    # 第 4 级: 分号/从句 (对于依然过长的单句)
    units = []
    for b3 in stage3:
        if len(b3) > 1200:
            units.extend(_split_with_regex(b3, _RE_CLAUSE_SPLIT))
        else:
            units.append(b3)

    return units if units else [text.strip()]


def recursive_semantic_chunk(
    text: str,
    paper_id: str,
    sec: str,
    target_size: int = 1100,
    overlap: int = 120,
    max_size: int = 1200,
) -> List[Dict[str, Any]]:
    """递归语义分块引擎 (F13).

    - 目标尺寸: 1000-1200 字符 (默认 target=1100, max=1200)
    - 滑动重叠: 100-150 字符 (默认 overlap=120)
    - 完整性约束: 公式、表格、句子绝不断裂
    - 命名契约: {paper_id}_Sec{sec}_C{idx:03d}
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return []

    # 若文本总长小于软上限，且不需拆分，直接作为单块输出
    if len(clean_text) <= max_size:
        return [{"chunk_id": f"{paper_id}_Sec{sec}_C001", "text": clean_text}]

    units = split_semantic_units(clean_text)
    if not units:
        return [{"chunk_id": f"{paper_id}_Sec{sec}_C001", "text": clean_text}]

    chunks: List[str] = []
    curr_units: List[str] = []
    curr_len = 0

    idx = 0
    while idx < len(units):
        u = units[idx]
        u_len = len(u)

        # 若单单元过长 (如大表格或长推导)，必须保留其完整性
        if u_len > max_size and not curr_units:
            chunks.append(u)
            idx += 1
            continue

        sep_len = 2 if curr_units else 0
        if curr_units and (curr_len + sep_len + u_len > max_size or curr_len >= target_size):
            # 封装当前 Chunk
            chunk_str = "\n\n".join(curr_units).strip()
            if chunk_str:
                chunks.append(chunk_str)

            # 计算 100-150 字符的语义滑动重叠 (挑选尾部完整单元)
            overlap_units: List[str] = []
            ov_len = 0
            for prev_u in reversed(curr_units):
                overlap_units.insert(0, prev_u)
                ov_len += len(prev_u)
                if ov_len >= overlap:
                    break

            # 防止无限回退死循环: 重叠单元数不能等于全量已选单元数
            if len(overlap_units) >= len(curr_units):
                overlap_units = overlap_units[1:]

            curr_units = list(overlap_units)
            curr_len = sum(len(x) for x in curr_units) + (2 * max(0, len(curr_units) - 1))

        curr_units.append(u)
        curr_len += (2 if len(curr_units) > 1 else 0) + u_len
        idx += 1

    if curr_units:
        final_chunk = "\n\n".join(curr_units).strip()
        # 避免末尾仅有重叠内容而无新增实义文本的重复块
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    return [
        {"chunk_id": f"{paper_id}_Sec{sec}_C{i + 1:03d}", "text": c}
        for i, c in enumerate(chunks)
    ]


def split_sections(
    text: str,
    paper_id: str,
    meta: Optional[Dict] = None,
    **kwargs,
) -> List[Dict]:
    """章节切分与递归语义分块."""
    target_size = kwargs.get("target_size", 1100)
    overlap = kwargs.get("overlap", 120)
    max_size = kwargs.get("max_size", 1200)

    lines = text.splitlines()
    hits: List[tuple] = []  # (line_no, name, sec)
    for i, ln in enumerate(lines):
        for name, pat, default_sec in HEADERS:
            if re.match(pat, ln.strip(), re.I):
                m = SEC_NUM.match(ln.strip())
                sec = m.group(1) if m else default_sec
                # method/experiments 尝试从标题捞数字节
                hits.append((i, name, sec))
                break
    if not hits:
        # 切分失败: 整篇降级为 Pass1-only, confidence=0
        if meta is not None:
            meta.update({"dropped_tail": False, "tail_at": None, "n_sections": 1})
        chunks = recursive_semantic_chunk(
            text, paper_id, "0", target_size=target_size, overlap=overlap, max_size=max_size
        )
        return [{
            "name": "full",
            "sec": "0",
            "text": text,
            "confidence": 0.0,
            "downgrade_pass1_only": True,
            "chunks": chunks or [{"chunk_id": f"{paper_id}_Sec0_C001", "text": text[:4000]}],
        }]
    hits.sort()
    # 文末截断点
    tail_at = None
    for i, ln in enumerate(lines):
        if TAIL_PAT.match(ln.strip()):
            tail_at = i
            break
    limit = tail_at if tail_at is not None else len(lines)
    dropped_tail = tail_at is not None
    secs: List[Dict] = []
    # 卷首语捕获: 首个标题前的长文本判为 intro (Sci Rep 等无引言标题格式), 防丢数据
    if hits and hits[0][0] > 0:
        pre = "\n".join(lines[:hits[0][0]]).strip()
        _META = re.compile(
            r"@|https?://|doi\.org|10\.\d{4}/|University|Hospital|Department|"
            r"Laboratory|Academy|Institute|www\.nature\.com|Scientific Reports|OPEN$",
            re.I,
        )
        _AUTH = re.compile(r"(\d.*?,){2,}|[A-Za-z\u4e00-\u9fff]+\s*&\s*[A-Z]")

        def _is_author(ln: str) -> bool:
            if _AUTH.search(ln):
                return True
            return len(re.findall(r"[A-Z][a-z]{2,}\d", ln)) >= 3

        body_lines = [
            ln
            for ln in pre.splitlines()
            if (len(ln.strip()) > 60 or re.search(r"[。．.!?]$", ln.strip()))
            and not _META.search(ln)
            and not _is_author(ln)
        ]
        pre_body = "\n".join(body_lines).strip()
        if len(pre_body) > 400:
            chunks = recursive_semantic_chunk(
                pre_body, paper_id, "1", target_size=target_size, overlap=overlap, max_size=max_size
            )
            secs.append({
                "name": "intro",
                "sec": "1",
                "text": pre_body,
                "confidence": 0.55,
                "downgrade_pass1_only": False,
                "preamble": True,
                "chunks": chunks,
            })
    for idx, (ln, name, sec) in enumerate(hits):
        name = _CANON.get(name, name)
        end = hits[idx + 1][0] if idx + 1 < len(hits) else limit
        body = "\n".join(lines[ln:end]).strip()
        conf = 0.9 if re.match(r"^\s*\d+", lines[ln].strip()) else 0.65
        if len(body) < 200:
            conf = min(conf, 0.5)
        # 递归语义分块 (1000-1200 字符，100-150 重叠，公式/表格/标点跨块保护)
        chunks = recursive_semantic_chunk(
            body, paper_id, sec, target_size=target_size, overlap=overlap, max_size=max_size
        )
        # 同名节合并 (防 Methods 标题与正文行重复命中)
        if secs and secs[-1]["name"] == name and secs[-1]["sec"] == sec:
            secs[-1]["text"] += "\n" + body
            secs[-1]["chunks"] = recursive_semantic_chunk(
                secs[-1]["text"], paper_id, sec, target_size=target_size, overlap=overlap, max_size=max_size
            )
            continue
        secs.append({
            "name": name,
            "sec": sec,
            "text": body,
            "confidence": conf,
            "downgrade_pass1_only": False,
            "chunks": chunks,
        })
    # 任一关键章节处于低置信状态时，整篇降级只跑 Pass1；禁止用不可靠分章继续方法/实验推断。
    if any(float(section.get("confidence", 0.0)) < 0.5 for section in secs):
        for section in secs:
            section["downgrade_pass1_only"] = True
    if meta is not None:
        meta.update({"dropped_tail": dropped_tail, "tail_at": tail_at, "n_sections": len(secs)})
    return secs


def chunk_sections(text: str, paper_id: str = "PAPER", **kwargs) -> List[Dict]:
    """Top-level semantic chunking entrypoint aggregating section chunks."""
    secs = split_sections(text, paper_id, **kwargs)
    chunks = []
    for s in secs:
        chunks.extend(s.get("chunks", []))
    return chunks


class ChunkingEngine:
    """工业级语义分块引擎 (面向 SPECIFICATION_REPORT 契约)."""

    def __init__(self, target_size: int = 1100, overlap: int = 120, max_size: int = 1200):
        self.target_size = target_size
        self.overlap = overlap
        self.max_size = max_size

    def __call__(self, text: str, paper_id: str = "PAPER", **kwargs) -> List[Dict[str, Any]]:
        opts = {"target_size": self.target_size, "overlap": self.overlap, "max_size": self.max_size}
        opts.update(kwargs)
        return chunk_sections(text=text, paper_id=paper_id, **opts)

    @staticmethod
    def recursive_semantic_chunk(
        text: str,
        paper_id: str,
        sec: str,
        target_size: int = 1100,
        overlap: int = 120,
        max_size: int = 1200,
    ) -> List[Dict[str, Any]]:
        return recursive_semantic_chunk(
            text=text,
            paper_id=paper_id,
            sec=sec,
            target_size=target_size,
            overlap=overlap,
            max_size=max_size,
        )

    @staticmethod
    def chunk_sections(text: str, paper_id: str = "PAPER", **kwargs) -> List[Dict[str, Any]]:
        return chunk_sections(text=text, paper_id=paper_id, **kwargs)

    @staticmethod
    def split_sections(text: str, paper_id: str, meta: Optional[Dict] = None, **kwargs) -> List[Dict[str, Any]]:
        return split_sections(text=text, paper_id=paper_id, meta=meta, **kwargs)


RecursiveSemanticChunker = ChunkingEngine

__all__ = [
    "chunk_sections",
    "split_sections",
    "recursive_semantic_chunk",
    "RecursiveSemanticChunker",
    "ChunkingEngine",
    "get_protected_spans",
    "split_semantic_units",
]
