"""预处理分流 (v5.0 最小可跑版, stdlib only).

生产三轨的离线子集:
- .txt/.md -> 标准通道 (text_rate=1.0)
- .pdf 有 PyMuPDF/fitz 才走真实检测, 缺则降级为 NEEDS_OCR + 只跑 Pass1
- 公式/复合图在此层只打标, 不强切 (宁可不切)
"""
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

FORMULA_PAT = re.compile(r"\$\$.+?\$\$|\$.+?\$|\\begin\{equation\}.*?\\end\{equation\}", re.S)
SUBFIG_PAT = re.compile(r"\(a\)|\(b\)|\(c\)|（a）|（b）", re.I)
TABLE_CAPTION_PAT = re.compile(r"(?im)^\s*(?:table|tab\.?|表)\s*[0-9]+[a-zA-Z]?\b")


@dataclass
class PreflightResult:
    paper_id: str
    sha: str
    route: str  # standard | hybrid | deep_ocr | needs_ocr_tool
    text_rate: float
    text: str
    warnings: List[str]
    page_rates: List[float] = field(default_factory=list)
    low_conf_pages: List[int] = field(default_factory=list)
    table_line_missing_pages: List[int] = field(default_factory=list)


def _table_lines_missing(page, page_text: str) -> bool:
    """有明确表题但 lines 策略找不到表格时，标记为需 Hybrid 复核。

    没有 find_tables 能力时返回 False：能力缺失不能当成“已证实缺线”。
    """
    if not TABLE_CAPTION_PAT.search(page_text or "") or not hasattr(page, "find_tables"):
        return False
    try:
        found = page.find_tables(strategy="lines")
        return not bool(getattr(found, "tables", None))
    except Exception:
        return False


def _sha(s: bytes) -> str:
    return hashlib.sha256(s).hexdigest()


def preflight(input_path: str, paper_id: str) -> PreflightResult:
    p = Path(input_path)
    raw = p.read_bytes()
    sha = _sha(raw)
    warnings: List[str] = []
    suffix = p.suffix.lower()
    from .text import sanitize_text
    if suffix in (".txt", ".md"):
        text = sanitize_text(raw.decode("utf-8", errors="replace"))
        return PreflightResult(paper_id, sha, "standard", 1.0, text, warnings)
    if suffix == ".pdf":
        try:
            import fitz  # type: ignore
        except Exception:
            warnings.append("缺PyMuPDF, PDF降级为needs_ocr_tool, 只跑Pass1")
            return PreflightResult(paper_id, sha, "needs_ocr_tool", 0.0, "", warnings)
        page_texts: List[str] = []
        page_rates: List[float] = []
        table_line_missing_pages: List[int] = []
        with fitz.open(str(p)) as doc:
            for pno, page in enumerate(doc):
                page_text = page.get_text(sort=True) or ""
                page_texts.append(page_text)
                try:
                    blocks = [b for b in page.get_text("blocks")
                              if len(b) > 6 and b[6] == 0 and str(b[4]).strip()]
                    layout = page.get_text("dict").get("blocks", [])
                    image_blocks = sum(1 for block in layout if block.get("type") == 1)
                except Exception:
                    blocks, image_blocks = [], 0
                chars = len(re.sub(r"\s+", "", page_text))
                # 可提取率不是“每块必须有40字”：该旧口径会惩罚短表格单元格。
                # 用最小有效字符置信 × 文本块在文本/图像内容块中的覆盖率。
                char_confidence = min(1.0, chars / 40.0)
                content_blocks = len(blocks) + image_blocks
                block_coverage = len(blocks) / content_blocks if content_blocks else 0.0
                page_rates.append(char_confidence * block_coverage)
                if _table_lines_missing(page, page_text):
                    table_line_missing_pages.append(pno)
        rate = sum(page_rates) / len(page_rates) if page_rates else 0.0
        low_conf_pages = sorted(set(
            [i for i, page_rate in enumerate(page_rates) if page_rate < 0.70] +
            table_line_missing_pages))
        # rate > 0.90 -> standard; 0.50 <= rate <= 0.90 -> hybrid; rate < 0.50 -> deep_ocr
        route = "standard" if rate > 0.9 else ("hybrid" if rate >= 0.5 else "deep_ocr")
        if table_line_missing_pages and route == "standard":
            route = "hybrid"
        text = sanitize_text("\n".join(page_texts))
        if FORMULA_PAT.search(text):
            warnings.append("检测到公式, 下游须过katex/sympy门禁")
        if SUBFIG_PAT.search(text):
            warnings.append("检测到(a)(b)疑似复合图标记, 下游按IoU>0.5才拆")
        if low_conf_pages:
            warnings.append(f"低置信页 {len(low_conf_pages)}/{len(page_rates)}，文本率<70%或表格线缺失")
        if table_line_missing_pages:
            warnings.append(
                f"表题存在但线框表未检出：页 {','.join(str(i + 1) for i in table_line_missing_pages)}")
        return PreflightResult(paper_id, sha, route, rate, text, warnings,
                               page_rates, low_conf_pages, table_line_missing_pages)
    raise ValueError(f"不支持的输入类型: {suffix} (最小闭环仅支持 .txt/.md/.pdf)")
