"""OCR 管道钩子 (v5.0 生产版).

- 本机有 tesseract 即真 OCR (按页) ; 缺失则返回 needs_tool, 上游降级只跑 Pass1.
- scanned 率 <50% 的 PDF 在此管道全量 OCR; 50-90% 只补低置信页 (由 preflight text_rate 决定).
"""
import shutil
import subprocess
import csv
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional


def has_ocr() -> bool:
    return shutil.which("tesseract") is not None


@lru_cache(maxsize=1)
def _available_languages() -> List[str]:
    try:
        r = subprocess.run(["tesseract", "--list-langs"], capture_output=True,
                           text=True, timeout=10)
        return [line.strip() for line in r.stdout.splitlines()[1:] if line.strip()]
    except Exception:
        return []


def _select_language(requested: str = "") -> tuple[str, List[str]]:
    available = set(_available_languages())
    wanted = [x.strip() for x in (requested or os.environ.get("PAPERBRAIN_OCR_LANG", "")).split("+")
              if x.strip()]
    if not wanted:
        wanted = (["chi_sim", "eng"] if "chi_sim" in available else ["eng"])
    selected = [lang for lang in wanted if lang in available]
    missing = [lang for lang in wanted if lang not in available]
    return "+".join(selected), missing


def _parse_tsv(payload: str) -> tuple[str, float]:
    words = []
    confidences = []
    last_line = None
    for row in csv.DictReader(io.StringIO(payload), delimiter="\t"):
        word = str(row.get("text", "")).strip()
        if not word:
            continue
        try:
            conf = float(row.get("conf", "-1"))
        except ValueError:
            conf = -1
        line = (row.get("block_num"), row.get("par_num"), row.get("line_num"))
        if words and line != last_line:
            words.append("\n")
        words.append(word)
        last_line = line
        if conf >= 0:
            confidences.append(conf)
    text = " ".join(words).replace(" \n ", "\n").strip()
    confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
    return text, confidence


def ocr_pdf(pdf_path: str, dpi: int = 200, workdir: str = "",
            pages: Optional[List[int]] = None, lang: str = "") -> Dict:
    """返回 {pages:[{text, conf}], ok}. 无 tesseract 即 ok=False.
    注: brew 版 tesseract 读 /tmp 受限, 中转图默认写 workdir (默认 pdf 同目录).
    """
    if not has_ocr():
        return {"pages": [], "ok": False, "reason": "needs_tool:缺tesseract (brew install tesseract)"}
    try:
        import fitz
    except Exception:
        return {"pages": [], "ok": False, "reason": "needs_tool:缺PyMuPDF"}
    doc = fitz.open(pdf_path)
    wd = Path(workdir) if workdir else Path(pdf_path).parent
    wd.mkdir(parents=True, exist_ok=True)
    selected_pages = (sorted(set(int(i) for i in pages if int(i) >= 0))
                      if pages is not None else list(range(len(doc))))
    page_results: List[Dict] = []
    language, missing_languages = _select_language(lang)
    if not language:
        doc.close()
        return {"pages": [], "ok": False, "reason": "needs_tool:缺请求的tesseract语言包",
                "missing_languages": missing_languages}
    try:
        for i, page in enumerate(doc):
            if i not in selected_pages:
                continue
            pix = page.get_pixmap(dpi=dpi)
            img = wd / f".pb_ocr_{Path(pdf_path).stem}_{i}.png"
            pix.save(str(img))
            try:
                r = subprocess.run(["tesseract", str(img), "stdout", "-l", language, "tsv"],
                                   capture_output=True, timeout=120)
                payload = r.stdout.decode("utf-8", errors="replace")
                txt, conf = _parse_tsv(payload) if r.returncode == 0 else ("", 0.0)
                page_results.append({"page": i, "text": txt, "conf": round(conf, 4)})
            finally:
                img.unlink(missing_ok=True)
    finally:
        doc.close()
    good = sum(p["conf"] >= 0.5 for p in page_results)
    coverage = good / len(selected_pages) if selected_pages else 0.0
    ok = bool(selected_pages) and len(page_results) == len(selected_pages) and coverage >= 0.90
    failed_pages = [p["page"] for p in page_results if p["conf"] < 0.5]
    return {"pages": page_results, "ok": ok, "reason": "" if ok else "ocr页级置信覆盖不足",
            "language": language, "missing_languages": missing_languages,
            "quality_coverage": round(coverage, 4), "failed_pages": failed_pages}
