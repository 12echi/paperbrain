"""配图 VL 解读与科研图表/表格结构化抽取 (v5.0 M2):
- PDF 内嵌位图 (XObject) 与原生矢量绘图 (Matplotlib / TikZ / Origin / ggplot) 统一抽取与局部高精光栅化渲染 (F05).
- 复合子图 (a)(b)(c) 识别与 IoU > 0.5 门禁拆分控制 (F08).
- PDF 结构化表格抽取与 Markdown 表格生成 (F06).
- 图表双向关联索引构建与引用解析 (F07).
- vision 模型多模态中文解读 (保留 opencode CLI 降级与 scratch 会话清理).
"""
import hashlib
import os
import re
import struct
import subprocess
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from . import config

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


# ==============================================================================
# 1. 矢量图形聚类与图注匹配辅助函数 (F05 / F08)
# ==============================================================================

FIG_CAPTION_RE = re.compile(
    r"(?i)^\s*(?:figure|fig\.?|图)\s*([0-9]+[a-zA-Z]?)[:.\s—–-]?\s*(.*)",
    re.DOTALL
)

TAB_CAPTION_RE = re.compile(
    r"(?i)^\s*(?:table|tab\.?|表)\s*([0-9]+[a-zA-Z]?)[:.\s—–-]?\s*(.*)",
    re.DOTALL
)

SUBFIG_LABEL_RE = re.compile(r"\(([a-zA-Z\d])\)")

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PRIVATE_PNG_CHUNKS = {b"eXIf", b"tEXt", b"zTXt", b"iTXt", b"tIME", b"iCCP"}


def strip_png_metadata(image_path: str) -> List[str]:
    """原子移除 PNG 中可能携带来源/设备信息的附加块。

    像素与透明度相关块保持不变；格式异常时失败关闭，避免把未经检查的图像送往云端。
    返回被移除的块类型，便于审计和测试。
    """
    path = Path(image_path)
    raw = path.read_bytes()
    if not raw.startswith(_PNG_SIGNATURE):
        raise ValueError("隐私清洗仅接受 PNG")
    pos = len(_PNG_SIGNATURE)
    kept = bytearray(_PNG_SIGNATURE)
    removed: List[str] = []
    saw_iend = False
    while pos < len(raw):
        if pos + 12 > len(raw):
            raise ValueError("PNG chunk 截断")
        size = struct.unpack(">I", raw[pos:pos + 4])[0]
        end = pos + 12 + size
        if end > len(raw):
            raise ValueError("PNG chunk 长度越界")
        kind = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + size]
        expected_crc = struct.unpack(">I", raw[pos + 8 + size:end])[0]
        if (zlib.crc32(kind + data) & 0xffffffff) != expected_crc:
            raise ValueError(f"PNG {kind!r} CRC 无效")
        if kind in _PRIVATE_PNG_CHUNKS:
            removed.append(kind.decode("ascii"))
        else:
            kept.extend(raw[pos:end])
        pos = end
        if kind == b"IEND":
            saw_iend = True
            break
    if not saw_iend or pos != len(raw):
        raise ValueError("PNG 缺 IEND 或包含尾随数据")
    if removed:
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".privacy", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(kept)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
    return removed


def _rect_values(rect) -> Tuple[float, float, float, float]:
    if hasattr(rect, "x0"):
        return float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
    return tuple(float(x) for x in rect[:4])  # type: ignore[return-value]


def _nearest_caption(image_bbox, captions: List[Dict[str, Any]],
                     max_gap: float = 180.0) -> Optional[Dict[str, Any]]:
    """按垂直距离、水平重叠与中心偏移匹配最近图注，避免总绑定页面首个 Caption。"""
    if not captions:
        return None
    ix0, iy0, ix1, iy1 = _rect_values(image_bbox)
    iw = max(1.0, ix1 - ix0)
    best = None
    best_score = float("inf")
    for cap in captions:
        cx0, cy0, cx1, cy1 = _rect_values(cap["bbox"])
        if cy0 >= iy1:
            gap = cy0 - iy1
        elif iy0 >= cy1:
            gap = iy0 - cy1
        else:
            gap = 0.0
        if gap > max_gap:
            continue
        overlap = max(0.0, min(ix1, cx1) - max(ix0, cx0))
        overlap_ratio = overlap / max(1.0, min(iw, cx1 - cx0))
        center_delta = abs((ix0 + ix1) / 2.0 - (cx0 + cx1) / 2.0)
        score = gap + 120.0 * (1.0 - min(1.0, overlap_ratio)) + 0.05 * center_delta
        if score < best_score:
            best_score, best = score, cap
    return best


def _caption_line_match(text: str, start: int, caption: str) -> bool:
    if not caption:
        return False
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    line = " ".join(text[line_start:line_end].split()).strip()
    expected = " ".join(str(caption).split()).strip()
    return bool(line and expected and line.casefold() == expected.casefold())


def _reference_contexts(text: str, item_id: str, caption: str = "",
                        radius: int = 220) -> List[str]:
    """抽取 Fig/Tab 正文引用附近的上下文段，供 Caption 绑定和 M1 审计。"""
    kind = "fig" if item_id.startswith("fig") else "tab"
    num = re.escape(item_id[len(kind):])
    prefix = (r"(?:\b(?:figure|fig\.?)|图)" if kind == "fig" else
              r"(?:\b(?:table|tab\.?)|表)")
    pat = re.compile(prefix + r"\s*" + num + r"\b", re.I)
    contexts: List[str] = []
    offset = 0
    for line in (text or "").splitlines(keepends=True):
        clean = " ".join(line.split())
        for match in pat.finditer(line):
            if _caption_line_match(text or "", offset + match.start(), caption):
                continue
            if clean and clean not in contexts:
                contexts.append(clean)
        offset += len(line)
    return contexts


def _find_captions_on_page(page, kind: str = "figure") -> List[Dict[str, Any]]:
    """在页面文本行首搜索 Figure 或 Table 的 Caption 锚点."""
    captions = []
    pattern = FIG_CAPTION_RE if kind == "figure" else TAB_CAPTION_RE
    prefix = "fig" if kind == "figure" else "tab"
    strong_pattern = re.compile(
        (r"(?i)^\s*(?:figure|fig\.?|图)\s*[0-9]+[a-zA-Z]?\s*[:.—–-]" if kind == "figure"
         else r"(?i)^\s*(?:table|tab\.?|表)\s*[0-9]+[a-zA-Z]?\s*[:.—–-]"))

    for item in _page_text_lines(page):
        raw = item["text"].strip()
        m = pattern.match(raw)
        if not m:
            continue
        num = m.group(1).lower()
        captions.append({
            "id": f"{prefix}{num}",
            "num": m.group(1),
            "text": " ".join(raw.split()),
            "bbox": fitz.Rect(item["bbox"]),
            "raw": raw,
            "strong": bool(strong_pattern.match(raw)),
        })
    # If a punctuated caption exists for an ID, discard sentence-like weak duplicates
    # such as "Table 1 reports ..." or "Figure 2 shows ..." on the same page.
    strong_ids = {item["id"] for item in captions if item["strong"]}
    captions = [item for item in captions
                if item["strong"] or item["id"] not in strong_ids]
    return sorted(captions, key=lambda item: (item["bbox"].y0, item["bbox"].x0))


def _page_text_lines(page) -> List[Dict[str, Any]]:
    """Return text lines with their own bboxes; block bboxes are too coarse for captions."""
    rows: List[Dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(str(span.get("text", "")) for span in line.get("spans", []))
            if text.strip() and len(line.get("bbox", [])) == 4:
                rows.append({"text": text, "bbox": tuple(line["bbox"])})
    return rows


def _borderless_table_clip(page, caption: Dict[str, Any],
                           bottom_limit: Optional[float] = None) -> Optional[Any]:
    """Find the local multi-column row run below one caption.

    Page-wide text-table inference readily turns prose into cells. The clip ends at
    the last nearby row that has at least two independently positioned text lines.
    """
    cap_box = fitz.Rect(caption["bbox"])
    max_bottom = min(page.rect.y1, cap_box.y1 + 500,
                     bottom_limit if bottom_limit is not None else page.rect.y1)
    candidates = []
    for line in _page_text_lines(page):
        box = fitz.Rect(line["bbox"])
        if box.y0 >= cap_box.y1 - 1 and box.y0 < max_bottom:
            candidates.append((box, line["text"]))
    candidates.sort(key=lambda item: (item[0].y0, item[0].x0))

    groups: List[List[Tuple[Any, str]]] = []
    for box, text in candidates:
        center = (box.y0 + box.y1) / 2.0
        if groups:
            prev = groups[-1][0][0]
            prev_center = (prev.y0 + prev.y1) / 2.0
        else:
            prev_center = float("inf")
        if groups and abs(center - prev_center) <= 3.0:
            groups[-1].append((box, text))
        else:
            groups.append([(box, text)])

    multi = []
    for group in groups:
        boxes = sorted((item[0] for item in group), key=lambda box: box.x0)
        distinct_columns = sum(
            1 for i, box in enumerate(boxes)
            if i == 0 or box.x0 - boxes[i - 1].x0 >= 12.0)
        if distinct_columns >= 2:
            multi.append(group)
    if not multi:
        return None

    first = multi[0]
    first_top = min(item[0].y0 for item in first)
    if first_top - cap_box.y1 > 180.0:
        return None
    accepted = [first]
    last_center = sum((item[0].y0 + item[0].y1) / 2.0 for item in first) / len(first)
    for group in multi[1:]:
        center = sum((item[0].y0 + item[0].y1) / 2.0 for item in group) / len(group)
        if center - last_center > 80.0:
            break
        accepted.append(group)
        last_center = center
    bottom = max(item[0].y1 for group in accepted for item in group) + 3.0
    return fitz.Rect(page.rect.x0, cap_box.y1, page.rect.x1, min(bottom, page.rect.y1))


def _cluster_drawings(page, gap: float = 35.0) -> List[Dict[str, Any]]:
    """扫描页面的 Drawing 矢量指令块，剔除全页边框与页眉线，按邻近度聚类."""
    drawings = page.get_drawings()
    if not drawings:
        return []

    page_rect = page.rect
    valid_drawings = []

    for d in drawings:
        r = fitz.Rect(d["rect"])
        # 过滤全页面背景/边框底色框
        if r.width > page_rect.width * 0.92 and r.height > page_rect.height * 0.92:
            continue
        # 过滤贯穿页面的页眉/页脚单根分割细线
        if r.height <= 1.5 and r.width > page_rect.width * 0.65:
            continue
        # 过滤孤立微小点
        if r.width <= 1.0 and r.height <= 1.0:
            continue

        cmd_count = len(d.get("items", []))
        valid_drawings.append({
            "rect": r,
            "cmd_count": cmd_count,
            "items": d.get("items", [])
        })

    if not valid_drawings:
        return []

    # 空间邻近连通分支聚类 (DSU / BFS)
    n = len(valid_drawings)
    expanded = [
        fitz.Rect(d["rect"].x0 - gap / 2.0, d["rect"].y0 - gap / 2.0,
                  d["rect"].x1 + gap / 2.0, d["rect"].y1 + gap / 2.0)
        for d in valid_drawings
    ]

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if expanded[i].intersects(expanded[j]):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        q = [i]
        visited[i] = True
        comp = []
        while q:
            curr = q.pop(0)
            comp.append(valid_drawings[curr])
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    q.append(neighbor)

        x0 = min(d["rect"].x0 for d in comp)
        y0 = min(d["rect"].y0 for d in comp)
        x1 = max(d["rect"].x1 for d in comp)
        y1 = max(d["rect"].y1 for d in comp)
        cluster_bbox = fitz.Rect(x0, y0, x1, y1)
        total_cmds = sum(d["cmd_count"] for d in comp)
        area = cluster_bbox.width * cluster_bbox.height

        # 门禁过滤: 仅保留具有实质绘图指令或相当面积的矢量图簇
        if (total_cmds >= 5 and cluster_bbox.width >= 50 and cluster_bbox.height >= 40 and area >= 2500) or \
           (total_cmds >= 3 and area >= 8000 and cluster_bbox.width >= 70 and cluster_bbox.height >= 50):
            clusters.append({
                "bbox": cluster_bbox,
                "cmd_count": total_cmds,
                "drawings": comp,
                "area": area
            })

    return clusters


def _calculate_iou(r1: fitz.Rect, r2: fitz.Rect) -> float:
    """计算两个矩形区域的交并比 (IoU)."""
    intersect = fitz.Rect(r1) & fitz.Rect(r2)
    if intersect.is_empty:
        return 0.0
    inter_area = intersect.width * intersect.height
    union_area = (r1.width * r1.height) + (r2.width * r2.height) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ==============================================================================
# 2. 局部高精光栅化与复合子图拆分 (F05 / F08)
# ==============================================================================

def render_vector_region(page, bbox: Union[Tuple[float, float, float, float], "fitz.Rect"],
                         out_path: str, dpi: int = 200) -> str:
    """对 PDF 页面局部 BBox 区域执行光栅化渲染，输出高清晰度 PNG."""
    rect = fitz.Rect(bbox) & page.rect
    pix = page.get_pixmap(clip=rect, dpi=dpi)
    if pix.n > 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))
    strip_png_metadata(str(out_path))
    return str(out_path)


def _detect_and_gate_subfigures(cluster: Dict[str, Any], caption: str,
                                page_no: int) -> Tuple[List[Dict[str, Any]], bool, str]:
    """F08: 识别 (a)(b)(c) 子图标签，满足 IoU > 0.5 且数量严格吻合时才拆分，否则保留整图."""
    sub_labels = SUBFIG_LABEL_RE.findall(caption)
    if len(sub_labels) < 2:
        return [], False, "single_or_no_subfig"

    # 分析 cluster 内部 drawings 细分子簇
    drawings = cluster.get("drawings", [])
    if len(drawings) < len(sub_labels):
        return [], True, "⚠️ [需人工拆图]"

    # 在较小间距下重聚类子区域
    sub_gap = 12.0
    k = len(drawings)
    exp = [
        fitz.Rect(d["rect"].x0 - sub_gap / 2.0, d["rect"].y0 - sub_gap / 2.0,
                  d["rect"].x1 + sub_gap / 2.0, d["rect"].y1 + sub_gap / 2.0)
        for d in drawings
    ]
    adj = [[] for _ in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            if exp[i].intersects(exp[j]):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * k
    sub_boxes = []
    for i in range(k):
        if visited[i]:
            continue
        q = [i]
        visited[i] = True
        comp = []
        while q:
            curr = q.pop(0)
            comp.append(drawings[curr])
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    q.append(neighbor)
        sb = fitz.Rect(
            min(d["rect"].x0 for d in comp),
            min(d["rect"].y0 for d in comp),
            max(d["rect"].x1 for d in comp),
            max(d["rect"].y1 for d in comp)
        )
        if sb.width >= 30 and sb.height >= 30:
            sub_boxes.append(sb)

    # 按自然阅读顺序 (从上到下，从左到右) 排序
    sub_boxes.sort(key=lambda b: (round(b.y0 / 50.0), b.x0))

    # 门禁条件: 数量必须相等
    if len(sub_boxes) != len(sub_labels):
        return [], True, "⚠️ [需人工拆图]"

    # 估计期望先验位置 (水平网格拆分先验)
    main_b = cluster["bbox"]
    n_subs = len(sub_labels)
    prior_w = main_b.width / n_subs
    priors = [
        fitz.Rect(main_b.x0 + i * prior_w, main_b.y0, main_b.x0 + (i + 1) * prior_w, main_b.y1)
        for i in range(n_subs)
    ]

    # 计算各子图与先验格子的 IoU 门禁
    min_iou = min(_calculate_iou(sb, pb) for sb, pb in zip(sub_boxes, priors))
    if min_iou <= 0.5:
        return [], True, "⚠️ [需人工拆图]"

    # 门禁通过: 输出可拆分子图元数据
    subfigs = []
    for lbl, sb in zip(sub_labels, sub_boxes):
        subfigs.append({
            "label": lbl,
            "bbox": [sb.x0, sb.y0, sb.x1, sb.y1],
            "iou": round(min_iou, 3)
        })
    return subfigs, False, "split_passed"


# ==============================================================================
# 3. 图像与矢量科研图表联合抽取 (F05 / F08)
# ==============================================================================

def extract_images(pdf_path: str, out_dir: str, max_images: int = 5,
                   min_edge: int = 300) -> List[Dict]:
    """抽取 PDF 中内嵌位图与原生矢量科研图表，支持光栅化渲染及元数据完整构建."""
    if not HAS_FITZ:
        return []

    max_images = max(0, min(5, int(max_images)))
    out = Path(out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cands = []
    seen = set()

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return []

    # ---------------- 轨道 1: 扫描内嵌位图 XObjects ----------------
    for pno, page in enumerate(doc):
        page_captions = _find_captions_on_page(page, kind="figure")
        for x in page.get_images(full=True):
            try:
                pix = fitz.Pixmap(doc, x[0])
            except Exception:
                continue
            if pix.width < min_edge or pix.height < min_edge:
                continue
            rects = list(page.get_image_rects(x[0]))
            best_rect = max(rects, key=lambda r: r.width * r.height) if rects else None
            if best_rect is not None:
                page_area = max(1.0, page.rect.width * page.rect.height)
                page_coverage = best_rect.width * best_rect.height / page_area
                if page_coverage >= 0.80:
                    # 整页扫描是 OCR 输入，不是科研配图；禁止把整页图误送 VL。
                    continue
            samples = pix.samples_mv if hasattr(pix, "samples_mv") else pix.samples
            h = hashlib.sha256(samples).hexdigest()[:12]
            if h in seen:
                continue
            seen.add(h)

            area = pix.width * pix.height
            # 尝试在页面上查找位图对应的 BBox
            img_bbox = ([best_rect.x0, best_rect.y0, best_rect.x1, best_rect.y1]
                        if best_rect is not None else None)

            matched = _nearest_caption(
                img_bbox or [0.0, 0.0, pix.width, pix.height], page_captions)
            matched_cap = matched["text"] if matched else ""
            matched_num = matched["num"] if matched else ""

            cands.append({
                "area": area,
                "page": pno,
                # 不持有所有候选 Pixmap；只保留 xref，入选 Top-K 后再解码，压低批量峰值内存。
                "xref": x[0],
                "hash": h,
                "bbox": img_bbox or [0.0, 0.0, float(pix.width), float(pix.height)],
                "caption": matched_cap,
                "num": matched_num or f"p{pno + 1}_img",
                "type": "figure",
                "is_vector": False
            })

    # ---------------- 轨道 2: 扫描原生矢量科研图表 (Drawings) ----------------
    for pno, page in enumerate(doc):
        clusters = _cluster_drawings(page)
        if not clusters:
            continue

        captions = _find_captions_on_page(page, kind="figure")

        # 匹配矢量图簇与 Caption
        used_clusters = set()
        for cap in captions:
            cap_b = cap["bbox"]
            best_idx = None
            best_dist = 999999.0

            for idx, cl in enumerate(clusters):
                if idx in used_clusters:
                    continue
                cl_b = cl["bbox"]
                # 检查垂直距离: 无论 Caption 在图下方还是上方
                if cl_b.y1 <= cap_b.y0 + 20:
                    dist = cap_b.y0 - cl_b.y1
                elif cap_b.y1 <= cl_b.y0 + 20:
                    dist = cl_b.y0 - cap_b.y1
                else:
                    dist = abs(cl_b.y0 - cap_b.y0)

                # 水平重叠奖励
                h_overlap = max(0.0, min(cl_b.x1, cap_b.x1) - max(cl_b.x0, cap_b.x0))
                if h_overlap > 0:
                    dist -= 30.0

                if dist < best_dist and dist < 180.0:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                used_clusters.add(best_idx)
                cl = clusters[best_idx]
                # 扩展裁剪区域 (包括图簇与图注，外扩 8pt 安全边界)
                margin = 8.0
                merged_rect = fitz.Rect(cl["bbox"])
                if best_dist <= 60.0:
                    merged_rect = merged_rect | cap_b
                merged_rect = fitz.Rect(
                    merged_rect.x0 - margin, merged_rect.y0 - margin,
                    merged_rect.x1 + margin, merged_rect.y1 + margin
                ) & page.rect

                subfigs, manual, status_tag = _detect_and_gate_subfigures(cl, cap["text"], pno)
                area_approx = int(merged_rect.width * merged_rect.height * 7.7)  # dpi 200 equivalent
                cands.append({
                    "area": area_approx,
                    "page": pno,
                    "pixmap": None,
                    "render_page": page,
                    "render_bbox": merged_rect,
                    "hash": hashlib.sha256(f"p{pno}_{cap['id']}".encode()).hexdigest()[:12],
                    "bbox": [merged_rect.x0, merged_rect.y0, merged_rect.x1, merged_rect.y1],
                    "caption": cap["text"],
                    "num": cap["num"],
                    "type": "figure",
                    "id": cap["id"],
                    "subfigures": subfigs,
                    "manual": manual,
                    "split_status": status_tag,
                    "is_vector": True
                })

        # 处理无显式 Caption 但绘图指令极其密集的独立图表
        for idx, cl in enumerate(clusters):
            if idx in used_clusters:
                continue
            if cl["cmd_count"] >= 15 and cl["area"] >= 12000:
                margin = 8.0
                render_rect = fitz.Rect(
                    cl["bbox"].x0 - margin, cl["bbox"].y0 - margin,
                    cl["bbox"].x1 + margin, cl["bbox"].y1 + margin
                ) & page.rect
                area_approx = int(render_rect.width * render_rect.height * 7.7)
                cands.append({
                    "area": area_approx,
                    "page": pno,
                    "pixmap": None,
                    "render_page": page,
                    "render_bbox": render_rect,
                    "hash": hashlib.sha256(f"p{pno}_vec_{idx}".encode()).hexdigest()[:12],
                    "bbox": [render_rect.x0, render_rect.y0, render_rect.x1, render_rect.y1],
                    "caption": "",
                    "num": f"{pno + 1}_{idx + 1}",
                    "type": "figure",
                    "id": f"fig_p{pno + 1}_{idx + 1}",
                    "subfigures": [],
                    "manual": False,
                    "is_vector": True
                })

    # ---------------- 排序、截断与落盘渲染 ----------------
    cands.sort(key=lambda c: c["area"], reverse=True)
    kept = []

    # 候选落盘可能因损坏 XObject/渲染异常失败；继续尝试后续候选，直到真正保留 Top-K。
    for item in cands:
        if len(kept) >= max_images:
            break
        pno = item["page"]
        h = item["hash"]
        num_str = item.get("num", "")
        fig_tag = f"fig{num_str}" if num_str and not str(num_str).startswith("fig") else (num_str or f"p{pno}_{h}")
        fp = out / f"fig_p{pno}_{fig_tag}_{h}.png"

        try:
            removed_metadata: List[str] = []
            if item["is_vector"]:
                render_vector_region(item["render_page"], item["render_bbox"], str(fp), dpi=200)
                # 计算光栅化后的像素面积
                pix_rendered = fitz.Pixmap(str(fp))
                pixels = pix_rendered.width * pix_rendered.height
            else:
                pix = fitz.Pixmap(doc, item["xref"])
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(str(fp))
                removed_metadata = strip_png_metadata(str(fp))
                pixels = item["area"]

            kept.append({
                "image": str(fp),
                "page": pno,
                "bbox": item["bbox"],
                "type": item["type"],
                "num": str(item.get("num", "")),
                "id": f"fig{item.get('num', '')}".lower(),
                "caption": item.get("caption", ""),
                "pixels": pixels,
                "metadata_stripped": True,
                "removed_metadata": removed_metadata,
                "subfigures": item.get("subfigures", []),
                "manual": item.get("manual", False)
            })
        except Exception:
            continue

    doc.close()
    return kept


# ==============================================================================
# 4. 表格结构化解析为 Markdown (F06)
# ==============================================================================

def format_markdown_table(data: List[List[Any]]) -> str:
    """将二维单元格网格转换为干净对齐的 Markdown 表格."""
    if not data or not data[0]:
        return ""

    cleaned = []
    for row in data:
        cleaned_row = []
        for cell in row:
            if cell is None:
                c = ""
            else:
                c = str(cell).replace("\r", " ").replace("\n", " ").strip()
                c = c.replace("|", "\\|")
            cleaned_row.append(c)
        cleaned.append(cleaned_row)

    num_cols = max(len(r) for r in cleaned)
    if num_cols == 0:
        return ""

    for r in cleaned:
        while len(r) < num_cols:
            r.append("")

    header_row = "| " + " | ".join(cleaned[0]) + " |"
    sep_row = "| " + " | ".join([":---"] * num_cols) + " |"
    lines = [header_row, sep_row]
    for r in cleaned[1:]:
        lines.append("| " + " | ".join(r) + " |")

    return "\n".join(lines)


def _table_cells(data: List[List[Any]]) -> List[List[str]]:
    """Return a rectangular, content-preserving cell grid for evaluation/storage."""
    if not data:
        return []
    width = max((len(row) for row in data), default=0)
    if width == 0:
        return []
    cells: List[List[str]] = []
    for row in data:
        cleaned = [str(cell or "").replace("\r", " ").replace("\n", " ").strip()
                   for cell in row]
        cleaned.extend([""] * (width - len(cleaned)))
        cells.append(cleaned)
    return cells


def extract_tables(doc_path: str) -> List[Dict[str, Any]]:
    """提取 PDF 中结构化表格并转换为 Markdown 格式，附带定位与编号元数据 (F06)."""
    if not HAS_FITZ:
        return []

    try:
        doc = fitz.open(str(doc_path))
    except Exception:
        return []

    tables = []
    for pno, page in enumerate(doc):
        if not hasattr(page, "find_tables"):
            continue
        page_captions = _find_captions_on_page(page, kind="table")
        detections: List[Tuple[List[List[Any]], Tuple[float, float, float, float],
                               str, Optional[int]]] = []

        def capture(table, strategy: str, caption_index: Optional[int]) -> None:
            # PyMuPDF Table objects can become stale after another find_tables call.
            # Materialize their only required fields immediately.
            try:
                extracted = table.extract()
                bbox = tuple(float(value) for value in table.bbox)
            except Exception:
                return
            if extracted and len(bbox) == 4:
                detections.append((extracted, bbox, strategy, caption_index))

        try:
            line_tables = page.find_tables(strategy="lines")
            for table in list(getattr(line_tables, "tables", []) or []):
                capture(table, "lines", None)
            if page_captions:
                # For every caption not already covered by a ruled table, try a local
                # borderless parse. Mixed ruled/borderless pages must not drop the latter.
                line_boxes = [fitz.Rect(row[1]) for row in detections if row[2] == "lines"]
                covered_caption_indices = set()
                for table_box in line_boxes:
                    nearest_index = None
                    nearest_distance = float("inf")
                    for candidate_index, candidate in enumerate(page_captions):
                        candidate_box = fitz.Rect(candidate["bbox"])
                        if candidate_box.y1 <= table_box.y0 + 20:
                            distance = table_box.y0 - candidate_box.y1
                        elif table_box.y1 <= candidate_box.y0 + 20:
                            distance = candidate_box.y0 - table_box.y1
                        else:
                            distance = 0.0
                        if distance < nearest_distance:
                            nearest_index, nearest_distance = candidate_index, distance
                    if nearest_index is not None and nearest_distance < 150.0:
                        covered_caption_indices.add(nearest_index)
                seen_boxes = set()
                for cap_index, caption in enumerate(page_captions):
                    if cap_index in covered_caption_indices:
                        continue
                    next_top = (page_captions[cap_index + 1]["bbox"].y0
                                if cap_index + 1 < len(page_captions) else None)
                    clip = _borderless_table_clip(page, caption, bottom_limit=next_top)
                    if clip is None or clip.height <= 3:
                        continue
                    found = page.find_tables(strategy="text", clip=clip,
                                             min_words_vertical=2)
                    for table in list(getattr(found, "tables", []) or []):
                        key = tuple(round(float(v), 2) for v in table.bbox)
                        if key not in seen_boxes:
                            seen_boxes.add(key)
                            capture(table, "text", cap_index)
        except Exception:
            continue

        used_captions = set()

        for idx, (data, bbox_values, detection_strategy,
                  forced_caption_idx) in enumerate(detections):
            if not data or len(data) < 1:
                continue
            # text strategy commonly inserts blank separator rows and may absorb the caption.
            data = [row for row in data if any(str(cell or "").strip() for cell in row)]
            if data and TAB_CAPTION_RE.search(" ".join(str(cell or "") for cell in data[0])):
                data = data[1:]
            if not data:
                continue

            cells = _table_cells(data)
            md = format_markdown_table(data)
            t_bbox = fitz.Rect(bbox_values)

            # 匹配距离最近的表格 Caption
            matched_cap = ""
            matched_num = ""
            best_dist = 999999.0
            best_c_idx = None

            if forced_caption_idx is not None:
                best_c_idx = forced_caption_idx
            else:
                for c_idx, cap in enumerate(page_captions):
                    if c_idx in used_captions:
                        continue
                    cap_b = cap["bbox"]
                    # 表格标题通常位于表格上方
                    if cap_b.y1 <= t_bbox.y0 + 20:
                        dist = t_bbox.y0 - cap_b.y1
                    else:
                        dist = abs(cap_b.y0 - t_bbox.y0)
                    if dist < best_dist and dist < 150.0:
                        best_dist = dist
                        best_c_idx = c_idx

            if best_c_idx is not None:
                used_captions.add(best_c_idx)
                matched_cap = page_captions[best_c_idx]["text"]
                matched_num = page_captions[best_c_idx]["num"]

            table_id = f"tab{matched_num.lower()}" if matched_num else f"tab_p{pno + 1}_{idx + 1}"
            tables.append({
                "page": pno,
                "table_id": table_id,
                "num": matched_num or str(idx + 1),
                "caption": matched_cap,
                "cells": cells,
                "markdown": md,
                "bbox": list(bbox_values),
                "rows": len(data),
                "cols": len(data[0]) if data else 0,
                "detection_strategy": detection_strategy,
            })

    doc.close()
    return tables


# ==============================================================================
# 5. 双向图表索引构建与引用解析 (F07)
# ==============================================================================

def build_figure_index(figures: List[Dict], tables: List[Dict],
                       sections: Optional[List[Dict]] = None,
                       paper_id: str = "") -> Dict[str, Any]:
    """构建图表与章节的双向索引 (F07): 支持正向 Sec->Figs/Tabs 与反向 Fig/Tab->Sec/Image/BBox."""
    fig_index: Dict[str, Any] = {}
    by_id: Dict[str, Dict[str, Any]] = {}

    # 注册所有 Figure 对象
    for f in figures:
        f_id = (f.get("id") or f"fig{f.get('num', '')}").lower()
        by_id[f_id] = {
            "id": f_id,
            "type": "figure",
            "num": f.get("num", ""),
            "page": f.get("page", 0),
            "bbox": f.get("bbox", []),
            "caption": f.get("caption", ""),
            "image_path": f.get("image", ""),
            "pixels": f.get("pixels", 0),
            "subfigures": f.get("subfigures", []),
            "manual": f.get("manual", False),
            "bound_contexts": list(f.get("bound_contexts", []))
        }

    # 注册所有 Table 对象
    for t in tables:
        t_id = (t.get("table_id") or f"tab{t.get('num', '')}").lower()
        by_id[t_id] = {
            "id": t_id,
            "type": "table",
            "num": t.get("num", ""),
            "page": t.get("page", 0),
            "bbox": t.get("bbox", []),
            "caption": t.get("caption", ""),
            "markdown": t.get("markdown", ""),
            "cells": t.get("cells", []),
            "rows": t.get("rows", 0),
            "cols": t.get("cols", 0),
            "bound_contexts": list(t.get("bound_contexts", []))
        }

    # 正向建立 Section 引用清单
    if sections:
        for s in sections:
            sec_num = s.get("sec", "0")
            seckey = f"{paper_id}_{sec_num}" if paper_id else str(sec_num)
            text = s.get("text", "")

            # 提取正文中引用的 Fig / Tab 编号
            f_matches = re.findall(r"(?i)\b(?:figure|fig\.?|图)\s*([0-9]+[a-zA-Z]?)", text)
            t_matches = re.findall(r"(?i)\b(?:table|tab\.?|表)\s*([0-9]+[a-zA-Z]?)", text)

            fl = [f"fig{x.lower()}" for x in f_matches]
            tl = [f"tab{x.lower()}" for x in t_matches]

            meta = {}
            for item_id in fl:
                if item_id in by_id:
                    meta[item_id] = by_id[item_id]
                    by_id[item_id]["host_section"] = seckey
                    hosts = by_id[item_id].setdefault("host_sections", [])
                    if seckey not in hosts:
                        hosts.append(seckey)
                    contexts = by_id[item_id].setdefault("bound_contexts", [])
                    contexts.extend(x for x in _reference_contexts(
                        text, item_id, str(by_id[item_id].get("caption", "")))
                                    if x not in contexts)
            for item_id in tl:
                if item_id in by_id:
                    meta[item_id] = by_id[item_id]
                    by_id[item_id]["host_section"] = seckey
                    hosts = by_id[item_id].setdefault("host_sections", [])
                    if seckey not in hosts:
                        hosts.append(seckey)
                    contexts = by_id[item_id].setdefault("bound_contexts", [])
                    contexts.extend(x for x in _reference_contexts(
                        text, item_id, str(by_id[item_id].get("caption", "")))
                                    if x not in contexts)

            fig_index[seckey] = {
                "figs": sorted(list(set(fl))),
                "tabs": sorted(list(set(tl))),
                "meta": meta
            }

    # 挂载双向反查字典
    fig_index["_by_id"] = by_id
    return fig_index


def resolve_citation(citation: str, fig_index: Dict[str, Any],
                     paper_id: str = "") -> Optional[Dict[str, Any]]:
    """将正文引文标签 (如 [Ref: Paper, Fig 1] 或 [Ref: 2024_NeurIPS_01, Tab 2]) 解析为确切图像与元数据."""
    if not citation or not fig_index:
        return None

    by_id = fig_index.get("_by_id", {})

    # 匹配引文中的 Fig 或 Tab 标识
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

    # 二次降级: 在各个节内 meta 中遍历查找
    for k, v in fig_index.items():
        if k.startswith("_"):
            continue
        meta = v.get("meta", {})
        if target_id and target_id in meta:
            return meta[target_id]

    return None


def extract_figures_and_tables(pdf_path: str, out_dir: str,
                               max_images: int = 5) -> Dict[str, Any]:
    """统一抽取入口: 抽取图像、表格并生成双向图表索引."""
    figs = extract_images(pdf_path, out_dir, max_images=max_images)
    tabs = extract_tables(pdf_path)
    index = build_figure_index(figs, tabs)
    return {
        "figures": figs,
        "tables": tabs,
        "fig_index": index
    }


class VisionExtractor:
    """视觉与多模态解析器公共契约类 (PB-SPEC-20260911)."""

    @staticmethod
    def extract_figures_and_tables(pdf_path: str, out_dir: str,
                                   max_images: int = 5) -> Dict[str, Any]:
        return extract_figures_and_tables(pdf_path, out_dir, max_images=max_images)

    @staticmethod
    def render_vector_region(page, bbox: Tuple[float, float, float, float],
                             out_path: str, dpi: int = 200) -> str:
        return render_vector_region(page, bbox, out_path, dpi=dpi)

    @staticmethod
    def extract_images(pdf_path: str, out_dir: str, max_images: int = 5,
                       min_edge: int = 300) -> List[Dict]:
        return extract_images(pdf_path, out_dir, max_images=max_images, min_edge=min_edge)

    @staticmethod
    def extract_tables(doc_path: str) -> List[Dict]:
        return extract_tables(doc_path)


# ==============================================================================
# 6. 多模态解读模型调用与会话管理 (保留生产原逻辑)
# ==============================================================================

def _cleanup_session(session_id: str):
    from . import config
    if not config.cli_cleanup() or not session_id:
        return
    try:
        subprocess.run([config.opencode_bin(), "session", "delete", session_id],
                       capture_output=True, timeout=60)
    except Exception:
        pass


def _prepare_privacy_safe_png(image_path: str) -> str:
    """重编码待上传图像并清除元数据；无法证明已清洗时拒绝继续。"""
    if not HAS_FITZ:
        raise RuntimeError("缺 PyMuPDF，无法完成图像隐私清洗")
    src = Path(image_path)
    if not src.is_file():
        raise RuntimeError(f"图像不存在: {src}")
    scratch = Path(config.opencode_run_dir())
    scratch.mkdir(parents=True, exist_ok=True)
    fd, safe_name = tempfile.mkstemp(prefix="pb_vl_", suffix=".png", dir=str(scratch))
    os.close(fd)
    try:
        pix = fitz.Pixmap(str(src))
        if pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        pix.save(safe_name)
        strip_png_metadata(safe_name)
        return safe_name
    except Exception:
        try:
            os.unlink(safe_name)
        except OSError:
            pass
        raise


def describe_figure(image_path: str, hint: str = "",
                    model: Optional[str] = None, timeout: int = 280,
                    retries: int = 1) -> Dict:
    """返回 {analysis} ; 失败抛 RuntimeError (上游降级跳过, 不伪造)."""
    import json as _json
    if not config.cloud_allowed():
        raise RuntimeError("VL 云端调用未授权；请显式开启 PAPERBRAIN_CLOUD_ALLOWED")
    model = model or config.vl_model()
    if "/" not in model:
        model = "opencode-go/" + model
    prompt = ("用中文描述这张科研配图: 图的类型、坐标轴及单位、关键曲线/数据点结论, ≤150字。"
              + (f" 图注线索: {hint[:200]}" if hint else ""))
    binp = config.opencode_bin()
    wd = config.opencode_run_dir()
    os.makedirs(wd, exist_ok=True)
    safe_image = _prepare_privacy_safe_png(image_path)
    last_error: Exception = RuntimeError("VL 调用未执行")
    try:
        for attempt in range(max(0, int(retries)) + 1):
            texts, session_id = [], ""
            attempted = recorded = False
            cmd = [binp, "run", "--format", "json", "-m", model, prompt,
                   "--file=" + safe_image]
            try:
                from .budget import check_budget
                from .llm import (_BUCKET, _budget_output_cap, _estimate_tokens,
                                  usage_snapshot)
                usage = usage_snapshot()
                _budget_output_cap("vision_text", _estimate_tokens(prompt), 600)
                next_image_count = int(usage.get("vision_images", 0)) + 1
                vision_budget = check_budget({}, num_images=next_image_count)
                if not vision_budget.ok:
                    raise RuntimeError(f"VL 预算超限，拒绝发送: {vision_budget.reasons}")
                _BUCKET.take()
                attempted = True
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=wd)
                for line in (r.stdout or "").splitlines():
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        ev = _json.loads(line)
                    except Exception:
                        continue
                    session_id = session_id or ev.get("sessionID", "")
                    if ev.get("type") == "text" and isinstance(ev.get("part"), dict):
                        t = ev["part"].get("text", "")
                        if t:
                            texts.append(t)
                out = "".join(texts).strip()
                from .llm import record_external_call
                record_external_call("vision_text", prompt, out, vision_images=1)
                recorded = True
                if r.returncode != 0 or not out:
                    raise RuntimeError(
                        f"VL 读图失败 rc={r.returncode}: {(r.stderr or '')[-200:]}")
                _cleanup_session(session_id)
                return {"analysis": out, "model": model,
                        "session_cleaned": bool(session_id), "metadata_stripped": True}
            except Exception as exc:
                last_error = exc
                if attempted and not recorded:
                    try:
                        from .llm import record_external_call
                        record_external_call("vision_text", prompt, "", vision_images=1)
                    except Exception:
                        pass
                _cleanup_session(session_id)
                if attempt < max(0, int(retries)):
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"VL 调用 {max(0, int(retries)) + 1} 次均失败: {last_error}")
    finally:
        try:
            os.unlink(safe_image)
        except OSError:
            pass


def vision_enabled() -> bool:
    if not config.vision_enabled() or not config.cloud_allowed():
        return False
    from shutil import which
    binp = config.opencode_bin()
    return bool(which(binp) or Path(binp).exists())
