"""复合子图二次分割 (v5.0 生产版, OpenCV 可选).

仅当 标签数==切块数 且 IoU>0.5 才拆, 否则保留整图 + 需人工拆图标记.
缺 cv2 时直接返回整图 + needs_tool 标记, 不硬切 (宁可不切).
"""
import re
from typing import Dict, List

SUBCAP_PAT = re.compile(r"\((a|b|c|d|e|f)\)", re.I)


def split_caption(caption: str) -> List[str]:
    parts = SUBCAP_PAT.split(caption or "")
    # split 交替返回 [前文, a, 文本a, b, 文本b...]
    out = []
    for i in range(1, len(parts), 2):
        out.append(parts[i + 1].strip() if i + 1 < len(parts) else "")
    return [o for o in out if o]


def split_figure(image_path: str, caption: str, iou_th: float = 0.5) -> Dict:
    subs = split_caption(caption)
    try:
        import cv2  # type: ignore
    except Exception:
        return {"blocks": [], "split": False, "reason": "needs_tool:缺opencv",
                "sub_captions": subs, "manual": bool(subs)}
    import numpy as np
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"blocks": [], "split": False, "reason": "读图失败", "sub_captions": subs, "manual": True}
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape
    boxes = []
    for c in cnts:
        x, y, bw_, bh_ = cv2.boundingRect(c)
        if bw_ * bh_ < (w * h * 0.02):  # 过滤噪点
            continue
        boxes.append((x, y, bw_, bh_))
    boxes.sort(key=lambda b: (b[1] // max(1, h // 4), b[0]))
    if subs and len(boxes) == len(subs):
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        centers_x = [b[0] + b[2] / 2 for b in boxes]
        centers_y = [b[1] + b[3] / 2 for b in boxes]
        horizontal = (max(centers_x) - min(centers_x)) >= (max(centers_y) - min(centers_y))
        priors = []
        for i in range(len(boxes)):
            if horizontal:
                step = (x1 - x0) / len(boxes)
                priors.append((x0 + i * step, y0, x0 + (i + 1) * step, y1))
            else:
                step = (y1 - y0) / len(boxes)
                priors.append((x0, y0 + i * step, x1, y0 + (i + 1) * step))

        def iou(box, prior):
            bx0, by0, bw, bh = box
            bx1, by1 = bx0 + bw, by0 + bh
            px0, py0, px1, py1 = prior
            iw = max(0.0, min(bx1, px1) - max(bx0, px0))
            ih = max(0.0, min(by1, py1) - max(by0, py0))
            inter = iw * ih
            union = bw * bh + (px1 - px0) * (py1 - py0) - inter
            return inter / union if union > 0 else 0.0

        ious = [iou(box, prior) for box, prior in zip(boxes, priors)]
        if ious and min(ious) > iou_th:
            return {"blocks": [{"bbox": b, "sub": s, "iou": round(score, 3)}
                               for b, s, score in zip(boxes, subs, ious)],
                    "split": True, "iou": round(min(ious), 3),
                    "reason": f"标签数==块数且 min IoU {min(ious):.3f}>{iou_th}"}
        return {"blocks": [], "split": False,
                "reason": f"标签数==块数但 min IoU {min(ious or [0]):.3f}<={iou_th}, 保留整图",
                "sub_captions": subs, "manual": True, "iou": round(min(ious or [0]), 3)}
    return {"blocks": [], "split": False,
            "reason": f"块数{len(boxes)}!=标签数{len(subs)}, 保留整图",
            "sub_captions": subs, "manual": True, "iou_th": iou_th}
