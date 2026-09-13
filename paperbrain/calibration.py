"""Validation of citation-scorer calibration records."""
import json
from pathlib import Path
from typing import Dict, Optional


def load_record(scorer: str, path: Optional[Path] = None,
                prompt_version: str = "v5.0") -> Dict:
    target = path or (Path(__file__).resolve().parent.parent / "thresholds.json")
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"verified": False, "reason": f"标定记录不可读: {exc}"}
    reasons = []
    if record.get("calibrated") is not True:
        reasons.append("记录未标为 calibrated")
    if record.get("scorer") != scorer:
        reasons.append(f"scorer 不匹配 ({record.get('scorer')} != {scorer})")
    if record.get("prompt_version") != prompt_version:
        reasons.append("prompt_version 不匹配")
    if record.get("calibration_method") != "empirical-roc-v1":
        reasons.append("缺 empirical-roc-v1 标定方法")
    n_pairs = int(record.get("n_pairs", 0) or 0)
    n_pos = int(record.get("n_pos", 0) or 0)
    n_neg = int(record.get("n_neg", 0) or 0)
    if n_pairs < 50 or n_pos < 20 or n_neg < 20 or n_pos + n_neg != n_pairs:
        reasons.append("需要至少 50 对且正负例各至少 20 对")
    digest = str(record.get("dataset_sha256", ""))
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        reasons.append("缺有效 dataset_sha256")
    try:
        gate = record["quality_gate"]
        max_fpr = float(gate["max_false_pass_rate"])
        min_recall = float(gate["min_positive_recall"])
        min_auc = float(gate["min_roc_auc"])
        fpr = float(record["false_pass_rate"])
        recall = float(record["positive_recall"])
        auc = float(record["roc_auc"])
        if not (0.0 <= max_fpr <= 0.05 and 0.80 <= min_recall <= 1.0 and
                0.80 <= min_auc <= 1.0 and fpr <= max_fpr and
                recall >= min_recall and auc >= min_auc):
            reasons.append("ROC 质量门禁未达标")
    except (KeyError, TypeError, ValueError):
        reasons.append("缺完整 ROC 质量记录")
    try:
        review_th = float(record["review_th"])
        pass_th = float(record["pass_th"])
        if not (0.0 <= review_th < pass_th <= 1.0):
            reasons.append("阈值范围无效")
    except (KeyError, TypeError, ValueError):
        review_th, pass_th = 0.35, 0.82
        reasons.append("阈值缺失")
    return {"verified": not reasons, "reason": "; ".join(reasons),
            "review_th": review_th, "pass_th": pass_th,
            "path": str(target), "record": record}


def enforce_review(report: Dict, calibration: Dict) -> Dict:
    """未验证标定时禁止 CLEAN，但保留逐条引文分数供人工复核。"""
    if calibration.get("verified"):
        return report
    report.setdefault("report", []).append({
        "citation": "[CALIBRATION]", "status": "NEEDS_REVIEW",
        "reason": calibration.get("reason") or "引文打分器缺生产标定记录",
    })
    if report.get("status") == "CLEAN":
        report["status"] = "NEEDS_REVIEW"
    report["is_clean"] = False
    return report
