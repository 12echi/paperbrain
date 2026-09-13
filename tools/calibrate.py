"""Calibrate the lexical citation scorer from a labeled claim/source set.

Input JSON must contain at least 50 rows and at least 20 positive/20 negative rows:
  [{"claim": "...", "source": "...", "label": true}, ...]
Rows may optionally carry a precomputed numeric ``score`` from the scorer being
calibrated. The output records the dataset hash and prompt/scorer version so a
small demo or an unrelated scorer cannot masquerade as production calibration.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _percentile(values, fraction: float) -> float:
    values = sorted(float(v) for v in values)
    if not values:
        raise ValueError("空分布")
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


def _roc_auc(pos, neg) -> float:
    """Mann-Whitney ROC AUC with half credit for tied scores."""
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def calibrate(rows, scorer_name: str, prompt_version: str,
              max_false_pass_rate: float = 0.05,
              min_positive_recall: float = 0.80,
              min_roc_auc: float = 0.80) -> dict:
    if not (0.0 <= max_false_pass_rate <= 0.05 and
            0.80 <= min_positive_recall <= 1.0 and
            0.80 <= min_roc_auc <= 1.0):
        raise ValueError("质量目标不得弱于 FPR<=0.05、Recall>=0.80、ROC-AUC>=0.80")
    if not isinstance(rows, list) or len(rows) < 50:
        raise ValueError("生产标定至少需要 50 对 claim-source")
    parsed = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not str(row.get("claim", "")).strip() or \
                not str(row.get("source", "")).strip() or "label" not in row:
            raise ValueError(f"第 {i} 行缺 claim/source/label")
        label = row["label"]
        if label not in (True, False, 0, 1):
            raise ValueError(f"第 {i} 行 label 必须为 bool/0/1")
        parsed.append({"claim": str(row["claim"]), "source": str(row["source"]),
                       "label": bool(label), "score": row.get("score")})
    n_pos = sum(x["label"] for x in parsed)
    n_neg = len(parsed) - n_pos
    if n_pos < 20 or n_neg < 20:
        raise ValueError("正例和负例各至少需要 20 对")

    if all(isinstance(x["score"], (int, float)) for x in parsed):
        scores = [float(x["score"]) for x in parsed]
        score_source = "precomputed"
    else:
        sys.path.insert(0, str(ROOT))
        from paperbrain.retrieval import make_scorer
        corpus = {f"S{i:03d}": row["source"] for i, row in enumerate(parsed)}
        scorer = make_scorer(corpus)
        scores = [scorer(row["claim"], row["source"]) for row in parsed]
        score_source = "computed"
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
        raise ValueError("score 必须是 0..1 的有限数")
    pos = [score for score, row in zip(scores, parsed) if row["label"]]
    neg = [score for score, row in zip(scores, parsed) if not row["label"]]
    neg_p95 = _percentile(neg, 0.95)
    pos_p10 = _percentile(pos, 0.10)
    candidates = sorted(set(scores + [1.0]))
    operating_points = []
    for threshold in candidates:
        fpr = sum(score >= threshold for score in neg) / len(neg)
        recall = sum(score >= threshold for score in pos) / len(pos)
        if fpr <= max_false_pass_rate:
            operating_points.append((recall, fpr, threshold))
    if operating_points:
        positive_recall, false_pass_rate, raw_pass_th = max(
            operating_points, key=lambda point: (point[0], -point[1], -point[2]))
    else:  # only possible when all negatives score 1.0
        raw_pass_th = 1.0
        false_pass_rate = sum(score >= raw_pass_th for score in neg) / len(neg)
        positive_recall = sum(score >= raw_pass_th for score in pos) / len(pos)
    pass_th = float(raw_pass_th)
    review_th = max(0.0, min(neg_p95, math.nextafter(pass_th, -math.inf)))
    auc = _roc_auc(pos, neg)
    quality_ok = (false_pass_rate <= max_false_pass_rate and
                  positive_recall >= min_positive_recall and auc >= min_roc_auc and
                  0.0 <= review_th < pass_th <= 1.0)
    dataset_payload = json.dumps(parsed, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")
    return {
        "calibrated": quality_ok,
        "calibration_method": "empirical-roc-v1",
        "scorer": scorer_name,
        "score_source": score_source,
        "prompt_version": prompt_version,
        "dataset_sha256": hashlib.sha256(dataset_payload).hexdigest(),
        "n_pairs": len(parsed), "n_pos": n_pos, "n_neg": n_neg,
        "pos_p10": round(pos_p10, 4), "neg_p95": round(neg_p95, 4),
        "review_th": review_th, "pass_th": pass_th,
        "false_pass_rate": round(false_pass_rate, 4),
        "positive_recall": round(positive_recall, 4),
        "roc_auc": round(auc, 4),
        "quality_gate": {"max_false_pass_rate": max_false_pass_rate,
                         "min_positive_recall": min_positive_recall,
                         "min_roc_auc": min_roc_auc},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", default=str(ROOT / "thresholds.json"))
    parser.add_argument("--scorer", default="tfidf-corpus")
    parser.add_argument("--prompt-version", default="v5.0")
    parser.add_argument("--max-false-pass-rate", type=float, default=0.05)
    parser.add_argument("--min-positive-recall", type=float, default=0.80)
    parser.add_argument("--min-roc-auc", type=float, default=0.80)
    args = parser.parse_args()
    try:
        with open(args.pairs, encoding="utf-8") as f:
            rows = json.load(f)
        result = calibrate(rows, args.scorer, args.prompt_version,
                           args.max_false_pass_rate, args.min_positive_recall,
                           args.min_roc_auc)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"标定失败: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
