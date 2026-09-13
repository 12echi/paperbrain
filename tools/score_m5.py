"""M5 contradiction golden gate; missing predictions count as misses."""
import argparse
import hashlib
import json
import sys

MIN_POSITIVES = 20
MIN_NEGATIVES = 10


def case_sha256(row):
    draft = row.get("draft") if isinstance(row, dict) else None
    memory = row.get("memory") if isinstance(row, dict) else None
    relations = memory.get("relations") if isinstance(memory, dict) else row.get("relations")
    if not isinstance(draft, str) or not draft.strip() or not isinstance(relations, list):
        raise ValueError("案例必须含非空 draft 与 memory.relations/relations 数组")
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict) or not all(
                isinstance(relation.get(field), str) for field in ("from", "to", "rel")):
            raise ValueError(f"relations[{index}] 缺字符串 from/to/rel")
    payload = {"draft": draft, "relations": relations}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _index(rows, label, allow_empty=False):
    if not isinstance(rows, list):
        raise ValueError(f"{label} 必须是 JSON list")
    out = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{i}] 必须是对象")
        key = str(row.get("id", "")).strip()
        if not key or key in out:
            raise ValueError(f"{label}[{i}] id 缺失或重复")
        out[key] = row
    if not out and not allow_empty:
        raise ValueError(f"{label} 为空")
    return out


def evaluate(golden_rows, pred_rows, min_recall=0.90, max_fpr=0.10,
             min_positives=MIN_POSITIVES, min_negatives=MIN_NEGATIVES):
    golden, pred = _index(golden_rows, "golden"), _index(pred_rows, "pred", allow_empty=True)
    for key, row in golden.items():
        if type(row.get("expected")) is not bool:
            raise ValueError(f"golden[{key}] expected 必须为 JSON boolean")
    for key, row in pred.items():
        if type(row.get("flagged")) is not bool:
            raise ValueError(f"pred[{key}] flagged 必须为 JSON boolean")
    source_errors = []
    case_hashes = []
    for key, row in golden.items():
        try:
            expected_hash = case_sha256(row)
            case_hashes.append(expected_hash)
        except ValueError as exc:
            source_errors.append(f"golden[{key}]: {exc}")
            continue
        actual_hash = str(pred.get(key, {}).get("case_sha256", "")).lower()
        if actual_hash != expected_hash:
            source_errors.append(f"pred[{key}] case_sha256 不匹配")
    if len(set(case_hashes)) != len(case_hashes):
        source_errors.append("golden 案例必须逐项唯一，禁止重复案例刷样本数")
    missing = sorted(set(golden) - set(pred))
    positives = [key for key, row in golden.items() if row["expected"] is True]
    negatives = [key for key, row in golden.items() if row["expected"] is False]
    if not positives:
        raise ValueError("golden 至少需要一个矛盾正例")
    tp = sum(pred.get(key, {}).get("flagged") is True for key in positives)
    fp = sum(pred.get(key, {}).get("flagged") is True for key in negatives)
    recall = tp / len(positives)
    precision = tp / (tp + fp) if tp + fp else 0.0
    fpr = fp / len(negatives) if negatives else 1.0
    source_bound = not source_errors
    dataset_complete = (len(positives) >= min_positives and len(negatives) >= min_negatives and
                        source_bound)
    ok = dataset_complete and not missing and recall >= min_recall and fpr <= max_fpr
    return {"n_golden": len(golden), "missing": missing,
            "source_errors": source_errors, "tp": tp, "fp": fp,
            "recall": round(recall, 4), "precision": round(precision, 4),
            "false_positive_rate": round(fpr, 4),
            "gate": {"min_recall": min_recall, "max_fpr": max_fpr,
                     "min_positives": min_positives,
                     "min_negatives": min_negatives, "source_bound": source_bound,
                     "dataset_complete": dataset_complete},
            "result": "PASS" if ok else "FAIL"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()
    try:
        with open(args.golden, encoding="utf-8") as f:
            golden = json.load(f)
        with open(args.pred, encoding="utf-8") as f:
            pred = json.load(f)
        result = evaluate(golden, pred)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"M5 输入无效: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
