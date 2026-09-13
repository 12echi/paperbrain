"""M3 graph gate: alias merge accuracy, generic rate, and schema leakage."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paperbrain.graph import ALLOWED_ENTITIES, is_generic

MIN_GOLDEN = 50


def case_sha256(row):
    if not isinstance(row, dict):
        raise ValueError("案例必须是对象")
    mention = row.get("mention")
    input_type = row.get("input_type", row.get("type"))
    if not isinstance(mention, str) or not mention.strip() or not isinstance(input_type, str):
        raise ValueError("案例必须含非空 mention 与字符串 input_type/type")
    payload = {"mention": mention, "input_type": input_type}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def policy_sha256():
    from paperbrain import config
    return hashlib.sha256(config.graph_policy_file().read_bytes()).hexdigest()


def _index(rows, label, allow_empty=False):
    if not isinstance(rows, list):
        raise ValueError(f"{label} 必须是 JSON list")
    out = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label}[{i}] 必须是对象")
        mention = str(row.get("mention", "")).strip()
        if not mention or mention in out:
            raise ValueError(f"{label}[{i}] mention 缺失或重复")
        out[mention] = row
    if not out and not allow_empty:
        raise ValueError(f"{label} 为空")
    return out


def evaluate(golden_rows, pred_rows, min_alias=0.97, max_generic=0.01,
             min_golden=MIN_GOLDEN):
    golden, pred = _index(golden_rows, "golden"), _index(pred_rows, "pred", allow_empty=True)
    missing = sorted(set(golden) - set(pred))
    extra = sorted(set(pred) - set(golden))
    source_errors = []
    expected_policy = policy_sha256()
    case_hashes = []
    for mention, row in golden.items():
        try:
            expected_case = case_sha256(row)
            case_hashes.append(expected_case)
        except ValueError as exc:
            source_errors.append(f"golden[{mention}]: {exc}")
            continue
        predicted = pred.get(mention, {})
        if str(predicted.get("case_sha256", "")).lower() != expected_case:
            source_errors.append(f"pred[{mention}] case_sha256 不匹配")
        if str(predicted.get("policy_sha256", "")).lower() != expected_policy:
            source_errors.append(f"pred[{mention}] graph policy 哈希不匹配")
    if len(set(case_hashes)) != len(case_hashes):
        source_errors.append("golden 案例必须逐项唯一")
    correct = sum(
        str(pred[m].get("canonical", "")).strip() == str(g.get("canonical", "")).strip() and
        str(pred[m].get("type", "")).strip() == str(g.get("type", "")).strip()
        for m, g in golden.items() if m in pred)
    alias_accuracy = correct / len(golden)
    outputs = [row for row in pred.values() if str(row.get("canonical", "")).strip()]
    generic = sum(is_generic(str(row.get("canonical", ""))) for row in outputs)
    generic_rate = generic / len(outputs) if outputs else 1.0
    schema_outside = sum(str(row.get("type", "")).strip() not in ALLOWED_ENTITIES
                         for row in outputs)
    source_bound = not source_errors
    dataset_complete = len(golden) >= min_golden and source_bound
    ok = (dataset_complete and not missing and not extra and alias_accuracy >= min_alias and
          generic_rate < max_generic and schema_outside == 0)
    return {"n_golden": len(golden), "n_pred": len(pred), "missing": missing,
            "extra": extra, "source_errors": source_errors,
            "alias_accuracy": round(alias_accuracy, 4),
            "generic_rate": round(generic_rate, 4), "schema_outside": schema_outside,
            "gate": {"min_alias": min_alias, "max_generic_exclusive": max_generic,
                     "schema_outside": 0, "min_golden": min_golden,
                     "source_bound": source_bound, "dataset_complete": dataset_complete},
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
        print(f"M3 输入无效: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
