"""Generate M5 contradiction predictions through the real consistency detector.

Input is a JSON list containing unique ``id``, non-empty ``draft``, and either
``memory: {relations: [...]}`` or a top-level ``relations`` list. Labels are ignored.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperbrain.consistency import check_contradicts
from tools.score_m5 import case_sha256


def generate(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases 必须是非空 JSON list")
    output, seen = [], set()
    for index, row in enumerate(cases):
        if not isinstance(row, dict):
            raise ValueError(f"cases[{index}] 必须是对象")
        case_id = str(row.get("id", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"cases[{index}] 缺唯一 id")
        seen.add(case_id)
        digest = case_sha256(row)
        memory = row.get("memory")
        if not isinstance(memory, dict):
            memory = {"relations": row.get("relations", [])}
        hits = check_contradicts(memory, row["draft"])
        output.append({"id": case_id, "case_sha256": digest,
                       "flagged": bool(hits), "hit_count": len(hits)})
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()
    try:
        cases = json.loads(Path(args.cases).expanduser().read_text(encoding="utf-8"))
        output = generate(cases)
        target = Path(args.pred).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"M5 预测生成失败: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"result": "OK", "cases": len(output),
                      "pred": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
