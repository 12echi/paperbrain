"""Generate M3 alias/schema predictions through the real graph sanitizer.

Each case needs unique ``mention`` and ``input_type`` (or ``type``). Expected
``canonical``/``type`` fields, when present, are ignored during prediction.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperbrain.graph import sanitize_entities
from tools.score_m3 import case_sha256, policy_sha256


def generate(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases 必须是非空 JSON list")
    output, seen = [], set()
    policy_digest = policy_sha256()
    for index, row in enumerate(cases):
        if not isinstance(row, dict):
            raise ValueError(f"cases[{index}] 必须是对象")
        mention = str(row.get("mention", "")).strip()
        if not mention or mention in seen:
            raise ValueError(f"cases[{index}] 缺唯一 mention")
        seen.add(mention)
        input_type = row.get("input_type", row.get("type"))
        digest = case_sha256({"mention": mention, "input_type": input_type})
        kept, stats = sanitize_entities([{"name": mention, "type": input_type}])
        entity = kept[0] if kept else {}
        output.append({"mention": mention, "case_sha256": digest,
                       "policy_sha256": policy_digest,
                       "canonical": entity.get("name", ""),
                       "type": entity.get("type", ""),
                       "rejected_schema": stats.get("rejected_schema", 0),
                       "rejected_generic": stats.get("rejected_generic", 0)})
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
        print(f"M3 预测生成失败: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"result": "OK", "cases": len(output),
                      "pred": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
