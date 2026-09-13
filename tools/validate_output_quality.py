"""Validate hash-bound, independent human review of user-visible outputs."""
import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SCHEMA = "paperbrain-output-quality-v1"
PROTOCOL = "independent-double-review-v1"
DIMENSIONS = (
    "factual_accuracy",
    "evidence_traceability",
    "learning_objective_quality",
    "insight_depth",
    "actionability",
    "language_coherence",
)
MIN_CASES = 20
MIN_DISTINCT_SOURCES = 10
MIN_TASKS = {"full_read": 10, "method": 5, "review": 5}
MIN_MEAN = 4.0
MAX_DISAGREEMENT = 2


def _resolve(value, base_dir):
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = Path(base_dir or Path.cwd()) / path
    return path.resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _review_time(value):
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.tzinfo is not None
    except ValueError:
        return False


def evaluate(report, base_dir=None, min_cases=MIN_CASES,
             min_distinct_sources=MIN_DISTINCT_SOURCES, min_tasks=None):
    errors = []
    if not isinstance(report, dict):
        raise ValueError("M8 report 必须是 JSON 对象")
    if report.get("schema") != SCHEMA:
        errors.append(f"schema 必须为 {SCHEMA}")
    if report.get("protocol") != PROTOCOL:
        errors.append(f"protocol 必须为 {PROTOCOL}")
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise ValueError("M8 report.cases 必须是数组")
    required_tasks = dict(MIN_TASKS if min_tasks is None else min_tasks)
    seen_ids, artifact_hashes, source_hashes = set(), set(), set()
    task_counts = Counter()
    dimension_values = {name: [] for name in DIMENSIONS}
    valid_cases = 0

    for index, row in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{prefix} 必须是对象")
            continue
        case_id = str(row.get("id", "")).strip()
        if not case_id or case_id in seen_ids:
            errors.append(f"{prefix}.id 缺失或重复")
        else:
            seen_ids.add(case_id)
        task = str(row.get("task", "")).strip()
        if task not in required_tasks:
            errors.append(f"{prefix}.task 不在 {sorted(required_tasks)}")
        else:
            task_counts[task] += 1
        try:
            source = _resolve(row.get("source_path"), base_dir)
            artifact = _resolve(row.get("artifact_path"), base_dir)
            if not source.is_file() or source.stat().st_size == 0:
                raise ValueError(f"source_path 无有效文件: {source}")
            if not artifact.is_file() or artifact.stat().st_size < 200:
                raise ValueError(f"artifact_path 不存在或小于200字节: {artifact}")
            actual_source = _sha256(source)
            actual_artifact = _sha256(artifact)
            if str(row.get("source_sha256", "")).lower() != actual_source:
                errors.append(f"{prefix}.source_sha256 与当前文件不匹配")
            if str(row.get("artifact_sha256", "")).lower() != actual_artifact:
                errors.append(f"{prefix}.artifact_sha256 与当前文件不匹配")
            source_hashes.add(actual_source)
            if actual_artifact in artifact_hashes:
                errors.append(f"{prefix} 产物内容重复，禁止复制样本刷数量")
            artifact_hashes.add(actual_artifact)
        except (OSError, ValueError) as exc:
            errors.append(f"{prefix}: {exc}")

        reviews = row.get("reviews")
        if not isinstance(reviews, list) or len(reviews) < 2:
            errors.append(f"{prefix}.reviews 至少需要两名独立人工评审")
            continue
        reviewer_ids, case_scores = set(), {name: [] for name in DIMENSIONS}
        case_valid = True
        for review_index, review in enumerate(reviews):
            rprefix = f"{prefix}.reviews[{review_index}]"
            if not isinstance(review, dict):
                errors.append(f"{rprefix} 必须是对象")
                case_valid = False
                continue
            reviewer = str(review.get("reviewer_id", "")).strip()
            if not reviewer or reviewer in reviewer_ids:
                errors.append(f"{rprefix}.reviewer_id 缺失或同案例重复")
                case_valid = False
            reviewer_ids.add(reviewer)
            if review.get("attestation") != "human":
                errors.append(f"{rprefix}.attestation 必须为 human")
                case_valid = False
            if not _review_time(review.get("reviewed_at")):
                errors.append(f"{rprefix}.reviewed_at 必须是带时区 ISO 时间")
                case_valid = False
            notes = review.get("notes")
            if not isinstance(notes, str) or len(notes.strip()) < 10:
                errors.append(f"{rprefix}.notes 至少10个字符")
                case_valid = False
            blockers = review.get("blocking_issues")
            if not isinstance(blockers, list) or any(not isinstance(item, str) for item in blockers):
                errors.append(f"{rprefix}.blocking_issues 必须是字符串数组")
                case_valid = False
            elif any(item.strip() for item in blockers):
                errors.append(f"{rprefix} 存在阻断问题")
                case_valid = False
            scores = review.get("scores")
            if not isinstance(scores, dict):
                errors.append(f"{rprefix}.scores 必须是对象")
                case_valid = False
                continue
            for dimension in DIMENSIONS:
                score = scores.get(dimension)
                if type(score) is not int or not 1 <= score <= 5:
                    errors.append(f"{rprefix}.scores.{dimension} 必须是1-5整数")
                    case_valid = False
                else:
                    case_scores[dimension].append(score)
        for dimension, values in case_scores.items():
            if len(values) >= 2 and max(values) - min(values) > MAX_DISAGREEMENT:
                errors.append(f"{prefix}.{dimension} 评审分歧超过 {MAX_DISAGREEMENT}")
                case_valid = False
            if values and sum(values) / len(values) < MIN_MEAN:
                errors.append(f"{prefix}.{dimension} 均分低于 {MIN_MEAN}")
                case_valid = False
        if case_valid:
            valid_cases += 1
            for dimension, values in case_scores.items():
                dimension_values[dimension].extend(values)

    if len(cases) < min_cases:
        errors.append(f"案例数 {len(cases)} < {min_cases}")
    if len(source_hashes) < min_distinct_sources:
        errors.append(f"不同源文件数 {len(source_hashes)} < {min_distinct_sources}")
    for task, required in required_tasks.items():
        if task_counts[task] < required:
            errors.append(f"任务 {task} 案例数 {task_counts[task]} < {required}")
    means = {name: (sum(values) / len(values) if values else 0.0)
             for name, values in dimension_values.items()}
    for name, mean in means.items():
        if mean < MIN_MEAN:
            errors.append(f"总体 {name} 均分 {mean:.3f} < {MIN_MEAN}")
    ok = not errors and valid_cases == len(cases)
    return {"result": "PASS" if ok else "FAIL", "cases": len(cases),
            "valid_cases": valid_cases, "distinct_sources": len(source_hashes),
            "task_counts": dict(task_counts),
            "dimension_means": {key: round(value, 4) for key, value in means.items()},
            "errors": errors}


def initialize(manifest, base_dir=None):
    """Create hash-bound review rows while leaving every human judgment empty."""
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("初始化 manifest 必须是非空 JSON 数组")
    cases, seen = [], set()
    for index, row in enumerate(manifest):
        if not isinstance(row, dict):
            raise ValueError(f"manifest[{index}] 必须是对象")
        case_id = str(row.get("id", "")).strip()
        task = str(row.get("task", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"manifest[{index}].id 缺失或重复")
        if task not in MIN_TASKS:
            raise ValueError(f"manifest[{index}].task 不在 {sorted(MIN_TASKS)}")
        seen.add(case_id)
        source = _resolve(row.get("source_path"), base_dir)
        artifact = _resolve(row.get("artifact_path"), base_dir)
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"manifest[{index}].source_path 无有效文件")
        if not artifact.is_file() or artifact.stat().st_size < 200:
            raise ValueError(f"manifest[{index}].artifact_path 不存在或小于200字节")
        blank_review = {
            "reviewer_id": "",
            "attestation": "human",
            "reviewed_at": "",
            "scores": {dimension: None for dimension in DIMENSIONS},
            "blocking_issues": ["REVIEW_REQUIRED"],
            "notes": "",
        }
        cases.append({"id": case_id, "task": task,
                      "source_path": str(source), "artifact_path": str(artifact),
                      "source_sha256": _sha256(source),
                      "artifact_sha256": _sha256(artifact),
                      "reviews": [copy.deepcopy(blank_review), copy.deepcopy(blank_review)]})
    return {"schema": SCHEMA, "protocol": PROTOCOL, "cases": cases}


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report")
    mode.add_argument("--init-manifest")
    parser.add_argument("--output", help="--init-manifest 模式的模板输出路径")
    args = parser.parse_args()
    try:
        if args.init_manifest:
            if not args.output:
                raise ValueError("--init-manifest 必须同时提供 --output")
            manifest_path = Path(args.init_manifest).expanduser().resolve()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            template = initialize(manifest, base_dir=manifest_path.parent)
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"result": "INITIALIZED", "cases": len(template["cases"]),
                              "output": str(output)}, ensure_ascii=False))
            return 0
        target = Path(args.report).expanduser().resolve()
        report = json.loads(target.read_text(encoding="utf-8"))
        result = evaluate(report, base_dir=target.parent)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"M8 输入无效: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
