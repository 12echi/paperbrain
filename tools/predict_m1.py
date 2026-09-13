"""Generate M1 predictions from the real offline parsing pipeline.

Manifest format: JSON list of {"id": "...", "path": "/path/to/source"}.
The output contains no source paths, only source SHA-256 and parser results.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperbrain.pipeline import run_outline


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise ValueError("manifest id 不能映射为空目录名")
    return safe[:120]


def generate(manifest: list, output_path: str, work_dir: Optional[str] = None) -> list:
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("manifest 必须是非空 JSON list")
    owned_tmp = None
    if work_dir:
        base = Path(work_dir).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
    else:
        owned_tmp = tempfile.mkdtemp(prefix="paperbrain-m1-")
        base = Path(owned_tmp)
    results = []
    seen = set()
    offline_keys = ("PAPERBRAIN_VISION", "PAPERBRAIN_CLOUD_ALLOWED",
                    "PAPERBRAIN_ALL_MODEL", "PAPERBRAIN_PROVIDER")
    old_env = {key: os.environ.get(key) for key in offline_keys}
    os.environ.update({"PAPERBRAIN_VISION": "0", "PAPERBRAIN_CLOUD_ALLOWED": "0",
                       "PAPERBRAIN_ALL_MODEL": "0", "PAPERBRAIN_PROVIDER": "off"})
    try:
        for index, row in enumerate(manifest):
            if not isinstance(row, dict):
                raise ValueError(f"manifest[{index}] 必须是对象")
            paper_id = str(row.get("id", "")).strip()
            if not paper_id or paper_id in seen:
                raise ValueError(f"manifest[{index}] 缺唯一 id")
            seen.add(paper_id)
            source = Path(str(row.get("path") or row.get("source_path") or "")).expanduser().resolve()
            if not source.is_file():
                raise ValueError(f"{paper_id}: 源文件不存在: {source}")
            id_digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:10]
            run_dir = base / f"{index + 1:03d}_{_safe_id(paper_id)}_{id_digest}"
            record_path = run_dir / "m1_prediction.json"
            run_outline(str(source), paper_id, str(run_dir), use_llm=False, vision=False,
                        m1_pred_path=str(record_path))
            results.append(json.loads(record_path.read_text(encoding="utf-8")))
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return results
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if owned_tmp:
            shutil.rmtree(owned_tmp, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--work-dir")
    args = parser.parse_args()
    try:
        manifest = json.loads(Path(args.manifest).expanduser().read_text(encoding="utf-8"))
        rows = generate(manifest, args.pred, args.work_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"M1 预测生成失败: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"result": "OK", "papers": len(rows),
                      "pred": str(Path(args.pred).expanduser().resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
