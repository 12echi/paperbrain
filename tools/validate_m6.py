"""M6 50-paper resource, resume, and idempotency gate.

Runs exactly 50 distinct documents offline, samples process/host/cgroup memory, then
re-runs with resume enabled and verifies that paper artifacts are byte-identical.
The gate intentionally fails outside a 3 GiB-limited container because Docker memory
compliance cannot be inferred from a local process run.
"""
import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _tree_signature(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "batch_ledger.csv":
            continue
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(bytes.fromhex(_sha(path)))
    return h.hexdigest()


def _read_int(path: str):
    try:
        value = Path(path).read_text(encoding="ascii").strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _host_rss_mb():
    try:
        out = subprocess.check_output(["ps", "-axo", "rss="], text=True, timeout=3)
        return sum(int(x) for x in out.split() if x.isdigit()) / 1024.0
    except Exception:
        return None


class MemorySampler:
    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.stop = threading.Event()
        self.host_peak_mb = 0.0
        self.cgroup_peak_mb = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            host = _host_rss_mb()
            if host is not None:
                self.host_peak_mb = max(self.host_peak_mb, host)
            current = _read_int("/sys/fs/cgroup/memory.current")
            if current is not None:
                self.cgroup_peak_mb = max(self.cgroup_peak_mb, current / 1024 / 1024)
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)


def _process_peak_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value / 1024 / 1024 if sys.platform == "darwin" else value / 1024


def _p95(values) -> float:
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    return values[max(0, math.ceil(0.95 * len(values)) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="包含真实论文的目录")
    parser.add_argument("--out", required=True, help="必须为空或不存在的验收输出目录")
    parser.add_argument("--task", default="full_read")
    parser.add_argument("--depth", default="standard")
    parser.add_argument("--host-peak-mb", type=float,
                        help="同次运行由 Activity Monitor 记录的整机峰值")
    parser.add_argument("--docker-peak-mb", type=float,
                        help="同次运行由 docker stats 记录的容器峰值")
    args = parser.parse_args()

    source = Path(args.dir).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not source.is_dir():
        print("输入目录不存在", file=sys.stderr)
        return 2
    if out.exists() and any(out.iterdir()):
        print("输出目录必须为空，避免把旧产物误算为续跑成功", file=sys.stderr)
        return 2
    candidates = sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"})
    unique = []
    seen_hashes = set()
    for path in candidates:
        digest = _sha(path)
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            unique.append((path, digest))
        if len(unique) == 50:
            break
    if len(unique) < 50:
        print(f"需要 50 篇内容不同的论文，实际只有 {len(unique)} 篇", file=sys.stderr)
        return 2

    from paperbrain.batch import run_batch

    old_env = {key: os.environ.get(key) for key in (
        "PAPERBRAIN_CLOUD_ALLOWED", "PAPERBRAIN_PROVIDER", "PAPERBRAIN_MEMORY_DB")}
    os.environ["PAPERBRAIN_CLOUD_ALLOWED"] = "0"
    os.environ["PAPERBRAIN_PROVIDER"] = "off"
    os.environ["PAPERBRAIN_MEMORY_DB"] = str(out / "memory.sqlite")
    items = [{"path": str(path), "paper_id": f"M6_{i + 1:02d}_{digest[:10]}"}
             for i, (path, digest) in enumerate(unique)]
    try:
        with MemorySampler() as sampler:
            first = run_batch(items, str(out), task=args.task, depth=args.depth,
                              resume=False, use_llm=False, vision=False)
            signature_before = _tree_signature(out)
            second = run_batch(items, str(out), task=args.task, depth=args.depth,
                               resume=True, use_llm=False, vision=False)
            signature_after = _tree_signature(out)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    cgroup_limit = _read_int("/sys/fs/cgroup/memory.max")
    cgroup_limit_mb = cgroup_limit / 1024 / 1024 if cgroup_limit is not None else None
    process_peak = _process_peak_mb()
    host_peak = sampler.host_peak_mb or None
    cgroup_peak = sampler.cgroup_peak_mb or None
    artifact_stable = signature_before == signature_after
    functional_ok = (first["failed"] == 0 and first["done"] == 50 and
                     second["skipped"] == 50 and artifact_stable)
    # ps RSS 和 cgroup.current 仅作诊断，不能冒充 v5 指定的 Activity Monitor/docker stats 口径。
    external_resource_evidence = args.host_peak_mb is not None and args.docker_peak_mb is not None
    resource_ok = (process_peak <= 8192 and external_resource_evidence and
                   args.host_peak_mb <= 8192 and args.docker_peak_mb <= 3072 and
                   cgroup_limit_mb is not None and cgroup_limit_mb <= 3072)
    result = {
        "papers": 50,
        "distinct_sha256": 50,
        "first_run": {"done": first["done"], "failed": first["failed"],
                      "elapsed_p95_s": round(_p95(r["elapsed"] for r in first["results"]), 3)},
        "resume_run": {"skipped": second["skipped"], "failed": second["failed"]},
        "idempotent_artifacts": artifact_stable,
        "memory_mb": {"process_peak": round(process_peak, 1),
                      "host_rss_proxy": round(host_peak, 1) if host_peak is not None else None,
                      "cgroup_sampled_peak": round(cgroup_peak, 1) if cgroup_peak is not None else None,
                      "cgroup_limit": round(cgroup_limit_mb, 1) if cgroup_limit_mb is not None else None,
                      "activity_monitor_peak": args.host_peak_mb,
                      "docker_stats_peak": args.docker_peak_mb},
        "external_resource_evidence": external_resource_evidence,
        "functional_ok": functional_ok,
        "resource_ok": resource_ok,
        "result": "PASS" if functional_ok and resource_ok else "FAIL",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
