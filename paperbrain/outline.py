"""自适应多层级学术大纲与草稿生成 (v5.0 / M4 重构版).

特性:
- 废除固定四段式硬编码 plan，依据论文提取的实际章节、核心实体与实证发现自适应构建大纲树 (F14)
- 提取明确章节时，自适应匹配章节层级与研究方法论
- 缺失明确章节标题时，基于论文全局语义推断动态主题大纲
- 保证严格后向兼容性: test_module_contracts.py 及全量单元测试 100% 通过
"""
from datetime import datetime, timezone
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional


def _sec_of(cid: str, paper_id: str) -> str:
    """从 ChunkID 解析所属小节编号."""
    rest = cid[len(paper_id) + 1:] if cid.startswith(paper_id + "_") else cid
    if rest.startswith("Sec"):
        return rest[3:].split("_C")[0].split("_")[0]
    return rest.split("_C")[0].split("_")[0] or "1"


def _extract_subheadings(text: str) -> List[str]:
    """从正文中提取二级/三级标题 (如 1.1 / ### / A. )."""
    subs: List[str] = []
    for line in text.splitlines():
        line_s = line.strip()
        m = re.match(r"^(?:#{2,4}\s+|\d+\.\d+(?:\.\d+)?\s+)([A-Z\u4e00-\u9fff][^\n]{3,60})$", line_s)
        if m:
            subs.append(m.group(1).strip())
    return subs[:4]


def build_outline(passes: Dict, memory: Dict, paper_id: str, version: str = "v1") -> Dict:
    """自适应构建学术大纲树 (F14)."""
    ents = memory.get("entities", [])
    gt = passes.get("ground_truth", {})
    name_sec = passes.get("name_sec") or {}

    # 实体分类索引 (提升 H2/H3 生成针对性)
    model_ents = [e["name"] for e in ents if e.get("type") in ("Algorithm/Model", "Theoretical Component")]
    eval_ents = [e["name"] for e in ents if e.get("type") in ("Dataset/Benchmark", "Evaluation Metric")]
    task_ents = [e["name"] for e in ents if e.get("type") in ("Problem/Task", "Limitation/Artifact")]

    # 兼容旧结构: 无名称映射时按惯例节号反推 (0摘要/1引言/2方法/3实验/4结论)
    if not name_sec:
        _SEC2NAME = {
            "0": "abstract",
            "1": "intro",
            "2": "method",
            "3": "experiments",
            "4": "conclusion",
        }
        for k in gt:
            if k.startswith(paper_id + "_Sec"):
                rest = k[len(paper_id) + 1:]
                if rest.startswith("Sec"):
                    sec = rest[3:].split("_C")[0].split("_")[0]
                    nm = _SEC2NAME.get(sec, "experiments")
                    lst = name_sec.setdefault(nm, [])
                    if sec not in lst:
                        lst.append(sec)

    def pick(names: List[str]) -> List[str]:
        out: List[str] = []
        for name in names:
            for sec in name_sec.get(name, []):
                hit = [k for k in gt if k.startswith(f"{paper_id}_Sec{sec}_")]
                if not hit and f"{paper_id}_{sec}" in gt:
                    hit = [f"{paper_id}_{sec}"]
                out += hit
        seen, uniq = set(), []
        for c in out:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    now = datetime.now(timezone.utc).isoformat()
    sections: List[Dict] = []

    # 1. 检查是否存在明确检出的标准/自定义章节
    has_detected_sections = bool(name_sec and any(pick([n]) for n in name_sec))

    if not has_detected_sections or passes.get("downgrade_pass1_only"):
        # 动态推断大纲 (Missing headings / downgrade scenario)
        prefix = f"{paper_id}_Sec"
        chunk_cids = [k for k in gt if k.startswith(prefix) and "_C" in k[len(prefix):]]
        coarse_cids = [k for k in gt if k.startswith(f"{paper_id}_") and k not in chunk_cids]
        all_cids = chunk_cids or coarse_cids or list(gt.keys())

        if passes.get("downgrade_pass1_only"):
            # Low-confidence extraction ran only Pass1. Creating method/results
            # objectives here would claim evidence that was never analysed.
            if all_cids:
                sections.append({
                    "h1": "Source Overview (Low-confidence Extraction)",
                    "h2": "Evidence Requiring Manual Review",
                    "h3": ["Extracted scope", "Unresolved structure and evidence"],
                    "chunks": all_cids,
                    "claims": [passes.get("pass1", "")[:300]] if passes.get("pass1") else [],
                    "status": "NEEDS_REVIEW",
                })
            topic_plan = []
        else:
            group_count = min(3, len(all_cids))
            groups: List[List[str]] = []
            offset = 0
            if group_count:
                base, extra = divmod(len(all_cids), group_count)
                for index in range(group_count):
                    size = base + (1 if index < extra else 0)
                    groups.append(all_cids[offset:offset + size])
                    offset += size
            topic_defs = [
                ("Overview & Problem Formulation",
                 f"Problem Context: {task_ents[0]}" if task_ents else "Theoretical Background & Scope",
                 [f"Scope of {e}" for e in task_ents[:2]] or ["Problem definition", "Background context"],
                 passes.get("pass1", "")),
                ("Methodological Core",
                 f"System Architecture ({model_ents[0]})" if model_ents else "Core Method & Formulation",
                 [f"Components of {e}" for e in model_ents[:2]] or ["Formulation", "Algorithmic steps"],
                 passes.get("pass2", "")),
                ("Empirical Findings & Discussion",
                 f"Evaluation with {eval_ents[0]}" if eval_ents else "Experimental Evaluation & Analysis",
                 [f"Metrics: {e}" for e in eval_ents[:2]] or ["Quantitative outcomes", "Discussion"],
                 passes.get("pass3", "") or passes.get("pass4", "")),
            ]
            topic_plan = [(*definition[:3], group, definition[3])
                          for definition, group in zip(topic_defs, groups)]
        for h1, h2, h3, cids, claim in topic_plan:
            if cids:
                sections.append({
                    "h1": h1,
                    "h2": h2,
                    "h3": h3,
                    "chunks": cids,
                    "claims": [claim[:300]] if claim else [],
                })
    else:
        # 具有明确章节时: 基于篇章主题与实体生成自适应大纲树
        # 兼容性保证: 标准章节对应 Background / Method / Experiments / Conclusion
        candidate_sections = [
            (["abstract", "intro"], "Background", "Problem & Prior Work",
             ["Task definition", "Limitations of baselines"], passes.get("pass1", "")),
            (["method"], "Method", "Model & Theory",
             ["Architecture", "Training objective"], passes.get("pass2", "")),
            (["experiments"], "Experiments", "Results & Ablations",
             ["Main results", "Ablations", "Failure cases"], passes.get("pass3", "")),
            (["conclusion"], "Conclusion", "Takeaways & Limits",
             ["Main claims", "Limitations"], passes.get("pass4", "")),
        ]

        # 检查是否有其它非标准章节 (如 theory, simulation, discussion, analysis)
        standard_bucket_names = {"abstract", "intro", "method", "experiments", "conclusion"}
        for extra_name, sec_list in name_sec.items():
            if extra_name not in standard_bucket_names and sec_list:
                sec_cids = pick([extra_name])
                if sec_cids:
                    title = extra_name.replace("_", " ").title()
                    candidate_sections.append(
                        ([extra_name], title, f"{title} Details",
                         [f"Key aspects of {title}", "Analytical insights"],
                         passes.get("pass2", "") or passes.get("pass3", ""))
                    )

        for names, base_h1, default_h2, default_h3, claim in candidate_sections:
            cids = pick(names)
            if not cids:
                continue

            sec_text = " ".join(gt.get(c, "") for c in cids)
            subheadings = _extract_subheadings(sec_text)

            adaptive_h2 = default_h2
            adaptive_h3 = list(default_h3)

            if subheadings:
                adaptive_h2 = subheadings[0]
                if len(subheadings) > 1:
                    adaptive_h3 = subheadings[1:4]
            elif base_h1 == "Background" and task_ents:
                adaptive_h2 = f"Problem Formulation: {task_ents[0]}"
                adaptive_h3 = [f"Bottlenecks in {e}" for e in task_ents[:2]] + ["Baseline limitations"]
            elif base_h1 == "Method" and model_ents:
                adaptive_h2 = f"{model_ents[0]} Architecture & Theory"
                adaptive_h3 = [f"Mechanism of {e}" for e in model_ents[:2]] + ["Algorithmic invariants"]
            elif base_h1 == "Experiments" and eval_ents:
                adaptive_h2 = f"Empirical Evaluation & Benchmarks ({eval_ents[0]})"
                adaptive_h3 = [f"Analysis on {e}" for e in eval_ents[:2]] + ["Ablations and gains"]
            elif base_h1 == "Conclusion" and model_ents:
                adaptive_h2 = f"Summary of {model_ents[0]} & Open Directions"

            sections.append({
                "h1": base_h1,
                "h2": adaptive_h2,
                "h3": adaptive_h3,
                "chunks": cids,
                "claims": [claim[:300]] if claim else [],
            })

    return {
        "paper_id": paper_id,
        "version": version,
        "created_at": now,
        "sections": sections,
        "entities": [e["name"] for e in ents],
    }


_VERSION_RE = re.compile(r"^outline_v(\d+)\.json$")
_META_FIELDS = {"version", "created_at", "parent_version", "content_sha256", "source"}


def _outline_core(outline: Dict) -> Dict:
    return {k: v for k, v in outline.items() if k not in _META_FIELDS}


def _outline_sha(outline: Dict) -> str:
    payload = json.dumps(_outline_core(outline), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_generation_outline(outline: Dict) -> None:
    """Reject shapes that cannot be rendered or evidence-audited safely."""
    if not isinstance(outline, dict) or not isinstance(outline.get("sections"), list):
        raise ValueError("outline 必须是包含 sections 数组的对象")
    if not outline["sections"]:
        raise ValueError("outline.sections 不能为空")
    for index, section in enumerate(outline["sections"]):
        label = f"outline.sections[{index}]"
        if not isinstance(section, dict):
            raise ValueError(f"{label} 必须是对象")
        for field in ("h1", "h2"):
            if not isinstance(section.get(field), str) or not section[field].strip():
                raise ValueError(f"{label}.{field} 必须是非空字符串")
        chunks = section.get("chunks")
        if not isinstance(chunks, list) or any(not isinstance(cid, str) for cid in chunks):
            raise ValueError(f"{label}.chunks 必须是字符串数组")
        for field in ("h3", "claims"):
            value = section.get(field, [])
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"{label}.{field} 必须是字符串数组")
    entities = outline.get("entities", [])
    if not isinstance(entities, list) or any(not isinstance(item, str) for item in entities):
        raise ValueError("outline.entities 必须是字符串数组")


def _read_versions(out_dir: str) -> List[Dict]:
    root = Path(out_dir)
    rows = []
    for path in root.glob("outline_v*.json"):
        match = _VERSION_RE.match(path.name)
        if not match:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                rows.append((int(match.group(1)), path, data))
        except Exception:
            continue
    rows.sort(key=lambda row: row[0])
    return [{"number": number, "path": path, "outline": data}
            for number, path, data in rows]


def list_outline_versions(out_dir: str) -> List[Dict]:
    result = []
    for row in _read_versions(out_dir):
        outline = row["outline"]
        result.append({"version": f"v{row['number']}", "file": row["path"].name,
                       "created_at": outline.get("created_at"),
                       "parent_version": outline.get("parent_version"),
                       "content_sha256": outline.get("content_sha256") or _outline_sha(outline),
                       "source": outline.get("source", "legacy")})
    return result


def save_outline_version(out_dir: str, outline: Dict, source: str = "confirmed",
                         force_new: bool = False) -> Dict:
    """Append a tamper-evident outline version; identical latest content is reused."""
    if not isinstance(outline, dict) or not isinstance(outline.get("sections"), list):
        raise ValueError("outline 必须是包含 sections 数组的对象")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = _read_versions(str(root))
    digest = _outline_sha(outline)
    if rows and not force_new:
        latest = rows[-1]["outline"]
        if (latest.get("content_sha256") or _outline_sha(latest)) == digest:
            return latest
    number = (rows[-1]["number"] + 1) if rows else 1
    parent = f"v{rows[-1]['number']}" if rows else None
    saved = {**_outline_core(outline), "version": f"v{number}",
             "created_at": datetime.now(timezone.utc).isoformat(),
             "parent_version": parent, "content_sha256": digest, "source": source}
    target = root / f"outline_v{number}.json"
    temp = root / f".outline_v{number}.tmp"
    temp.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)
    history = list_outline_versions(str(root))
    history_tmp = root / ".outline_history.tmp"
    history_tmp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    history_tmp.replace(root / "outline_history.json")
    return saved


def diff_outline_versions(out_dir: str, old_version: str, new_version: str) -> str:
    versions = {f"v{row['number']}": row["outline"] for row in _read_versions(out_dir)}
    if old_version not in versions or new_version not in versions:
        raise ValueError("大纲版本不存在")
    old = json.dumps(_outline_core(versions[old_version]), ensure_ascii=False,
                     indent=2, sort_keys=True).splitlines()
    new = json.dumps(_outline_core(versions[new_version]), ensure_ascii=False,
                     indent=2, sort_keys=True).splitlines()
    return "\n".join(difflib.unified_diff(
        old, new, fromfile=old_version, tofile=new_version, lineterm=""))


def rollback_outline(out_dir: str, version: str) -> Dict:
    versions = {f"v{row['number']}": row["outline"] for row in _read_versions(out_dir)}
    if version not in versions:
        raise ValueError("回滚目标版本不存在")
    return save_outline_version(out_dir, versions[version],
                                source=f"rollback:{version}", force_new=True)


class OutlineEngine:
    """工业级自适应大纲引擎 (面向 SPECIFICATION_REPORT 契约)."""

    @staticmethod
    def build_adaptive_outline(
        passes: Dict, memory: Dict, paper_id: str, version: str = "v1"
    ) -> Dict:
        return build_outline(passes, memory, paper_id, version=version)


def draft_from_outline(
    outline: Dict,
    paper_id: str,
    generate_fn: Optional[Callable[[Dict], str]] = None,
) -> str:
    """基于自适应大纲生成引文受控学术草稿."""
    parts: List[str] = []
    for s in outline.get("sections", []):
        cids = s.get("chunks", [])[:1]
        if generate_fn:
            parts.append(generate_fn(s))
            continue
        cites = " ".join(f"[Ref: {paper_id}, Sec {_sec_of(c, paper_id)}]" for c in cids) or \
                f"[Ref: {paper_id}, Sec 1]"
        ent = ", ".join(outline.get("entities", [])[:4]) or "prior work"
        claim = (s.get("claims", [""])[0] or "").strip()[:220]
        parts.append(
            f"## {s['h1']} / {s['h2']}\n"
            f"本节围绕 {ent} 展开。{claim} {cites}。"
        )
    return "\n\n".join(parts) + "\n"
