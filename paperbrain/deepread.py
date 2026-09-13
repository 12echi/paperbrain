"""深度精读 (v5.0 A+B+C): 穿透式解读 + 提问驱动 + 全局综合。

设计目标: 不做分节泛泛摘要, 而是重建"问题→空白→洞见→方法→证据→局限"的论证链,
并用多视角提问逼出论文核心, 最后给出全局定位。全部经已接入模型完成; 无模型时
回退为"基于 Pass 纪要的结构化拼装", 保证页面始终有内容, 但会标注 llm=false。
"""
import json
import re
from pathlib import Path
from typing import Dict, List

from .llm_ops import _extract_json


_REF_TAG = re.compile(r"[\[【]\s*ref\s*[:：]\s*[^,\]】，]+?\s*[,，]\s*[^\]】]+?\s*[\]】]", re.I)
_QUESTION_PERSPECTIVES = {"方法学家", "统计学家", "实践者", "怀疑者"}


def _load(out: Path, name: str, dflt):
    try:
        return json.loads((out / name).read_text(encoding="utf-8"))
    except Exception:
        return dflt


def _sanitize_synthesis(value) -> Dict:
    """Accept only the documented global-synthesis schema from model JSON."""
    if not isinstance(value, dict):
        return {}
    cleaned: Dict = {}
    raw_map = value.get("argument_map")
    argument_map = []
    if isinstance(raw_map, list):
        for row in raw_map[:5]:
            if not isinstance(row, dict) or not isinstance(row.get("claim"), str):
                continue
            claim = row["claim"].strip()
            if not claim:
                continue
            argument_map.append({
                "claim": claim[:500],
                "evidence": row.get("evidence", "")[:800].strip()
                if isinstance(row.get("evidence", ""), str) else "",
                "limitation": row.get("limitation", "")[:500].strip()
                if isinstance(row.get("limitation", ""), str) else "",
            })
    if argument_map:
        cleaned["argument_map"] = argument_map
    raw_positioning = value.get("positioning")
    if isinstance(raw_positioning, dict):
        positioning = {}
        for field in ("improves_upon", "contradicts", "gap_left"):
            item = raw_positioning.get(field)
            if isinstance(item, str) and item.strip():
                positioning[field] = item.strip()[:500]
        if positioning:
            cleaned["positioning"] = positioning
    field_view = value.get("field_view")
    if isinstance(field_view, str) and field_view.strip():
        cleaned["field_view"] = field_view.strip()[:800]
    return cleaned


def _sanitize_questions(value, limit: int = 8) -> List[Dict[str, str]]:
    """Accept only distinct, non-empty questions from the requested perspectives."""
    if not isinstance(value, list):
        return []
    cleaned: List[Dict[str, str]] = []
    seen = set()
    for row in value:
        if not isinstance(row, dict):
            continue
        perspective = row.get("perspective")
        question = row.get("q")
        if not isinstance(perspective, str) or not isinstance(question, str):
            continue
        perspective, question = perspective.strip(), question.strip()
        if perspective not in _QUESTION_PERSPECTIVES or len(question) < 4:
            continue
        question = question[:500]
        key = re.sub(r"\s+", "", question).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append({"perspective": perspective, "q": question})
        if len(cleaned) >= limit:
            break
    return cleaned


def build_context(out_dir: str, max_chars: int = 8000) -> str:
    """兼容旧接口: 委托全局上下文构建器 (按节配额 + 关注点相关)。"""
    from .context import build_paper_context
    return build_paper_context(out_dir, budget=max_chars)


def _fallback_markdown(paper_id: str, out: Path, focus: str = "") -> str:
    passes = _load(out, "passes.json", {})
    mem = _load(out, "memory.json", {})
    lines = [f"# 深度解读 · {paper_id}", "",
             "> 离线模式（未接入模型）：以下为基于结构化纪要的骨架，接入模型后可获得穿透式论证链。", ""]
    lines += ["## 一句话主张", (passes.get("pass1", "") or "—")[:400], ""]
    lines += ["## 方法与关键设计", (passes.get("pass2", "") or "—")[:400], ""]
    lines += ["## 证据强度", (passes.get("pass3", "") or "—")[:400], ""]
    lines += ["## 局限与批判", (passes.get("pass4", "") or "—")[:400], ""]
    ents = [e.get("name") for e in (mem.get("entities") or [])]
    if ents:
        lines += ["## 关键实体", "、".join(ents), ""]
    if focus:
        lines += ["## 针对你的关注点", f"（离线模式未展开：{focus}）", ""]
    return "\n".join(lines)


def _assemble(paper_id: str, deep_md: str, qa_md: str, synth: Dict, focus: str) -> str:
    """完整深读 (详情版)。"""
    out = [f"# 完整深读 · {paper_id}", ""]
    if focus:
        out += [f"> 关注点：{focus}", ""]
    out += [deep_md.strip(), ""]
    if synth:
        am = synth.get("argument_map") or []
        if am:
            out += ["## 论证地图 (主张 → 证据 → 局限)", ""]
            for a in am:
                if isinstance(a, dict):
                    out.append(f"- **主张**：{a.get('claim','')}")
                    out.append(f"  - 证据：{a.get('evidence','')}")
                    out.append(f"  - 局限：{a.get('limitation','')}")
            out.append("")
        pos = synth.get("positioning") or {}
        if pos:
            out += ["## 全局定位", "",
                    f"- 改进自：{pos.get('improves_upon','—')}",
                    f"- 与…矛盾：{pos.get('contradicts','—')}",
                    f"- 遗留空白：{pos.get('gap_left','—')}", ""]
        if synth.get("field_view"):
            out += ["## 领域视野", str(synth["field_view"]), ""]
    if qa_md:
        out += ["## 多视角拷问", "", qa_md.strip(), ""]
    return "\n".join(out)


def _sections(md: str) -> Dict[str, str]:
    """把 markdown 按 '## 标题' 切成 {标题: 正文}。"""
    parts = re.split(r"^##\s+", md or "", flags=re.M)
    out: Dict[str, str] = {}
    for b in parts[1:]:
        head, _, body = b.partition("\n")
        out[head.strip()] = body.strip()
    return out


def _prose(text: str) -> str:
    """去掉列表标记/代码围栏/引用, 合成一段干净散文。"""
    lines = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s or s.startswith(">") or s.startswith("```"):
            continue
        s = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", s)
        if not s:
            continue
        lines.append(s)
    return " ".join(lines)


def _first_sentences(text: str, n: int = 1, cap: int = 160) -> str:
    from .text import split_sentences
    sents = [s.strip() for s in split_sentences(_prose(text)) if len(s.strip()) > 4]
    s = " ".join(sents[:n]).strip()
    if len(s) <= cap:
        return s
    refs = list(dict.fromkeys(match.group(0) for match in _REF_TAG.finditer(s)))
    shortened = s[:cap].rstrip() + "…"
    for ref in refs:
        if ref not in shortened:
            shortened += f" {ref}"
    return shortened


def _aggregate_verification(results: Dict[str, Dict]) -> Dict:
    """Fail closed across every user-visible deep-read artifact."""
    statuses = [str(result.get("status", "NEEDS_REVIEW")) for result in results.values()]
    if any(status == "DIRTY" for status in statuses):
        status = "DIRTY"
    elif any(status == "NO_CITATION" for status in statuses):
        status = "NO_CITATION" if statuses and all(s == "NO_CITATION" for s in statuses) else "DIRTY"
    elif any(status != "CLEAN" for status in statuses):
        status = "NEEDS_REVIEW"
    else:
        status = "CLEAN"
    reports = []
    for artifact, result in results.items():
        for row in result.get("report", []):
            reports.append({**row, "artifact": artifact})
    return {"status": status, "is_clean": status == "CLEAN", "report": reports,
            "artifacts": results}


def _brief(paper_id: str, full_md: str, qa_md: str, synth: Dict,
           questions: List[Dict], focus: str) -> str:
    """精读简报: 一屏读懂的成果正文 (去冗余, 详情另存完整深读)。"""
    sec = _sections(full_md)
    am = (synth or {}).get("argument_map") or []
    pos = (synth or {}).get("positioning") or {}
    out = [f"# 精读简报 · {paper_id}", ""]
    if focus:
        out += [f"> 关注点：{focus}", ""]

    claim = _first_sentences(sec.get("一句话主张", ""), 2, 240)
    out += ["## 一句话结论", claim or "—", ""]

    pts = []
    for key in ("研究问题与空白 (Gap)", "核心洞见 (Key Insight)",
                "方法与关键设计", "证据强度"):
        s = _first_sentences(sec.get(key, ""), 1, 150)
        if s:
            pts.append(f"- **{key.split(' (')[0]}**：{s}")
    if pts:
        out += ["## 核心要点"] + pts + [""]

    qs = []
    lim = _first_sentences(sec.get("局限与威胁效度", ""), 1, 150)
    if lim:
        qs.append(f"- 作者/本解读指出：{lim}")
    for a in am[:2]:
        if isinstance(a, dict) and a.get("limitation"):
            qs.append(f"- 局限：{a['limitation'][:120]}")
    if pos.get("gap_left") and pos["gap_left"] != "材料未提供":
        qs.append(f"- 遗留空白：{pos['gap_left'][:120]}")
    if qs:
        out += ["## 主要质疑"] + qs + [""]

    rep = _first_sentences(sec.get("可复现性", ""), 1, 140)
    ext = _first_sentences(sec.get("延伸设想 (If I were to extend)", ""), 1, 160)
    if rep or ext:
        out += ["## 复现与延伸"]
        if rep:
            out.append(f"- 复现：{rep}")
        if ext:
            out.append(f"- 延伸：{ext}")
        out.append("")

    if questions:
        out += ["## 关键追问"]
        for q in questions[:3]:
            if isinstance(q, dict):
                out.append(f"- **{q.get('perspective','')}**：{q.get('q','')}")
        out.append("")

    field_view = (synth or {}).get("field_view")
    if field_view:
        out += ["## 领域定位", str(field_view)[:200], ""]
    out += ["> 详细论证链与全部问答见「完整深读」。" ]
    return "\n".join(out)


def build(out_dir: str, paper_id: str, use_llm: bool = True,
          focus: str = "") -> Dict:
    """执行 A+B+C。返回 {markdown, llm, questions, synthesis}。"""
    out = Path(out_dir)
    try:
        from .llm import usage_snapshot
        usage_before = usage_snapshot()
    except Exception:
        usage_before = {"input": 0, "output": 0, "calls": 0,
                        "vision_images": 0, "by_stage": {}}
    from .context import build_paper_context
    ctx = build_paper_context(out_dir, focus=focus)
    llm_ok = False
    deep_md = qa_md = ""
    synth: Dict = {}
    questions: List[Dict] = []
    warnings: List[str] = []

    if use_llm and ctx:
        try:
            from . import llm
            deep_md = llm.deep_analyze(ctx, focus=focus, paper_id=paper_id)
            llm_ok = True
        except Exception as e:
            deep_md = ""  # 不把失败信息当正文
            warnings.append(f"深度解读生成失败：{e}")
        try:
            from . import llm
            qj = llm.gen_questions(ctx)
            arr = _extract_json(qj)
            questions = _sanitize_questions(arr)
            if isinstance(arr, list) and arr and not questions:
                warnings.append("多视角问题 JSON 不符合结构契约")
            if questions:
                qa_md = llm.answer_questions(ctx, json.dumps(questions, ensure_ascii=False), paper_id)
                llm_ok = True
        except Exception:
            questions, qa_md = [], ""
        try:
            from . import llm
            so = llm.global_synthesis(ctx, paper_id=paper_id)
            obj = _extract_json(so)
            if isinstance(obj, dict):
                synth = _sanitize_synthesis(obj)
                if synth:
                    llm_ok = True
                else:
                    warnings.append("全局综合返回的 JSON 不符合结构契约")
            else:
                warnings.append("全局综合未返回 JSON 对象")
        except Exception:
            synth = {}

    if not deep_md and not (qa_md or synth):
        full = _fallback_markdown(paper_id, out, focus)
        brief = full
    else:
        if not deep_md:
            deep_md = "## 一句话主张\n（本次深度解读生成失败或超时，已保留多视角拷问与全局综合；可重跑）"
        full = _assemble(paper_id, deep_md, qa_md, synth, focus)
        brief = _brief(paper_id, deep_md, qa_md, synth, questions, focus)
    if warnings:
        synth["_warnings"] = warnings

    (out / "deepread.md").write_text(brief, encoding="utf-8")
    (out / "deepread_full.md").write_text(full, encoding="utf-8")
    if questions:
        (out / "deepread_questions.json").write_text(
            json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        (out / "deepread_questions.json").unlink(missing_ok=True)
    if synth:
        (out / "deepread_synthesis.json").write_text(
            json.dumps(synth, ensure_ascii=False, indent=2), encoding="utf-8")
    empty_verify = {"status": "NO_CITATION", "is_clean": False, "report": []}
    verify_result = _aggregate_verification({"brief": dict(empty_verify),
                                             "full": dict(empty_verify)})
    try:
        from .verifier import CitationVerifierV5
        from .retrieval import make_scorer
        state = _load(out, "state.json", {})
        gt = state.get("ground_truth") or {}
        if gt:
            scorer = make_scorer(gt)
            from .calibration import load_record, enforce_review
            calibration = load_record("tfidf-corpus")
            pass_th, review_th = 0.82, 0.35
            if calibration["verified"]:
                pass_th, review_th = calibration["pass_th"], calibration["review_th"]
            if llm_ok:
                try:
                    from . import llm_ops
                    if llm_ops.enabled():
                        scorer, pass_th, review_th = llm_ops.make_nli_scorer(), 0.82, 0.35
                        calibration = load_record(
                            "nli", Path(__file__).resolve().parent.parent / "thresholds_nli.json")
                        if calibration["verified"]:
                            pass_th, review_th = calibration["pass_th"], calibration["review_th"]
                except Exception:
                    pass
            verifier = CitationVerifierV5(
                gt, state.get("fig_index") or {}, semantic_scorer=scorer,
                threshold=pass_th, review_threshold=review_th,
                section_chunks=state.get("section_chunks") or {}
            )
            artifact_results = {
                "brief": enforce_review(verifier.verify_draft(brief), calibration),
                "full": enforce_review(verifier.verify_draft(full), calibration),
            }
            verify_result = _aggregate_verification(artifact_results)
    except Exception as e:
        verify_result = {"status": "NEEDS_REVIEW", "is_clean": False, "report": [],
                         "error": str(e)[:160]}
    (out / "deepread_verify.json").write_text(
        json.dumps(verify_result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 深读是同一篇论文预算的一部分；把本阶段增量合并回 state，供后续生成或标准深度报告使用。
    budget_ok = True
    budget_tokens: Dict[str, int] = {}
    combined_usage: Dict = {}
    try:
        from .llm import usage_snapshot
        from .budget import check_budget
        usage_after = usage_snapshot()
        state_path = out / "state.json"
        state = _load(out, "state.json", {})
        prior = state.get("llm_usage") or {}
        delta_input = max(0, int(usage_after.get("input", 0)) - int(usage_before.get("input", 0)))
        delta_output = max(0, int(usage_after.get("output", 0)) - int(usage_before.get("output", 0)))
        delta_calls = max(0, int(usage_after.get("calls", 0)) - int(usage_before.get("calls", 0)))
        combined_usage = {"input": int(prior.get("input", 0)) + delta_input,
                          "output": int(prior.get("output", 0)) + delta_output,
                          "calls": int(prior.get("calls", 0)) + delta_calls,
                          "vision_images": int(prior.get("vision_images", 0)) + max(
                              0, int(usage_after.get("vision_images", 0)) -
                              int(usage_before.get("vision_images", 0))),
                          "by_stage": dict(prior.get("by_stage") or {})}
        before_stage = usage_before.get("by_stage") or {}
        for key, value in (usage_after.get("by_stage") or {}).items():
            delta = max(0, int(value) - int(before_stage.get(key, 0)))
            combined_usage["by_stage"][key] = combined_usage["by_stage"].get(key, 0) + delta
        combined_usage["total"] = combined_usage["input"] + combined_usage["output"]
        budget_tokens = ({k: int(v) for k, v in combined_usage["by_stage"].items() if int(v) > 0}
                         if combined_usage["total"] else dict(state.get("budget_tokens") or
                                                               state.get("pass_tokens") or {}))
        budget = check_budget(
            budget_tokens, num_images=int(combined_usage.get("vision_images",
                                                             state.get("vision_attempts", 0))))
        budget_ok = budget.ok
        if state:
            state["llm_usage"] = combined_usage
            state["budget_tokens"] = budget_tokens
            state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    except Exception:
        budget_ok = False
    return {"markdown": brief, "full_markdown": full, "llm": llm_ok,
            "questions": questions, "synthesis": synth,
            "verify": verify_result["status"], "is_clean": verify_result["is_clean"],
            "budget_ok": budget_ok, "budget_tokens": budget_tokens,
            "llm_usage": combined_usage,
            "files": ["deepread.md", "deepread_full.md"]}
