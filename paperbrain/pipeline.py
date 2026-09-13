"""端到端 Pipeline (v5.0 最小可跑闭环).

执行顺序 (失败即停, 不伪造):
  preflight -> split_sections -> run_passes(预算门禁) -> build_memory(SQLite)
  -> build_outline(版本化) -> draft_from_outline -> CitationVerifierV5 -> 落盘

落盘 out/:
  sections.json / passes.json / memory.json / outline_<ver>.json / draft.md
  / verify.json / ledger.csv / report.md
"""
import csv
import json
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .preflight import preflight
from .sections import split_sections
from .passes import run_passes
from .memory import build_memory
from .outline import build_outline, draft_from_outline, _sec_of
from .verifier import CitationVerifierV5
from .budget import check_budget, estimate_vision_tokens
from .retrieval import make_scorer
from .formulas import check_formulas, mark_text
from . import config

# 分步计时钩子: 线程局部存储, 避免多请求并发相互覆盖或清空
_LOCAL = threading.local()


def set_trace(fn: Optional[Callable[[str], None]]):
    _LOCAL.trace = fn


def get_trace() -> Optional[Callable[[str], None]]:
    return getattr(_LOCAL, "trace", None)


def _mark(name: str, trace_fn: Optional[Callable[[str], None]] = None):
    fn = trace_fn or get_trace()
    if fn is not None:
        try:
            fn(name)
        except Exception:
            pass


def _ledger_row(paper_id: str, sha: str, pt: Dict, vision: int,
                cost_note: str = "offline-mock-0", usage: Optional[Dict] = None,
                budget_verified: bool = True) -> Dict:
    total = sum(pt.values()) + vision
    usage = usage or {}
    return {"paper_id": paper_id, "pdf_sha": sha[:12], "text_tokens": sum(pt.values()),
            "vision_tokens": vision, "total": total, "input_tokens": usage.get("input", 0),
            "output_tokens": usage.get("output", 0), "llm_calls": usage.get("calls", 0),
            "budget_verified": budget_verified, "cost": cost_note}


def _budget_tokens(fallback: Dict[str, int], usage: Optional[Dict]) -> Dict[str, int]:
    """有模型台账时以完整调用输入+输出为准；纯离线时沿用产出估值。"""
    if usage and int(usage.get("total", 0)) > 0:
        return {str(k): int(v) for k, v in (usage.get("by_stage") or {}).items() if int(v) > 0}
    return {str(k): int(v) for k, v in (fallback or {}).items()}


def run_outline(input_path: str, paper_id: str, out_dir: str,
                use_llm: bool = True, vision: bool = False,
                trace_fn: Optional[Callable[[str], None]] = None,
                m1_pred_path: Optional[str] = None) -> Dict:
    """第一阶段: 预处理->分章->Pass->记忆->大纲 (存 state.json, 不写草稿).
    用户确认/编辑 outline 后调 generate_from_outline (STORM 两步走)."""
    if trace_fn is not None:
        set_trace(trace_fn)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _lm = None
    _c0 = 0
    try:
        import paperbrain.llm as _lm
        from .llm import reset_occli_session, reset_usage
        reset_occli_session()  # 线程内独立会话, 防跨篇上下文串扰
        reset_usage()
        _c0 = _lm.CALL_COUNT
    except Exception:
        pass

    pf = preflight(input_path, paper_id)
    _mark("preflight", trace_fn=trace_fn)
    warnings = list(pf.warnings)
    text, route = pf.text or "", pf.route

    # 扫描版自动 OCR: deep_ocr 全量, hybrid 低置信页按页局部补偿; 失败则降级 Pass1 (不伪造)
    if str(input_path).lower().endswith(".pdf"):
        if route == "deep_ocr":
            try:
                from .ocr import ocr_pdf
                oc = ocr_pdf(str(input_path), workdir=str(out))
                if oc.get("ok"):
                    text = "\n".join(p["text"] for p in oc["pages"])
                    route = "ocr_recovered"
                    warnings.append(f"OCR 恢复 {len(oc['pages'])} 页")
                else:
                    warnings.append(f"OCR 失败 ({oc.get('reason','')}), 降级只跑 Pass1")
                    route = "needs_ocr_tool"
            except Exception as e:
                warnings.append(f"OCR 异常 {e}, 降级只跑 Pass1")
                route = "needs_ocr_tool"
        elif route == "hybrid":
            # 解除字数硬互斥: 逐页检查置信度与文本密度, 仅对低文本率页面进行按页局部 OCR 补全
            try:
                import fitz
                from .ocr import has_ocr, ocr_pdf
                page_texts = []
                with fitz.open(str(input_path)) as _doc:
                    for pg in _doc:
                        ptxt = pg.get_text(sort=True) or ""
                        page_texts.append(ptxt)
                # 与 preflight 共用同一页级 <70% 判定，禁止另一套“少于100字”启发式漂移。
                low_conf_pages = list(pf.low_conf_pages)
                # Missing table lines are handled by caption-anchored local text parsing;
                # OCR only pages whose extractable text itself is below 70%.
                ocr_pages = [pno for pno in low_conf_pages
                             if pno < len(pf.page_rates) and pf.page_rates[pno] < 0.70]

                if ocr_pages and has_ocr():
                    oc = ocr_pdf(str(input_path), workdir=str(out), pages=ocr_pages)
                    recovered = {int(page.get("page", -1)): page
                                 for page in oc.get("pages", [])}
                    if oc.get("ok") and all(pno in recovered for pno in ocr_pages):
                        for pno in ocr_pages:
                            ocr_txt = recovered[pno].get("text", "")
                            if len(ocr_txt.strip()) > len(page_texts[pno].strip()):
                                page_texts[pno] = ocr_txt
                        text = "\n".join(page_texts)
                        warnings.append(f"Hybrid OCR 补偿 {len(ocr_pages)} 个低文本率页")
                    elif not oc.get("ok"):
                        warnings.append("Hybrid OCR 补偿跳过 (OCR 质量未达标)")
                elif ocr_pages and not has_ocr():
                    warnings.append("缺 tesseract, Hybrid 低置信页未做 OCR 补偿")
            except Exception as e:
                warnings.append(f"Hybrid OCR 检查跳过 ({e})")

    fchk = check_formulas(text)
    extracted_text = text
    _mark("formulas", trace_fn=trace_fn)
    if fchk["dirty"]:
        warnings.append(f"公式{len(fchk['formulas'])}个, 其中{fchk['dirty']}个未通过校验已打标")
        text = mark_text(text, fchk)
    sec_meta: Dict = {}
    secs = split_sections(text, paper_id, sec_meta)
    _mark("sections", trace_fn=trace_fn)
    # 全模型模式: 分章也交给模型 (失败则保留规则切分, 不崩)
    all_model = False
    try:
        from . import llm_ops
        all_model = use_llm and llm_ops.enabled()
        if all_model:
            seg = llm_ops.segment(text, paper_id)
            if seg:
                secs = seg
                sec_meta["model_segmented"] = True
                warnings.append("分章由模型完成")
    except Exception as e:
        warnings.append(f"模型分章跳过 ({e})")
    if sec_meta.get("dropped_tail"):
        warnings.append("已截断文末元数据节 (References/Acknowledgements 等)")
    # 图表本地抽取始终执行；只有 VL 描述需要显式云端授权。
    tables_info: Dict = {"tables": 0, "cells": 0}
    table_records: List[Dict] = []
    figure_records: List[Dict] = []
    if str(input_path).lower().endswith(".pdf"):
        try:
            from .vision import extract_tables
            table_records = extract_tables(str(input_path))
            tables_info["tables"] = len(table_records)
            tables_info["cells"] = sum(
                int(t.get("rows", 0)) * int(t.get("cols", 0)) for t in table_records)
            if tables_info["tables"]:
                warnings.append(f"检测到表格{tables_info['tables']}张/约{tables_info['cells']}单元格")
        except Exception as e:
            warnings.append(f"表格清点跳过 ({e})")
        try:
            from .vision import extract_images
            figure_records = extract_images(
                str(input_path), str(out), max_images=config.max_images())
            if figure_records:
                warnings.append(f"本地抽取图表 {len(figure_records)} 张（已剥离元数据）")
        except Exception as e:
            warnings.append(f"图表抽取跳过 ({e})")
    _mark("tables", trace_fn=trace_fn)
    # VL 读图 (默认关; PAPERBRAIN_VISION=1 或 vision=True 开启, 单图约1~2分钟)
    figure_notes: List[Dict] = []
    if (vision or config.vision_enabled()) and \
            str(input_path).lower().endswith(".pdf"):
        if not config.cloud_allowed():
            warnings.append("VL 读图已请求但云端未授权，未发送任何图像")
        else:
            try:
                from .vision import describe_figure
                for fig in figure_records:
                    try:
                        an = describe_figure(fig["image"], hint=fig.get("caption", ""))
                        fig.update(an)
                        figure_notes.append(fig)
                    except Exception as e:
                        warnings.append(f"配图跳过 {Path(fig['image']).name} ({e})")
                if figure_notes:
                    warnings.append(f"VL 解读配图 {len(figure_notes)} 张")
            except Exception as e:
                warnings.append(f"VL 读图未启用 ({e})")
    if figure_notes:
        note_txt = "\n".join(f"[Fig p{f['page']}] {f['analysis']}" for f in figure_notes)
        for s in secs:
            if s.get("name") == "experiments":
                s["text"] += "\n" + note_txt
                s["chunks"].append({"chunk_id": f"{paper_id}_Sec{s.get('sec','3')}_C{len(s['chunks'])+1:03d}",
                                    "text": note_txt})
                break
    if route == "needs_ocr_tool":
        secs = [{"name": "full", "sec": "0", "text": "", "confidence": 0.0,
                 "downgrade_pass1_only": True, "chunks": []}]

    llm_used = bool(figure_notes)

    def _summ(kind: str, text: str) -> str:
        nonlocal llm_used
        if use_llm:
            try:
                from .llm import summarize
                # 先规则预digest到 6000 字内, 再一次 LLM 调用 (每 Pass 仅 1 次, 省额度)
                from .passes import _even_reduce
                # 每个窗口都先参与本地归约，再只调用模型一次；避免“单次调用快”但静默只读开头。
                short = (_even_reduce([text[i:i + 3000] for i in range(0, len(text), 3000)], 6000)
                         if len(text) > 6000 else text)
                s = summarize(kind, short)
                llm_used = True
                return s
            except Exception:
                pass
        from .passes import extractive
        limits = {"pass1": 7500, "pass2": 10500, "pass3": 12000, "pass4": 6000}
        return extractive(text, limits.get(kind, 7500))

    passes = run_passes(secs, paper_id, summarize_fn=lambda k, t: _summ(k, t),
                        figures=figure_records, tables=table_records)
    _mark("passes", trace_fn=trace_fn)
    # 全模型模式: 图谱抽取交给模型 (空结果回退规则种子)
    _extract_fn = None
    if all_model:
        def _extract_fn(full: str):
            from . import llm_ops
            ents, rels = llm_ops.extract_graph(full)
            if not ents:
                from .memory import rule_extract
                return rule_extract(full)
            return ents, rels
    mem = build_memory(secs, paper_id, str(out / "paperbrain.db"), extract_fn=_extract_fn)
    if all_model:
        try:
            graph_status = llm_ops.last_graph_status()
            mem.setdefault("stats", {})["model_graph"] = graph_status
            if graph_status.get("status") == "MANUAL_REVIEW":
                queue_item = {"paper_id": paper_id, **graph_status}
                (out / "graph_review_queue.json").write_text(
                    json.dumps([queue_item], ensure_ascii=False, indent=2), encoding="utf-8")
                warnings.append("模型图谱两次 Schema 拒收，已规则回退并进入人工复核队列")
        except Exception as e:
            warnings.append(f"图谱人工队列记录失败 ({e})")
    _mark("memory", trace_fn=trace_fn)
    try:
        from .llm import usage_snapshot
        llm_usage = usage_snapshot()
    except Exception:
        llm_usage = {"input": 0, "output": 0, "total": 0, "calls": 0,
                     "vision_images": 0, "by_stage": {}}
    budget_tokens = _budget_tokens(passes["pass_tokens"], llm_usage)
    vision_attempts = int(llm_usage.get("vision_images", 0))
    outline_budget = check_budget(budget_tokens, num_images=vision_attempts)
    if not outline_budget.ok:
        raise RuntimeError(f"预算超限拒绝执行: {outline_budget.reasons}")
    outline = build_outline(passes, mem, paper_id, version="v1")
    from .outline import save_outline_version
    outline = save_outline_version(str(out), outline, source="generated")
    _mark("outline", trace_fn=trace_fn)
    slim_secs = [{**s, "text": s.get("text", "")[:3000],
                  "chunks": [{**c, "text": c.get("text", "")[:1500]} for c in s.get("chunks", [])]}
                 for s in secs]
    (out / "sections.json").write_text(json.dumps(slim_secs, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "passes.json").write_text(json.dumps({k: (v[:2000] if isinstance(v, str) else v) for k, v in passes.items() if k != "ground_truth"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "memory.json").write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "formulas.json").write_text(json.dumps(fchk, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "figures.json").write_text(json.dumps(
        [{"image": Path(f["image"]).name, "page": f["page"], "pixels": f["pixels"],
          "caption": f.get("caption", ""), "metadata_stripped": f.get("metadata_stripped", False),
          "analysis": f.get("analysis", "")} for f in figure_records],
        ensure_ascii=False, indent=2), encoding="utf-8")
    state = {"paper_id": paper_id, "sha": pf.sha, "route": route, "warnings": warnings,
             "fchk": fchk, "llm_used_outline": llm_used,
             "llm_calls_outline": ((_lm.CALL_COUNT - _c0) if _lm else 0),
             "tables": tables_info,
             "n_figures": len(figure_notes),
             "vision_attempts": vision_attempts,
             "n_extracted_figures": len(figure_records),
             "figures": [{"image": Path(f["image"]).name, "page": f["page"],
                          "analysis": f.get("analysis", "")[:500]} for f in figure_notes],
             "coverage": passes.get("coverage", 1.0), "total_chars": passes.get("total_chars", 0),
             "section_chunks": {f"{paper_id}_{s.get('sec', '0')}":
                                [c["text"] for c in s.get("chunks", [])] for s in secs},
             "downgrade": passes["downgrade_pass1_only"], "pass_tokens": passes["pass_tokens"],
             "budget_tokens": budget_tokens, "llm_usage": llm_usage,
             "ground_truth": {k: v[:1500] for k, v in passes["ground_truth"].items()},
             "fig_index": passes["fig_index"], "mem_saved": mem["saved"]}
    (out / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    if m1_pred_path:
        # Explicit evaluation output only: normal runs do not duplicate full extracted text.
        by_id = passes.get("fig_index", {}).get("_by_id", {})
        captions = [
            {"figure_id": item_id, "caption": item.get("caption", ""),
             "bound_contexts": list(item.get("bound_contexts", []))}
            for item_id, item in by_id.items()
            if item.get("type") == "figure" and item.get("caption")
        ]
        prediction = {
            "id": paper_id,
            "source_sha256": pf.sha,
            "text": extracted_text,
            "formulas": fchk.get("formulas", []),
            "tables": [
                {"table_id": t.get("table_id", ""), "cells": t.get("cells", [])}
                for t in table_records
            ],
            "captions": captions,
        }
        pred_path = Path(m1_pred_path).expanduser().resolve()
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return {"out": str(out), "outline": outline, "route": route,
            "downgrade": passes["downgrade_pass1_only"], "llm_used": llm_used,
            "warnings": warnings, "pass_tokens": budget_tokens}


def generate_from_outline(out_dir: str, paper_id: str, outline: Optional[Dict] = None,
                          semantic_scorer: Optional[Callable[[str, str], float]] = None,
                          scorer_threshold: float = 0.82, use_llm: bool = True,
                          draft_override: Optional[str] = None,
                          trace_fn: Optional[Callable[[str], None]] = None) -> Dict:
    """第二阶段: 用确认后大纲生成草稿->验真->台账->报告.
    draft_override: 外部模型草稿 (如 Muse 会话精读) 直接验真, 记 model=muse-session.
    无外部草稿时才用内置 RAG 回退."""
    if trace_fn is not None:
        set_trace(trace_fn)
    out = Path(out_dir)
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    outline = outline or json.loads((out / "outline_v1.json").read_text(encoding="utf-8"))
    from .outline import save_outline_version, validate_generation_outline
    validate_generation_outline(outline)
    outline = save_outline_version(str(out), outline, source="confirmed")
    gt = state["ground_truth"]
    outline_paper_mismatch = bool(outline.get("paper_id") and
                                  str(outline.get("paper_id")) != str(paper_id))
    heading_keys = [
        (str(sec.get("h1", "")).strip().casefold(),
         str(sec.get("h2", "")).strip().casefold())
        for sec in outline.get("sections", []) if isinstance(sec, dict)
    ]
    duplicate_outline_headings = len(heading_keys) != len(set(heading_keys))
    try:
        from .llm import reset_usage
        reset_usage()
    except Exception:
        pass

    from .retrieval import build_index, retrieve as _retrieve
    from .text import split_sentences as _ssplit
    llm_used = bool(state.get("llm_used_outline"))
    llm_calls = int(state.get("llm_calls_outline", 0))
    review_sections_expected = sum(sec.get("generation_mode") == "review"
                                   for sec in outline.get("sections", []))
    review_model_sections = 0
    missing_evidence_sections = []
    invalid_chunk_references = []

    def _gen(sec: Dict) -> str:
        nonlocal llm_used, llm_calls, review_model_sections
        raw_cids = sec.get("chunks", [])
        if not isinstance(raw_cids, list):
            invalid_chunk_references.append("<chunks field is not a list>")
            cids = []
        else:
            cids = [cid for cid in raw_cids if isinstance(cid, str)]
            invalid_chunk_references.extend(
                str(cid)[:120] for cid in raw_cids
                if not isinstance(cid, str) or cid not in gt)
        ent = ", ".join(outline.get("entities", [])[:4]) or "prior work"
        sub = {c: gt[c] for c in cids if c in gt}
        if not sub:
            missing_evidence_sections.append(f"{sec.get('h1', '')} / {sec.get('h2', '')}".strip())
            return f"## {sec['h1']} / {sec['h2']}\n⚠️ [缺少可引用原文，本节跳过]"
        claims = sec.get("claims") or []
        primary_claim = claims[0] if claims else ""
        q = (primary_claim or f"{sec['h1']} {sec['h2']}") + " " + \
            " ".join(outline.get("entities", [])[:4])
        ranked = _retrieve(q, build_index(sub), top_k=min(3, len(sub)))
        evidence = "\n".join(f"[Sec {_sec_of(cid, paper_id)}] {sub[cid]}" for cid, _ in ranked)
        primary_cid = ranked[0][0] if ranked else next(iter(sub))
        qsec = _sec_of(primary_cid, paper_id)
        cites = f"[Ref: {paper_id}, Sec {qsec}]"
        body = ""
        if use_llm:
            try:
                from .llm import draft_review_section, draft_section
                import paperbrain.llm as _llmmod
                before = _llmmod.CALL_COUNT
                if sec.get("generation_mode") == "review":
                    body = draft_review_section(sec["h1"], sec["h2"],
                                                evidence[:4200], ent, paper_id)
                    if body and body.strip():
                        review_model_sections += 1
                else:
                    body = draft_section(sec["h1"], sec["h2"],
                                         evidence[:4200], ent)
                llm_calls += _llmmod.CALL_COUNT - before
                llm_used = True
            except Exception:
                body = ""
        if not body:
            # RAG 式接地: 仅在本节分配的 chunk 内检索 (防跨节串引), 首实质句逐字引用
            quote = ""
            for cid, _ in ranked[:2]:
                for sent in _ssplit(sub[cid]):
                    if len(sent) > 40:
                        quote, qsec = sent[:220], _sec_of(cid, paper_id)
                        break
                if quote:
                    break
            ent = ", ".join(outline.get("entities", [])[:4]) or "prior work"
            cites = f"[Ref: {paper_id}, Sec {qsec}]"
            if sec.get("generation_mode") == "review":
                body = (f"可核对的证据摘录：{quote}。离线规则模式不能可靠判断对照公平性、"
                        "统计充分性或拒稿风险；这些结论保持待复核。") if quote else \
                    "未找到可引用的审稿证据；离线规则模式不生成无依据的优点、缺陷或拒稿结论。"
            else:
                body = f"本节围绕 {ent} 展开。{quote}" if quote else \
                    f"本节围绕 {ent} 展开。{primary_claim[:220].strip()}"
        return f"## {sec['h1']} / {sec['h2']}\n{body} {cites}。"
    draft = draft_from_outline(outline, paper_id, generate_fn=_gen)

    model_tag = "规则回退"
    if draft_override and draft_override.strip():
        draft = draft_override.strip() + "\n"
        (out / "draft_session.md").write_text(draft, encoding="utf-8")
        model_tag = "muse-session"
        llm_used = True  # 会话模型实质参与, 台账如实记录
    elif llm_used:
        import os as _oss
        model_tag = _oss.environ.get("PAPERBRAIN_MODEL", "llm")

    # 全局一致性: 术语统一 + 过渡句 + 矛盾复用检查 (验真之前做, 过渡句无引用不污染门禁)
    from .consistency import polish
    mem_for_polish = {"entities": [], "relations": []}
    try:
        mem_for_polish = json.loads((out / "memory.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    draft, consistency = polish(draft, outline, mem_for_polish)
    _mark("draft", trace_fn=trace_fn)
    if consistency.get("contradictions"):
        state["warnings"].append(f"矛盾关系待复核: {consistency['contradictions'][:2]}")
    (out / "consistency.json").write_text(json.dumps(consistency, ensure_ascii=False, indent=2), encoding="utf-8")

    all_model = False
    calibration_verified = semantic_scorer is not None
    calibration_reason = "调用方显式注入 scorer/threshold"
    if semantic_scorer is not None:
        scorer, pass_th, review_th = semantic_scorer, scorer_threshold, 0.35
    else:
        try:
            from . import llm_ops
            all_model = use_llm and llm_ops.enabled()
        except Exception:
            all_model = False
        if all_model:
            # 全模型: NLI 蕴含由模型裁决 (词法链作失败兜底), 阈值用 embedding 线 0.82
            scorer, pass_th, review_th = llm_ops.make_nli_scorer(), 0.82, 0.35
            from .calibration import load_record
            cal = load_record("nli", Path(__file__).resolve().parent.parent / "thresholds_nli.json")
            calibration_verified = bool(cal["verified"])
            calibration_reason = cal["reason"]
            if calibration_verified:
                pass_th, review_th = cal["pass_th"], cal["review_th"]
        else:
            # 离线词法链: 仅接受带数据哈希且至少 50 对的生产标定记录。
            scorer, pass_th, review_th = make_scorer(gt), scorer_threshold, 0.35
            from .calibration import load_record
            cal = load_record("tfidf-corpus")
            calibration_verified = bool(cal["verified"])
            calibration_reason = cal["reason"]
            if calibration_verified:
                pass_th, review_th = cal["pass_th"], cal["review_th"]

    # 验真器会先本地粗排再只调用 Top-2 NLI；保留全节候选，不能用截头换速度。
    section_chunks = state.get("section_chunks")

    v = CitationVerifierV5(gt, state["fig_index"],
                           semantic_scorer=scorer, threshold=pass_th, review_threshold=review_th,
                           section_chunks=section_chunks)
    vr = v.verify_draft(draft)
    if not calibration_verified:
        from .calibration import enforce_review
        vr = enforce_review(vr, {"verified": False, "reason": calibration_reason})
    _mark("verify", trace_fn=trace_fn)

    try:
        from .llm import usage_snapshot
        gen_usage = usage_snapshot()
    except Exception:
        gen_usage = {"input": 0, "output": 0, "total": 0, "calls": 0,
                     "vision_images": 0, "by_stage": {}}
    prior_usage = state.get("llm_usage") or {}
    combined_usage = {
        "input": int(prior_usage.get("input", 0)) + int(gen_usage.get("input", 0)),
        "output": int(prior_usage.get("output", 0)) + int(gen_usage.get("output", 0)),
        "calls": int(prior_usage.get("calls", 0)) + int(gen_usage.get("calls", 0)),
        "vision_images": (int(prior_usage.get("vision_images", 0)) +
                          int(gen_usage.get("vision_images", 0))),
        "by_stage": dict(prior_usage.get("by_stage") or {}),
    }
    for key, value in (gen_usage.get("by_stage") or {}).items():
        combined_usage["by_stage"][key] = combined_usage["by_stage"].get(key, 0) + int(value)
    combined_usage["total"] = combined_usage["input"] + combined_usage["output"]
    budget_tokens = _budget_tokens(state.get("budget_tokens") or state["pass_tokens"], combined_usage)
    budget_verified = not bool(draft_override and draft_override.strip())
    if not budget_verified:
        # 会话外生成的输入账单不可见；至少计入交付草稿输出，并禁止宣称已完成预算验收。
        from .passes import estimate_tokens
        budget_tokens["external_draft"] = estimate_tokens(draft)
    vision_attempts = int(combined_usage.get("vision_images",
                                             state.get("vision_attempts", 0)))
    vision = estimate_vision_tokens(vision_attempts)
    br = check_budget(budget_tokens, num_images=vision_attempts)
    ledger = _ledger_row(paper_id, state["sha"], budget_tokens, vision,
                         usage=combined_usage, budget_verified=budget_verified)

    # M5 与整体放行门禁：引用 CLEAN 不得掩盖公式、预算、术语、过渡或矛盾风险。
    quality_reports = []
    if float(consistency.get("term_consistency_ratio", 1.0)) < 0.99:
        quality_reports.append({"citation": "[TERM_CONSISTENCY]", "status": "UNVERIFIED",
                                "reason": "术语一致率低于 99%"})
    if not consistency.get("transition_complete", True):
        quality_reports.append({"citation": "[TRANSITION]", "status": "UNVERIFIED",
                                "reason": "小节过渡不完整"})
    if consistency.get("contradictions"):
        quality_reports.append({"citation": "[CONTRADICTION]", "status": "NEEDS_REVIEW",
                                "reason": "草稿同时涉及已知 Contradicts 关系，必须复核"})
    if state.get("downgrade"):
        quality_reports.append({"citation": "[LOW_CONFIDENCE_EXTRACTION]", "status": "UNVERIFIED",
                                "reason": "源文档解析置信度不足且仅执行 Pass1，禁止放行生成结果"})
    if (review_sections_expected and not draft_override and
            review_model_sections != review_sections_expected):
        quality_reports.append({"citation": "[REVIEW_NOT_GENERATED]", "status": "UNVERIFIED",
                                "reason": ("专用审稿生成未完整运行；离线证据摘录不能替代同行评审 "
                                           f"({review_model_sections}/{review_sections_expected})")})
    if missing_evidence_sections and not draft_override:
        quality_reports.append({"citation": "[MISSING_SECTION_EVIDENCE]", "status": "UNVERIFIED",
                                "reason": "大纲小节没有有效 ChunkID 证据: " +
                                          "; ".join(missing_evidence_sections[:5])})
    if invalid_chunk_references and not draft_override:
        quality_reports.append({"citation": "[INVALID_CHUNK_REFERENCE]", "status": "UNVERIFIED",
                                "reason": "大纲引用不存在的 ChunkID: " +
                                          ", ".join(dict.fromkeys(invalid_chunk_references[:10]))})
    if outline_paper_mismatch:
        quality_reports.append({"citation": "[OUTLINE_PAPER_MISMATCH]", "status": "UNVERIFIED",
                                "reason": (f"大纲 paper_id={outline.get('paper_id')} 与当前 "
                                           f"paper_id={paper_id} 不一致")})
    if duplicate_outline_headings:
        quality_reports.append({"citation": "[DUPLICATE_OUTLINE_HEADING]", "status": "UNVERIFIED",
                                "reason": "大纲包含重复的 h1/h2 标题，无法建立唯一小节身份"})
    if state.get("fchk", {}).get("dirty"):
        quality_reports.append({"citation": "[FORMULA]", "status": "UNVERIFIED",
                                "reason": "存在未通过双校验的公式"})
    if not br.ok:
        quality_reports.append({"citation": "[BUDGET]", "status": "UNVERIFIED",
                                "reason": "; ".join(br.reasons)})
    if not budget_verified:
        quality_reports.append({"citation": "[BUDGET_UNVERIFIED]", "status": "NEEDS_REVIEW",
                                "reason": "外部会话草稿缺少可核验的完整输入/重试 Token 台账"})
    if quality_reports:
        vr["report"].extend(quality_reports)
        if any(r["status"] == "UNVERIFIED" for r in vr["report"]):
            vr["status"] = "DIRTY"
        elif vr["status"] == "CLEAN":
            vr["status"] = "NEEDS_REVIEW"
        vr["is_clean"] = False

    (out / "outline_confirmed.json").write_text(json.dumps(outline, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "draft.md").write_text(draft, encoding="utf-8")
    (out / "verify.json").write_text(json.dumps(vr, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "ledger.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ledger.keys()))
        w.writeheader()
        w.writerow(ledger)
    fchk, warnings = state["fchk"], state["warnings"]
    tables_info = state.get("tables", {})
    report = (f"# {paper_id} 运行报告\n\n- route={state['route']} downgrade={state['downgrade']} llm={llm_used} model={model_tag} llm_calls={llm_calls}\n"
              f"- coverage={state.get('coverage', 1.0)} (chars={state.get('total_chars', 0)})\n"
              f"- budget ok={br.ok and budget_verified} {br.reasons} tokens={budget_tokens}\n"
              f"- memory {state['mem_saved']}\n"
              f"- formulas total={len(fchk['formulas'])} unverified={fchk['dirty']}\n"
              f"- tables={tables_info}\n"
              f"- verify status={vr['status']} is_clean={vr['is_clean']} n={len(vr['report'])}\n"
              f"- calibration verified={calibration_verified} reason={calibration_reason}\n"
              f"- consistency={consistency}\n"
              f"- warnings={warnings}\n")
    (out / "report.md").write_text(report, encoding="utf-8")
    return {"out": str(out), "route": state["route"],
            "budget_ok": br.ok and budget_verified, "llm_used": llm_used,
            "model": model_tag, "formulas_dirty": fchk["dirty"],
            "verify": vr["status"], "is_clean": vr["is_clean"],
            "calibration_verified": calibration_verified, "ledger": ledger}


def run_paper(input_path: str, paper_id: str, out_dir: str,
              model_version: str = "mock-1.0",
              semantic_scorer: Optional[Callable[[str, str], float]] = None,
              scorer_threshold: float = 0.82,
              use_llm: bool = True, vision: bool = False,
              confirm_outline: bool = False,
              trace_fn: Optional[Callable[[str], None]] = None) -> Dict:
    """两阶段全流程；默认停在大纲，显式确认后才允许生成。"""
    o = run_outline(input_path, paper_id, out_dir, use_llm=use_llm, vision=vision, trace_fn=trace_fn)
    if not confirm_outline:
        return {"out": str(out_dir), "route": o["route"], "downgrade": o["downgrade"],
                "warnings": o["warnings"], "outline": o["outline"],
                "verify": "AWAITING_CONFIRMATION", "is_clean": False,
                "budget_ok": True, "awaiting_confirmation": True}
    g = generate_from_outline(out_dir, paper_id, o["outline"], semantic_scorer,
                              scorer_threshold, use_llm, trace_fn=trace_fn)
    g.update({"downgrade": o["downgrade"], "warnings": o["warnings"]})
    return g
