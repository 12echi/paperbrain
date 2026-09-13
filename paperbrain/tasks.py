"""解读类别与任务调度 (v5.0).

把"论文进来做什么"显式化。每类任务产出各自的重点报告, 都落盘 + 可入记忆库。
- full_read  全文精读: 分章→纪要→图谱→大纲→草稿→三阶段验真
- outline    速览大纲: 只出骨架+宏观纪要 (最快)
- method     方法拆解: 聚焦模型/公式/训练目标 + 公式门禁
- figures    图表识别: 抽图/表 + (可选)VL解读
- review     审稿批判: 优点/质疑/拒稿点, 逐条带引用
- memory     记忆沉淀: 生成学习记忆条目 (要点/实体/关系)
"""
from typing import Callable, Dict, List, Optional

from . import config

TASKS: Dict[str, Dict] = {
    "full_read": {"label": "全文精读", "icon": "read",
                  "desc": "完整读通: 分章、四遍纪要、图谱、大纲、草稿、引文验真"},
    "outline": {"label": "速览大纲", "icon": "list",
                "desc": "只出三级大纲与宏观骨架, 秒级了解论文讲了什么"},
    "method": {"label": "方法拆解", "icon": "gear",
               "desc": "聚焦方法/模型/公式/训练目标, 公式过 katex+sympy 门禁"},
    "figures": {"label": "图表识别", "icon": "image",
                "desc": "抽取论文配图与表格, 可选 VL 模型读图并中文解读"},
    "review": {"label": "审稿批判", "icon": "shield",
               "desc": "以苛刻审稿人视角给出优点/质疑/拒稿风险, 逐条可溯源"},
    "memory": {"label": "记忆沉淀", "icon": "brain",
               "desc": "把要点、实体、关系沉淀进学习记忆库, 供日后快速调用"},
}


def task_labels() -> List[Dict]:
    return [{"id": k, **v} for k, v in TASKS.items()]


def run_task(task: str, input_path: str, paper_id: str, out_dir: str,
             use_llm: bool = True, vision: bool = False,
             scorer_threshold: float = 0.82, focus: str = "",
             depth: str = "standard",
             progress: Optional[Callable[[str], None]] = None) -> Dict:
    """按类别执行。返回 {task, out, verify, markdown, ...}。失败即抛, 不伪造。"""
    from pathlib import Path
    def _p(s: str):
        if progress:
            try:
                progress(s)
            except Exception:
                pass
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    task = task if task in TASKS else "full_read"

    if task == "outline":
        from .pipeline import run_outline
        _p("预处理·分章·宏观纪要")
        o = run_outline(input_path, paper_id, out_dir, use_llm=use_llm, vision=vision)
        md = _outline_md(o)
        (out / "report.md").write_text(md, encoding="utf-8")
        return {"task": task, "out": str(out), "verify": "NOT_RUN",
                "is_clean": None, "markdown": md, "outline": o["outline"],
                "warnings": o["warnings"]}

    if task == "figures":
        _p("抽取图表")
        return _run_figures(input_path, paper_id, out_dir, vision)

    if task == "method":
        from .pipeline import run_outline, generate_from_outline
        _p("预处理·分章·方法定位")
        o = run_outline(input_path, paper_id, out_dir, use_llm=use_llm, vision=False)
        _p("方法草稿·公式门禁·验真")
        g = generate_from_outline(out_dir, paper_id, _method_outline(o["outline"]),
                                  scorer_threshold=scorer_threshold, use_llm=use_llm)
        md = _read_md(out)
        return {"task": task, **g, "markdown": md, "warnings": o["warnings"]}

    if task == "review":
        from .pipeline import run_outline, generate_from_outline
        _p("预处理·分章·四遍纪要")
        o = run_outline(input_path, paper_id, out_dir, use_llm=use_llm, vision=vision)
        _p("审稿批判·引文验真")
        g = generate_from_outline(out_dir, paper_id, _review_outline(o["outline"]),
                                  scorer_threshold=scorer_threshold, use_llm=use_llm)
        md = _read_md(out)
        return {"task": task, **g, "markdown": md, "warnings": o["warnings"]}

    if task == "memory":
        from .pipeline import run_outline
        _p("预处理·纪要·图谱")
        o = run_outline(input_path, paper_id, out_dir, use_llm=use_llm, vision=False)
        md = _memory_md(o)
        (out / "report.md").write_text(md, encoding="utf-8")
        return {"task": task, "out": str(out), "verify": "N/A", "is_clean": None,
                "markdown": md, "warnings": o["warnings"]}

    # 默认: full_read —— 按 depth 三档预算执行 (Q9)
    #   fast:     仅 分章+四遍纪要+大纲 (最省)
    #   standard: fast + 穿透式深读 A+B+C (默认; 草稿/验真按需)
    #   deep:     standard + 综述草稿 + 引文验真 (+ VL 读图, 若开启)
    from .pipeline import run_outline
    depth = depth if depth in ("fast", "standard", "deep") else "standard"
    _p("预处理·分章·四遍纪要")
    o = run_outline(input_path, paper_id, out_dir, use_llm=use_llm,
                    vision=(vision and depth == "deep"))
    ledger = {"total": sum(o.get("pass_tokens", {}).values()),
              "text_tokens": sum(o.get("pass_tokens", {}).values()),
              "vision_tokens": 0, "paper_id": paper_id, "cost": "n/a"}
    result = {"task": "full_read", "out": str(out_dir), "route": o.get("route"),
              "budget_ok": True, "model": "?", "verify": "NOT_RUN", "is_clean": None,
              "ledger": ledger, "warnings": o["warnings"], "depth": depth}

    if depth == "fast":
        md = _outline_md(o)
        (out / "report.md").write_text(md, encoding="utf-8")
        result.update({"markdown": md})
        return result

    _p("穿透式深读·多视角拷问·全局综合")
    md = _outline_md(o)
    deep_llm = False
    try:
        from .deepread import build as _deep
        dr = _deep(out_dir, paper_id, use_llm=use_llm, focus=focus)
        md = dr["markdown"] or md
        deep_llm = bool(dr.get("llm"))
        if dr.get("budget_tokens"):
            total_text = sum(dr["budget_tokens"].values())
            ledger.update({"total": total_text, "text_tokens": total_text,
                           "input_tokens": (dr.get("llm_usage") or {}).get("input", 0),
                           "output_tokens": (dr.get("llm_usage") or {}).get("output", 0),
                           "llm_calls": (dr.get("llm_usage") or {}).get("calls", 0)})
        result["budget_ok"] = bool(dr.get("budget_ok", False))
    except Exception as e:
        o["warnings"].append(f"深读跳过 ({e})")
    result.update({"markdown": md, "deepread_llm": deep_llm})
    if depth == "standard" and 'dr' in locals():
        result.update({"verify": dr.get("verify", "NEEDS_REVIEW"),
                       "is_clean": bool(dr.get("is_clean", False))})

    if depth == "deep":
        _p("等待用户确认大纲")
        result.update({"verify": "AWAITING_CONFIRMATION", "is_clean": False,
                       "awaiting_confirmation": True, "outline": o["outline"]})
    return result


# ------------------------------------------------------------------ helpers
def _read_md(out) -> str:
    from pathlib import Path
    p = Path(out) / "draft.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _load_json(out, name):
    import json
    from pathlib import Path
    p = Path(out) / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _outline_md(o: Dict) -> str:
    ol = o.get("outline", {})
    lines = [f"# 速览大纲 · {ol.get('paper_id','')}", ""]
    for sec in ol.get("sections", []):
        lines.append(f"## {sec['h1']} / {sec['h2']}")
        for k in sec.get("h3", []):
            lines.append(f"- {k}")
        cids = ", ".join(sec.get("chunks", []))
        lines.append(f"- 依据: {cids or '—'}")
        lines.append("")
    ent = ol.get("entities", [])
    if ent:
        lines.append("## 核心实体")
        lines.append("- " + "、".join(ent))
    return "\n".join(lines) + "\n"


def _method_outline(outline: Dict) -> Dict:
    secs = [s for s in outline.get("sections", []) if s.get("h1") == "Method"]
    if secs:
        return {**outline, "sections": secs}
    missing = {"h1": "Method", "h2": "Evidence unavailable",
               "h3": ["Manual section identification required"],
               "chunks": [], "claims": [], "status": "NEEDS_REVIEW",
               "generation_mode": "method"}
    return {**outline, "sections": [missing]}


def _review_outline(outline: Dict) -> Dict:
    secs = []
    for s in outline.get("sections", []):
        if s.get("h1") in ("Conclusion", "Method", "Experiments"):
            secs.append({**s, "h1": s["h1"] + "·审稿视角",
                         "generation_mode": "review"})
    return {**outline, "sections": secs or outline.get("sections", [])}


def _memory_md(o: Dict) -> str:
    import json
    ol = o.get("outline", {})
    mem = _load_json(o["out"], "memory.json")
    passes = _load_json(o["out"], "passes.json")
    lines = [f"# 学习记忆 · {ol.get('paper_id','')}", ""]
    lines.append("## 宏观要点")
    lines.append(_clean(passes.get("pass1", "")) or "—")
    lines.append("")
    ents = mem.get("entities", [])
    if ents:
        lines.append("## 关键实体")
        for e in ents:
            lines.append(f"- {e['name']}（{e['type']}）")
    rels = mem.get("relations", [])
    if rels:
        lines.append("## 关系")
        for r in rels[:20]:
            lines.append(f"- {r['from']} --[{r['rel']}]--> {r['to']}")
    return "\n".join(lines) + "\n"


def _clean(t: str) -> str:
    import re
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]", "", t or "").strip()


def _run_figures(input_path: str, paper_id: str, out_dir: str, vision: bool) -> Dict:
    import json
    from pathlib import Path
    out = Path(out_dir)
    lines = [f"# 图表识别 · {paper_id}", ""]
    figs, tables = [], []
    if str(input_path).lower().endswith(".pdf"):
        try:
            from .vision import extract_images
            # 每次 describe_figure 只发送 1 图；单篇有效视觉上限由配置硬封顶为 5。
            figs = extract_images(str(input_path), str(out),
                                  max_images=config.max_images())
        except Exception as e:
            lines.append(f"> 配图抽取失败: {e}")
        try:
            import fitz
            with fitz.open(str(input_path)) as doc:
                for pno, pg in enumerate(doc):
                    for t in pg.find_tables():
                        extracted = t.extract()
                        tables.append({"page": pno, "rows": len(t.rows),
                                       "cols": len(t.cols), "cells": len(t.cells),
                                       "sample": [extracted[0] if extracted else []]})
        except Exception as e:
            lines.append(f"> 表格抽取跳过: {e}")
    lines.append(f"## 配图（共 {len(figs)} 张）")
    for f in figs:
        lines.append(f"- 第 {f['page']+1} 页, {f['pixels']} 像素, 文件 {Path(f['image']).name}")
        if vision:
            try:
                from .vision import describe_figure
                an = describe_figure(f["image"], hint=f.get("caption", ""))
                f.update(an)
                lines.append(f"  - 解读: {an.get('analysis','')}")
            except Exception as e:
                lines.append(f"  - 解读失败: {e}")
    lines.append("")
    lines.append(f"## 表格（共 {len(tables)} 张）")
    for t in tables:
        lines.append(f"- 第 {t['page']+1} 页, {t['rows']}行×{t['cols']}列, {t['cells']} 单元格")
    md = "\n".join(lines) + "\n"
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "figures.json").write_text(json.dumps(figs, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (out / "tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    return {"task": "figures", "out": str(out), "verify": "N/A", "is_clean": None,
            "markdown": md, "n_figures": len(figs), "n_tables": len(tables)}
