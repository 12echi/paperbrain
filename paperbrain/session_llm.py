"""Session 模型交接 (把当作模型用): prompt 导出 + 答案导入.

没有 API Key 时, 由本会话(或其他强模型)充当 LLM:
1. export_prompts(out_dir) -> prompts.json {summaries:{pass1..}, sections:[...]}
2. 人/模型填写答案存 answers.json {summaries:{...}, drafts:{h1: text-with-citations}}
3. import_answers(out_dir, answers) -> 覆盖 outline claims + 生成带引用草稿走验真
(APPEND_ONLY: 答案只允许收紧引用, 不允许新增无源断言)
"""
import json
from pathlib import Path
from typing import Dict


def export_prompts(out_dir: str) -> Dict:
    out = Path(out_dir)
    state = json.loads((out / "state.json").read_text(encoding="utf-8"))
    outline = json.loads((out / "outline_v1.json").read_text(encoding="utf-8"))
    secs = {s["name"]: s.get("text", "") for s in
            json.loads((out / "sections.json").read_text(encoding="utf-8"))}
    gt = state["ground_truth"]
    src = {"pass1": "\n".join([secs.get("abstract", ""), secs.get("intro", ""),
                               secs.get("conclusion", "")]),
           "pass2": secs.get("method", ""), "pass3": secs.get("experiments", ""),
           "pass4": "基于前三遍纪要批判"}
    prompts = {"paper_id": state["paper_id"], "summaries": {}, "sections": []}
    for k in ("pass1", "pass2", "pass3", "pass4"):
        prompts["summaries"][k] = (f"用中文总结(≤300字,只基于原文):\n{src[k][:4000]}")
    for s in outline["sections"]:
        ctx = "\n".join(gt.get(c, "")[:1200] for c in s.get("chunks", []))
        prompts["sections"].append({"h1": s["h1"], "h2": s["h2"], "chunks": s["chunks"],
                                    "prompt": f"写综述小节[{s['h1']}/{s['h2']}],只用以下原文,每段末标[Ref: {state['paper_id']}, Sec X]:\n{ctx[:3000]}"})
    (out / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    return prompts


def import_answers(out_dir: str, answers: Dict) -> str:
    """answers: {drafts: [{h1, text}]} -> draft_session.md (验真由调用方跑)."""
    out = Path(out_dir)
    parts = []
    for d in answers.get("drafts", []):
        parts.append(f"## {d.get('h1','')}\n{d.get('text','').strip()}")
    draft = "\n\n".join(parts) + "\n"
    (out / "draft_session.md").write_text(draft, encoding="utf-8")
    return draft
