"""深度精读 A+B+C 测试 (离线回退 + 模拟模型路径)."""
import shutil
import tempfile
import unittest
import json
from pathlib import Path

import paperbrain.llm as llm


class TestDeepRead(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        from paperbrain.pipeline import run_outline
        run_outline("demo/sample_paper.txt", "DR01", self.out, use_llm=False)

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_fallback_offline(self):
        from paperbrain.deepread import build
        r = build(self.out, "DR01", use_llm=False)
        self.assertIn("深度解读", r["markdown"])
        self.assertFalse(r["llm"])
        self.assertTrue((Path(self.out) / "deepread.md").exists())

    def test_mocked_llm_full(self):
        from paperbrain import deepread
        orig = {k: getattr(llm, k) for k in
                ("deep_analyze", "gen_questions", "answer_questions", "global_synthesis")}
        try:
            llm.deep_analyze = lambda ctx, focus="", paper_id="": (
                "## 一句话主张\n它主张 X。\n\n"
                "## 研究问题与空白 (Gap)\n先前做法不足。\n\n"
                "## 核心洞见 (Key Insight)\n关键想法是 Y。\n\n"
                "## 方法与关键设计\n采用 Z。\n\n"
                "## 证据强度\n在数据集上提升 2 点。\n\n"
                "## 可复现性\n代码未公开。\n\n"
                "## 延伸设想 (If I were to extend)\n可加入方差分析。")
            llm.gen_questions = lambda ctx, n=7: '[{"perspective":"怀疑者","q":"对照是否公平?"}]'
            llm.answer_questions = lambda ctx, qj, pid: "**对照是否公平?** 是, 见 [Ref: DR01, Sec 3]。"
            llm.global_synthesis = lambda ctx, paper_id="": ('{"argument_map":[{"claim":"C","evidence":"E","limitation":"L"}],'
                                                             '"positioning":{"improves_upon":"A","contradicts":"材料未提供","gap_left":"G"},'
                                                             '"field_view":"位于 X 领域。"}')
            r = deepread.build(self.out, "DR01", use_llm=True, focus="方法为何有效")
            self.assertTrue(r["llm"])
            brief = r["markdown"]
            full = r["full_markdown"]
            # 成果正文=精读简报 (简洁), 完整深读含论证地图/全局定位/领域视野/拷问
            for key in ("一句话结论", "核心要点", "主要质疑", "关键追问", "关注点"):
                self.assertIn(key, brief)
            for key in ("论证地图", "全局定位", "领域视野", "多视角拷问"):
                self.assertIn(key, full)
            self.assertEqual(len(r["questions"]), 1)
            self.assertIn("improves_upon", r["synthesis"]["positioning"])
            self.assertTrue((Path(self.out) / "deepread_questions.json").exists())
            self.assertTrue((Path(self.out) / "deepread_full.md").exists())
            verification = json.loads(
                (Path(self.out) / "deepread_verify.json").read_text(encoding="utf-8"))
            self.assertEqual(verification["artifacts"]["brief"]["status"], "NO_CITATION")
            self.assertEqual(r["verify"], "DIRTY")
            self.assertFalse(r["is_clean"])
        finally:
            for k, v in orig.items():
                setattr(llm, k, v)

    def test_context_has_sections(self):
        from paperbrain.deepread import build_context
        ctx = build_context(self.out)
        self.assertIn("abstract", ctx.lower())
        self.assertGreater(len(ctx), 200)

    def test_brief_truncation_preserves_selected_sentence_citation(self):
        from paperbrain.deepread import _first_sentences
        source = ("A supported result contains enough explanatory prose to exceed a deliberately "
                  "small display cap while retaining provenance [Ref: DR01, Sec 3].")
        brief = _first_sentences(source, n=1, cap=48)
        self.assertIn("[Ref: DR01, Sec 3]", brief)

    def test_synthesis_schema_is_sanitized_before_rendering(self):
        from paperbrain.deepread import _sanitize_synthesis
        self.assertEqual(_sanitize_synthesis({
            "argument_map": "not-a-list", "positioning": "not-an-object",
            "field_view": ["not", "text"], "unexpected": "ignored",
        }), {})
        cleaned = _sanitize_synthesis({
            "argument_map": [{"claim": " C ", "evidence": " E ", "limitation": " L "}],
            "positioning": {"gap_left": " G ", "extra": "ignored"},
            "field_view": " F ", "unexpected": "ignored",
        })
        self.assertEqual(cleaned["argument_map"][0]["claim"], "C")
        self.assertEqual(cleaned["positioning"], {"gap_left": "G"})
        self.assertEqual(cleaned["field_view"], "F")

    def test_question_schema_rejects_empty_unknown_and_duplicate_rows(self):
        from paperbrain.deepread import _sanitize_questions
        cleaned = _sanitize_questions([
            {"perspective": "怀疑者", "q": " 对照是否公平？ "},
            {"perspective": "怀疑者", "q": "对照 是否公平？"},
            {"perspective": "作者", "q": "为什么？"},
            {"perspective": "统计学家", "q": ""},
            {"perspective": "方法学家", "q": "消融是否隔离了关键变量？"},
            "bad-row",
        ])
        self.assertEqual(cleaned, [
            {"perspective": "怀疑者", "q": "对照是否公平？"},
            {"perspective": "方法学家", "q": "消融是否隔离了关键变量？"},
        ])

    def test_invalid_question_rerun_removes_stale_question_artifact(self):
        from paperbrain import deepread
        question_path = Path(self.out) / "deepread_questions.json"
        question_path.write_text('[{"perspective":"怀疑者","q":"旧问题"}]',
                                 encoding="utf-8")
        original = {name: getattr(llm, name) for name in
                    ("deep_analyze", "gen_questions", "global_synthesis")}
        try:
            llm.deep_analyze = lambda ctx, focus="", paper_id="": "## 一句话主张\n有效正文"
            llm.gen_questions = lambda ctx: '[{"perspective":"作者","q":"空泛问题？"}]'
            llm.global_synthesis = lambda ctx, paper_id="": "{}"
            result = deepread.build(self.out, "DR01", use_llm=True)
        finally:
            for name, value in original.items():
                setattr(llm, name, value)
        self.assertEqual(result["questions"], [])
        self.assertFalse(question_path.exists())
        self.assertIn("多视角问题 JSON 不符合结构契约",
                      result["synthesis"].get("_warnings", []))

    def test_global_synthesis_prompt_uses_real_paper_id(self):
        original = llm.chat
        captured = []
        try:
            llm.chat = lambda messages, **kwargs: captured.append(messages[0]["content"]) or "{}"
            llm.global_synthesis("### Sec 2\nEvidence", paper_id="REAL_01")
        finally:
            llm.chat = original
        self.assertIn("[Ref: REAL_01, Sec X]", captured[0])
        self.assertNotIn("[Ref: PaperID", captured[0])

    def test_all_model_stage_failures_return_complete_offline_skeleton(self):
        original = {name: getattr(llm, name) for name in
                    ("deep_analyze", "gen_questions", "answer_questions", "global_synthesis")}

        def fail(*args, **kwargs):
            raise RuntimeError("model unavailable")

        try:
            for name in original:
                setattr(llm, name, fail)
            from paperbrain.deepread import build
            result = build(self.out, "DR01", use_llm=True)
        finally:
            for name, value in original.items():
                setattr(llm, name, value)

        self.assertFalse(result["llm"])
        self.assertIn("离线模式", result["markdown"])
        self.assertIn("方法与关键设计", result["markdown"])
        self.assertTrue(result["synthesis"].get("_warnings"))


if __name__ == "__main__":
    unittest.main()
