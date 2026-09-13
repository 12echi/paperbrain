"""一致性/去重/两步走测试."""
import shutil
import unittest
from pathlib import Path


class TestLexicalSim(unittest.TestCase):
    def test_alias_variant_merges(self):
        from paperbrain.graph import lexical_sim, sanitize_entities
        self.assertGreaterEqual(lexical_sim("ImageNet-1k", "ImageNet 1k"), 0.85)
        self.assertLess(lexical_sim("LET", "RBE"), 0.85)
        kept, stats = sanitize_entities(
            [{"name": "ImageNet 1k", "type": "Dataset/Benchmark"},
             {"name": "ImageNet-1k", "type": "Dataset/Benchmark"}],
            sim=lexical_sim, merge_th=0.85)
        self.assertEqual(stats["merged"], 1)
        self.assertEqual(len(kept), 1)


class TestConsistency(unittest.TestCase):
    def test_term_consistency_uses_identifier_boundaries(self):
        from paperbrain.consistency import term_consistency_ratio, unify_terms

        self.assertEqual(term_consistency_ratio("BLEU-4"), 1.0)
        self.assertEqual(term_consistency_ratio("bleu4"), 0.0)
        self.assertEqual(term_consistency_ratio("xbleu4x"), 1.0)
        self.assertEqual(unify_terms("xbleu4x"), ("xbleu4x", 0))

    def test_empty_or_invalid_runtime_aliases_cannot_corrupt_text(self):
        from paperbrain.consistency import term_consistency_ratio, unify_terms

        aliases = {"": "INJECTED", "blank": "", None: "NONE", "valid": "VALID"}
        self.assertEqual(unify_terms("text valid", aliases), ("text VALID", 1))
        self.assertEqual(term_consistency_ratio("unrelated text", aliases), 1.0)

    def test_versioned_policy_alias_is_used_without_substring_corruption(self):
        import json
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from paperbrain.consistency import unify_terms
        with tempfile.TemporaryDirectory() as td:
            policy = Path(td, "policy.json")
            policy.write_text(json.dumps({
                "schema": "paperbrain-graph-policy-v1", "reviewed_at": None,
                "aliases": {"abc": "ABC-Canonical"}, "blacklist": []}), encoding="utf-8")
            with patch.dict(os.environ, {"PAPERBRAIN_GRAPH_POLICY": str(policy)}, clear=False):
                text, changed = unify_terms("abc is used; xabcx is a different identifier")
        self.assertEqual(text, "ABC-Canonical is used; xabcx is a different identifier")
        self.assertEqual(changed, 1)

    def test_unify_and_transition(self):
        from paperbrain.consistency import polish
        ol = {"paper_id": "P", "version": "v1", "created_at": "t",
              "sections": [
                  {"h1": "A", "h2": "a", "h3": [], "chunks": ["P_Sec1_C001"], "claims": ["c"]},
                  {"h1": "B", "h2": "b", "h3": [], "chunks": ["P_Sec2_C001"], "claims": ["c"]}],
              "entities": []}
        draft = "## A / a\nWe use FlashAttention 2 here [Ref: P, Sec 1]。\n\n## B / b\nNext [Ref: P, Sec 2]。"
        out, rep = polish(draft, ol, {"relations": []})
        self.assertIn("FlashAttention-v2", out)
        self.assertNotIn("FlashAttention 2", out.replace("FlashAttention-v2", ""))
        self.assertEqual(rep["transitions"], 1)
        self.assertIn("Building on", out)
        self.assertNotIn("承接上节", out)

    def test_chinese_draft_gets_chinese_transition(self):
        from paperbrain.consistency import polish
        outline = {"sections": [
            {"h1": "研究背景", "h2": "问题定义", "claims": ["现有方法存在计算瓶颈"]},
            {"h1": "研究方法", "h2": "模型架构", "claims": ["本文提出改进模型"]},
        ]}
        draft = "## 研究背景 / 问题定义\n本文分析现有方法的局限。\n\n## 研究方法 / 模型架构\n本文介绍新的模型。"

        output, report = polish(draft, outline, {"relations": []})

        self.assertIn("承接", output)
        self.assertNotIn("Building on", output)
        self.assertTrue(report["transition_complete"])

    def test_reordered_headings_fail_closed_without_wrong_transitions(self):
        from paperbrain.consistency import polish
        outline = {"sections": [
            {"h1": "Method", "h2": "Model", "claims": ["method claim"]},
            {"h1": "Results", "h2": "Evaluation", "claims": ["result claim"]},
        ]}
        draft = "## Results / Evaluation\nResults.\n\n## Method / Model\nMethod."

        output, report = polish(draft, outline, {"relations": []})

        self.assertEqual(output, draft)
        self.assertEqual(report["transitions"], 0)
        self.assertEqual(report["expected_transitions"], 1)
        self.assertFalse(report["transition_complete"])

    def test_contradict_flag(self):
        from paperbrain.consistency import polish
        ol = {"paper_id": "P", "version": "v1", "created_at": "t", "sections": [], "entities": []}
        mem = {"relations": [{"from": "X", "to": "Y", "rel": "Contradicts", "ev": "e"}]}
        _, rep = polish("X beats Y here.", ol, mem)
        self.assertEqual(len(rep["contradictions"]), 1)


class TestSessionHandoff(unittest.TestCase):
    def test_export_import(self):
        import json
        from paperbrain.session_llm import export_prompts, import_answers
        p = export_prompts("out/claude01")
        self.assertIn("summaries", p)
        self.assertTrue(len(p["sections"]) >= 3)
        d = import_answers("out/claude01", {"drafts": [{"h1": "T", "text": "x"}]})
        self.assertIn("## T", d)


class TestTwoPhase(unittest.TestCase):
    def test_external_draft_verified(self):
        # 外部模型草稿直验真 (Muse 会话模式), 记 model 标签
        from paperbrain.pipeline import run_outline, generate_from_outline
        out = Path("out/test_muse")
        if out.exists():
            shutil.rmtree(out)
        o = run_outline("demo/sample_paper.txt", "2024_NeurIPS_01", str(out))
        secs = {s["sec"] for s in
                __import__("json").loads((out / "sections.json").read_text(encoding="utf-8"))}
        self.assertTrue(secs)
        g = generate_from_outline(
            str(out), "2024_NeurIPS_01", o["outline"],
            draft_override="External finding here [Ref: 2024_NeurIPS_01, Sec 0]。")
        self.assertEqual(g["model"], "muse-session")
        self.assertIn(g["verify"], ("CLEAN", "NEEDS_REVIEW", "DIRTY"))

    def test_outline_then_generate(self):
        from paperbrain.pipeline import run_outline, generate_from_outline
        out = Path("out/test_twophase")
        if out.exists():
            shutil.rmtree(out)
        o = run_outline("demo/sample_paper.txt", "2024_NeurIPS_01", str(out))
        self.assertIn("sections", o["outline"])
        self.assertTrue((out / "outline_v1.json").exists())
        g = generate_from_outline(str(out), "2024_NeurIPS_01", o["outline"])
        self.assertEqual(g["verify"], "NEEDS_REVIEW")
        self.assertFalse(g["calibration_verified"])
        self.assertTrue((out / "consistency.json").exists())


if __name__ == "__main__":
    unittest.main()
