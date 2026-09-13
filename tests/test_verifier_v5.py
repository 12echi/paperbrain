"""Verifier V5 生产门禁测试 (stdlib unittest, 离线可跑).
覆盖 v4 审计全部 P0: 全角/多图/Fig-only/无scorer/附录/大小写漂移/零引用.
"""
import unittest

from paperbrain.verifier import CitationVerifierV5
from paperbrain.ids import norm_sec, norm_figtab, parse_loc, make_cache_key
from paperbrain.budget import check_budget
from paperbrain.graph import sanitize_entities


def hi(a: str, b: str) -> float:
    return 0.9


def lo(a: str, b: str) -> float:
    return 0.1


GT = {
    "2024_NeurIPS_01_3.2": "We improve BLEU-4 by 2 points on HumanEval.",
    "2024_NeurIPS_01_fig2": "Figure 2 caption.",
    "2024_NeurIPS_01_A.1": "Appendix result.",
}
FIG = {"2024_NeurIPS_01_3.2": {"figs": ["fig2"], "tabs": ["tab1"]}}


class TestV5(unittest.TestCase):
    def test_clean(self):
        v = CitationVerifierV5(GT, FIG, semantic_scorer=hi)
        r = v.verify_draft("结论见 [Ref: 2024_NeurIPS_01, Sec 3.2]。")
        self.assertEqual(r["status"], "CLEAN")
        self.assertTrue(r["is_clean"])

    def test_low_score_dirty(self):
        v = CitationVerifierV5(GT, FIG, semantic_scorer=lo)
        r = v.verify_draft("结论见 [Ref: 2024_NeurIPS_01, Sec 3.2]。")
        self.assertEqual(r["status"], "DIRTY")

    def test_no_citation(self):
        v = CitationVerifierV5(GT, FIG, semantic_scorer=hi)
        r = v.verify_draft("无引用的段落。")
        self.assertEqual(r["status"], "NO_CITATION")
        self.assertFalse(r["is_clean"])

    def test_fullwidth_brackets(self):
        v = CitationVerifierV5(GT, FIG, semantic_scorer=hi)
        r = v.verify_draft("见【Ref: 2024_NeurIPS_01, Sec 3.2】。")
        self.assertEqual(r["status"], "CLEAN")

    def test_fullwidth_comma(self):
        v = CitationVerifierV5(GT, FIG, semantic_scorer=hi)
        r = v.verify_draft("见 [Ref: 2024_NeurIPS_01， Sec 3.2]。")
        self.assertEqual(r["status"], "CLEAN")

    def test_fig_only_global_check(self):
        # Fig 9 不在全局清单 -> UNVERIFIED (不再绕过)
        v = CitationVerifierV5(GT, FIG, semantic_scorer=hi)
        r = v.verify_draft("见 [Ref: 2024_NeurIPS_01, Fig 9]。")
        self.assertEqual(r["status"], "DIRTY")

    def test_fig_only_no_index_needs_review(self):
        # 有 gt 但无清单 -> NEEDS_REVIEW, 不得 PASS
        gt2 = {"2024_NeurIPS_01_fig9": "cap"}
        v = CitationVerifierV5(gt2, {}, semantic_scorer=hi)
        r = v.verify_draft("见 [Ref: 2024_NeurIPS_01, Fig 9]。")
        self.assertEqual(r["status"], "NEEDS_REVIEW")

    def test_multi_fig_rejected(self):
        v = CitationVerifierV5(GT, FIG, semantic_scorer=hi)
        r = v.verify_draft("见 [Ref: 2024_NeurIPS_01, Sec 3.2, Fig 2, Tab 1]。")
        self.assertEqual(r["status"], "DIRTY")
        self.assertIn("多图表", r["report"][0]["reason"])

    def test_no_scorer_needs_review(self):
        v = CitationVerifierV5(GT, FIG, semantic_scorer=None)
        r = v.verify_draft("见 [Ref: 2024_NeurIPS_01, Sec 3.2]。")
        self.assertEqual(r["status"], "NEEDS_REVIEW")
        self.assertFalse(r["is_clean"])
        self.assertNotEqual(r["report"][0]["status"], "PASS")

    def test_appendix(self):
        v = CitationVerifierV5(GT, FIG, semantic_scorer=hi)
        r = v.verify_draft("见 [Ref: 2024_NeurIPS_01, Sec A.1]。")
        self.assertEqual(r["status"], "CLEAN")
        self.assertEqual(norm_sec("Appendix A.1"), "A.1")

    def test_case_drift_needs_review(self):
        v = CitationVerifierV5(GT, FIG, semantic_scorer=hi)
        r = v.verify_draft("见 [Ref: 2024_neurips_01, Sec 3.2]。")
        self.assertEqual(r["status"], "NEEDS_REVIEW")

    def test_sec_fig_ok(self):
        v = CitationVerifierV5(GT, FIG, semantic_scorer=hi)
        r = v.verify_draft("见 [Ref: 2024_NeurIPS_01, Sec 3.2, Fig 2]。")
        self.assertEqual(r["status"], "CLEAN")

    def test_sec_fig_mismatch(self):
        v = CitationVerifierV5(GT, FIG, semantic_scorer=hi)
        r = v.verify_draft("见 [Ref: 2024_NeurIPS_01, Sec 3.2, Fig 9]。")
        self.assertEqual(r["status"], "DIRTY")

    def test_chinese_fig(self):
        self.assertEqual(norm_figtab("图2"), "fig2")
        self.assertEqual(norm_figtab("表3"), "tab3")

    def test_cache_key_requires_versions(self):
        with self.assertRaises(ValueError):
            make_cache_key("a", "b", "p", "")
        k = make_cache_key("pdf", "sec", "p1", "m1", "e1")
        self.assertEqual(len(k), 32)

    def test_budget_dual_cap(self):
        # 9 图必超 vision
        r = check_budget({"pass1": 2000, "pass2": 3000, "pass3_text": 3000, "pass4": 1500},
                         num_images=9)
        self.assertFalse(r.ok)
        self.assertTrue(any("vision" in x for x in r.reasons))
        ok = check_budget({"pass1": 2000, "pass2": 3000, "pass3_text": 3000, "pass4": 1500},
                          num_images=3)
        self.assertTrue(ok.ok)

    def test_graph_schema(self):
        items = [{"name": "FlashAttention 2", "type": "Algorithm/Model"},
                 {"name": "model", "type": "Algorithm/Model"},
                 {"name": "X", "type": "Author"}]
        kept, stats = sanitize_entities(items)
        self.assertEqual(stats["rejected_schema"], 1)
        self.assertEqual(stats["rejected_generic"], 1)
        self.assertEqual(kept[0]["name"], "FlashAttention-v2")


if __name__ == "__main__":
    unittest.main()
