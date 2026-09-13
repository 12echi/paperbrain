"""P1 深优回归测试: 尾部截断/公式打标复用/动态大纲/无源跳过."""
import unittest


class TestPackageHygiene(unittest.TestCase):
    def test_all_compile(self):
        import compileall
        from pathlib import Path
        ok = compileall.compile_dir("paperbrain", quiet=1)
        self.assertTrue(ok)
        self.assertTrue((Path("paperbrain") / "__init__.py").exists())


class TestTailCut(unittest.TestCase):
    def test_references_dropped(self):
        from paperbrain.sections import split_sections
        text = ("Abstract\nSome intro text here about methods and results with enough length "
                "to pass confidence thresholds for the abstract section splitter logic.\n"
                "Methods\nWe did things with models and evaluation metrics in detail.\n"
                "References\n[1] Someone et al. 2020. A great paper.\n[2] Other. 2021.")
        meta = {}
        secs = split_sections(text, "T1", meta)
        self.assertTrue(meta.get("dropped_tail"))
        all_text = "\n".join(s["text"] for s in secs)
        self.assertNotIn("Someone et al", all_text)

    def test_no_tail(self):
        from paperbrain.sections import split_sections
        meta = {}
        split_sections("Abstract\n" + "x" * 500 + "\nMethods\n" + "y" * 500, "T2", meta)
        self.assertFalse(meta.get("dropped_tail"))


class TestFormulasMark(unittest.TestCase):
    def test_mark_reuses_check(self):
        from paperbrain.formulas import check_formulas, mark_text
        t = r"好公式 $E=mc^2$ 坏公式 $\zzz{{{@@@$ 结束"
        marked = mark_text(t, check_formulas(t))
        self.assertIn("FORMULA_UNVERIFIED", marked)
        self.assertIn("E=mc^2", marked)


class TestDynamicOutline(unittest.TestCase):
    def test_empty_section_skipped(self):
        from paperbrain.outline import build_outline
        passes = {"ground_truth": {"P_Sec2_C001": "method text about models"},
                  "pass1": "a", "pass2": "b", "pass3": "c", "pass4": "d"}
        mem = {"entities": []}
        ol = build_outline(passes, mem, "P")
        h1s = [s["h1"] for s in ol["sections"]]
        self.assertIn("Method", h1s)
        self.assertNotIn("Background", h1s)
        self.assertNotIn("Experiments", h1s)

    def test_generate_fn_passthrough(self):
        from paperbrain.outline import draft_from_outline
        d2 = draft_from_outline(
            {"paper_id": "P", "version": "v1", "created_at": "t",
             "sections": [{"h1": "M", "h2": "x", "h3": [], "chunks": ["P_Sec2_C001"], "claims": ["c"]}],
             "entities": []},
            "P", generate_fn=lambda s: "body [Ref: P, Sec 2]")
        self.assertIn("[Ref: P, Sec 2]", d2)


class TestFullRead(unittest.TestCase):
    def test_claim_span_exact(self):
        # 换行缩进下取证必须命中引用所在句, 不得漂移到下节 (回归)
        from paperbrain.verifier import extract_claim
        draft = ("## A\nFirst claim here. [Ref: P, Sec 1]。\n\n"
                 "## B\nSecond statement words.\n   wrapped continuation line here. [Ref: P, Sec 2]。\n\n"
                 "## C\nThird one here. [Ref: P, Sec 3]。")
        import re
        from paperbrain.verifier import CIT_PAT
        ms = list(CIT_PAT.finditer(draft))
        c = extract_claim(draft, ms[1].start(), ms[1].end())
        self.assertIn("continuation", c["sentence"])

    def test_long_section_fully_covered(self):
        from paperbrain.passes import run_passes
        # 30k 字符方法节: 必须全覆盖且预算通过 (原来 [:9000] 静默丢 70%)
        method = "".join(f"Step {i}: we tune model layer {i} with loss {i}. " for i in range(600))
        secs = [{"name": "method", "sec": "2", "text": "Methods\n" + method,
                 "confidence": 0.9, "downgrade_pass1_only": False,
                 "chunks": [{"chunk_id": f"L_Sec2_C{i+1:03d}", "text": method[j:j+1500]}
                            for i, j in enumerate(range(0, len(method), 1500))]}]
        r = run_passes(secs, "L")
        self.assertEqual(r["coverage"], 1.0)
        # 均分归约: 头/中/尾窗口都必须有代表 (原来只取头部)
        for probe in ("Step 0", "Step 313", "Step 558"):
            self.assertIn(probe, r["pass2"] + r["pass4"])

    def test_section_chunk_max_scoring(self):
        from paperbrain.verifier import CitationVerifierV5
        from paperbrain.retrieval import make_scorer
        chunks = [f"filler content about nothing {i} " * 20 for i in range(5)]
        chunks.append("The decisive result is a forty two percent gain on the benchmark.")
        gt = {"P_3": " ".join(chunks)}
        sc = {f"P_Sec3_C{i+1:03d}": c for i, c in enumerate(chunks)}
        v = CitationVerifierV5(gt, {}, semantic_scorer=make_scorer(gt),
                               threshold=0.6, review_threshold=0.5,
                               section_chunks={"P_3": chunks})
        r = v.verify_draft("We report the decisive result of forty two percent gain [Ref: P, Sec 3].")
        self.assertEqual(r["status"], "CLEAN")


if __name__ == "__main__":
    unittest.main()
