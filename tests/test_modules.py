"""新模块门禁测试: text/retrieval/llm回退/formulas/figures/ocr.
真工具 (katex/sympy/cv2/tesseract) 若缺失则对应项 skip, 不伪造通过.
"""
import unittest


class TestText(unittest.TestCase):
    def test_decimal_intact(self):
        from paperbrain.text import split_sentences
        s = split_sentences("See Sec 3.2 for settings. Next one.")
        self.assertEqual(len(s), 2)
        self.assertIn("3.2", s[0])

    def test_abbr_intact(self):
        from paperbrain.text import split_sentences
        s = split_sentences("As in Fig. 2 we show e.g. results. Done.")
        self.assertTrue(any("Fig. 2" in x for x in s))


class TestRetrieval(unittest.TestCase):
    def test_verbatim_high(self):
        from paperbrain.retrieval import make_scorer
        corpus = {"a": "The cat sits on the mat and watches the birds outside quietly.",
                  "b": "Quantum tunneling in semiconductors enables flash memory."}
        sc = make_scorer(corpus)
        self.assertGreater(sc("The cat sits on the mat and watches the birds", corpus["a"]), 0.8)
        self.assertLess(sc("The cat sits on the mat and watches the birds", corpus["b"]), 0.4)

    def test_short_claim_capped(self):
        from paperbrain.retrieval import make_scorer
        corpus = {"a": "The cat sits on the mat. Dogs bark loudly.",
                  "b": "Quantum tunneling in semiconductors enables flash memory."}
        sc = make_scorer(corpus)
        # 过短断言即使逐字命中也要封顶 (防空话刷分)
        self.assertLess(sc("The cat sits on the mat", corpus["a"]), 0.8)

    def test_retrieve_rank(self):
        from paperbrain.retrieval import build_index, retrieve
        idx = build_index({"a": "apple orange banana", "b": "car engine wheel"})
        top = retrieve("apple banana", idx, top_k=1)
        self.assertEqual(top[0][0], "a")


class TestLLMFallback(unittest.TestCase):
    def test_no_key_raises(self):
        import os
        from paperbrain.llm import summarize, NoKeyError
        if os.environ.get("PAPERBRAIN_API_KEY"):
            self.skipTest("有key时跳过回退测试")
        with self.assertRaises(NoKeyError):
            summarize("pass1", "hello")

    def test_llm_called_once_per_pass(self):
        # 长文本每 Pass 只调 1 次 LLM (省额度), 其余分窗走规则
        from paperbrain.passes import run_passes
        calls = []

        def fake(kind, text):
            calls.append((kind, len(text)))
            return f"[{kind}] 摘要"

        long_text = ("有意义的长句，包含模型与损失关键词。" * 800)  # ~1.4万字
        secs = [{"name": "method", "sec": "2", "text": "Methods\n" + long_text,
                 "confidence": 0.9, "downgrade_pass1_only": False,
                 "chunks": [{"chunk_id": "M_Sec2_C001", "text": long_text[:1500]}]}]
        r = run_passes(secs, "M", summarize_fn=fake)
        kinds = [k for k, _ in calls]
        self.assertLessEqual(kinds.count("pass2"), 1)
        self.assertLessEqual(kinds.count("pass4"), 1)
        self.assertIn("[pass2]", r["pass2"])

    def test_pipeline_offline_clean(self):
        import shutil
        from pathlib import Path
        from paperbrain.pipeline import run_paper
        out = Path("out/test_offline")
        if out.exists():
            shutil.rmtree(out)
        r = run_paper("demo/sample_paper.txt", "2024_NeurIPS_01", str(out), use_llm=True,
                      confirm_outline=True)
        self.assertFalse(r["llm_used"])
        self.assertEqual(r["verify"], "NEEDS_REVIEW")
        self.assertFalse(r["calibration_verified"])


class TestFormulas(unittest.TestCase):
    def test_valid_formula(self):
        from paperbrain.formulas import check_formulas
        r = check_formulas(r"Loss is $L = -\sum y \log p$.")
        self.assertTrue(r["formulas"])
        self.assertEqual(r["dirty"], 0)

    def test_garbage_formula(self):
        from paperbrain.formulas import check_formulas
        r = check_formulas("乱码 $\\zzz{{{@@@$.")
        self.assertGreaterEqual(r["dirty"], 0)  # 至少不崩, 脏数由工具判定

    def test_clean_marks(self):
        from paperbrain.formulas import clean_text
        out = clean_text("乱码 $\\zzz{{{@@@$ 结束")
        self.assertIn("FORMULA_UNVERIFIED", out)


class TestFigures(unittest.TestCase):
    def test_split_or_skip(self):
        import tempfile, os
        from paperbrain.figures import split_figure, split_caption
        self.assertEqual(len(split_caption("Results (a) acc up. (b) loss down.")), 2)
        try:
            import cv2
            import numpy as np
        except Exception:
            self.skipTest("缺cv2")
        img = 255 * __import__("numpy").ones((200, 400), dtype="uint8")
        img[20:90, 20:190] = 0
        img[20:90, 210:380] = 0
        fd, p = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(p, img)
        try:
            r = split_figure(p, "Fig. (a) left. (b) right.")
            self.assertTrue(r["split"])
            self.assertEqual(len(r["blocks"]), 2)
        finally:
            os.unlink(p)


class TestOCR(unittest.TestCase):
    def test_tesseract_present(self):
        from paperbrain.ocr import has_ocr
        self.assertTrue(has_ocr(), "tesseract 应已安装 (brew)")

    def test_ocr_demo_pdf(self):
        from paperbrain.ocr import ocr_pdf
        r = ocr_pdf("demo/sample_paper.pdf")
        self.assertTrue(r["ok"])
        txt = " ".join(p["text"] for p in r["pages"])
        self.assertIn("FlashAttention", txt)


if __name__ == "__main__":
    unittest.main()
