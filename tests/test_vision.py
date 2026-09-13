"""视觉读图测试: 真提取 + mock 解读 (不烧额度)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestVisionExtract(unittest.TestCase):
    def test_borderless_table_falls_back_to_text_strategy(self):
        from paperbrain.vision import extract_tables
        with tempfile.TemporaryDirectory() as td:
            import fitz
            pdf = Path(td, "borderless.pdf")
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((60, 60), "Table 1. Results")
            rows = [("Method", "Accuracy", "Latency"), ("A", "91.0", "12"),
                    ("B", "93.0", "10"), ("C", "95.0", "9")]
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    page.insert_text((60 + j * 140, 100 + i * 25), value)
            doc.save(str(pdf))
            doc.close()
            tables = extract_tables(str(pdf))
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["detection_strategy"], "text")
        self.assertEqual((tables[0]["rows"], tables[0]["cols"]), (4, 3))
        self.assertEqual(tables[0]["cells"][0], ["Method", "Accuracy", "Latency"])
        self.assertEqual(tables[0]["cells"][-1], ["C", "95.0", "9"])
        self.assertNotIn("Table 1. Results", tables[0]["markdown"])
        self.assertIn("Accuracy", tables[0]["markdown"])

    def test_caption_anchor_does_not_absorb_body_prose_into_borderless_table(self):
        from paperbrain.vision import _find_captions_on_page, extract_tables
        with tempfile.TemporaryDirectory() as td:
            import fitz
            pdf = Path(td, "anchored.pdf")
            doc = fitz.open()
            page = doc.new_page(width=600, height=800)
            page.insert_text((50, 175), "As shown in Fig. 1, accuracy rises by ten points.")
            page.insert_text((70, 365), "Fig. 1. Accuracy by method")
            page.insert_text((70, 420), "Table 1. Results")
            rows = [("Method", "Accuracy", "Latency"), ("A", "91.0", "12"),
                    ("B", "95.0", "9")]
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    page.insert_text((70 + j * 140, 450 + i * 24), value)
            page.insert_text((50, 560),
                             "Table 1 reports the exact values used in the comparison.")
            doc.save(str(pdf))
            doc.close()
            with fitz.open(str(pdf)) as opened:
                fig_caps = _find_captions_on_page(opened[0], kind="figure")
            tables = extract_tables(str(pdf))
        self.assertEqual([c["text"] for c in fig_caps], ["Fig. 1. Accuracy by method"])
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["table_id"], "tab1")
        self.assertEqual(tables[0]["caption"], "Table 1. Results")
        self.assertEqual(tables[0]["cells"], [list(row) for row in rows])

    def test_two_borderless_tables_on_one_page_do_not_merge(self):
        from paperbrain.vision import extract_tables
        with tempfile.TemporaryDirectory() as td:
            import fitz
            pdf = Path(td, "two-tables.pdf")
            doc = fitz.open()
            page = doc.new_page(width=600, height=800)
            expected = []
            for number, top, values in ((1, 70, [("A", "B"), ("1", "2")]),
                                        (2, 250, [("C", "D"), ("3", "4")])):
                page.insert_text((60, top), f"Table {number}. Results {number}")
                for i, row in enumerate(values):
                    for j, value in enumerate(row):
                        page.insert_text((60 + j * 160, top + 35 + i * 24), value)
                expected.append([list(row) for row in values])
            doc.save(str(pdf))
            doc.close()
            tables = extract_tables(str(pdf))
        self.assertEqual([t["table_id"] for t in tables], ["tab1", "tab2"])
        self.assertEqual([t["cells"] for t in tables], expected)

    def test_ruled_table_does_not_disable_borderless_table_on_same_page(self):
        from paperbrain.vision import extract_tables
        with tempfile.TemporaryDirectory() as td:
            import fitz
            pdf = Path(td, "mixed-tables.pdf")
            doc = fitz.open()
            page = doc.new_page(width=600, height=800)
            page.insert_text((60, 50), "Table 1. Ruled")
            # 2x2 ruled table from y=75..145.
            for y in (75, 110, 145):
                page.draw_line((60, y), (380, y), color=(0, 0, 0), width=1)
            for x in (60, 220, 380):
                page.draw_line((x, 75), (x, 145), color=(0, 0, 0), width=1)
            for x, y, value in ((75, 98, "A"), (235, 98, "B"),
                                (75, 133, "1"), (235, 133, "2")):
                page.insert_text((x, y), value)
            page.insert_text((60, 240), "Table 2. Borderless")
            rows = [("C", "D"), ("3", "4"), ("5", "6")]
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    page.insert_text((60 + j * 160, 275 + i * 24), value)
            doc.save(str(pdf))
            doc.close()
            tables = extract_tables(str(pdf))
        by_id = {table["table_id"]: table for table in tables}
        self.assertEqual(set(by_id), {"tab1", "tab2"})
        self.assertEqual(by_id["tab2"]["cells"], [list(row) for row in rows])
        self.assertEqual(by_id["tab2"]["detection_strategy"], "text")

    def test_pipeline_m1_prediction_keeps_real_cells_and_body_binding_without_table_ocr(self):
        from paperbrain.pipeline import run_outline
        from paperbrain.preflight import preflight
        with tempfile.TemporaryDirectory() as td:
            import fitz
            root = Path(td)
            image = root / "figure.png"
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 300), False)
            pix.clear_with(0x5588CC)
            pix.save(str(image))
            pdf, out, pred = root / "paper.pdf", root / "out", root / "pred.json"
            doc = fitz.open()
            page = doc.new_page(width=600, height=800)
            page.insert_text((50, 50), "3 Experiments")
            body = "As shown in Fig. 1, accuracy rises by ten points on the held-out set."
            page.insert_text((50, 85), body)
            page.insert_image(fitz.Rect(80, 120, 480, 420), filename=str(image))
            page.insert_text((80, 445), "Fig. 1. Accuracy by method")
            page.insert_text((60, 500), "Table 1. Results")
            rows = [("Method", "Score"), ("A", "91"), ("B", "95")]
            for i, row in enumerate(rows):
                for j, value in enumerate(row):
                    page.insert_text((60 + j * 180, 530 + i * 24), value)
            doc.save(str(pdf))
            doc.close()
            self.assertEqual(preflight(str(pdf), "P").route, "hybrid")
            with patch("paperbrain.ocr.ocr_pdf") as ocr_call:
                run_outline(str(pdf), "P", str(out), use_llm=False, vision=False,
                            m1_pred_path=str(pred))
                ocr_call.assert_not_called()
            record = json.loads(pred.read_text(encoding="utf-8"))
        self.assertEqual(record["tables"][0]["cells"], [list(row) for row in rows])
        self.assertEqual(record["captions"][0]["caption"], "Fig. 1. Accuracy by method")
        self.assertEqual(record["captions"][0]["bound_contexts"], [body])

    def test_extract_from_image_pdf(self):
        from paperbrain.vision import extract_images
        with tempfile.TemporaryDirectory() as td:
            try:
                import fitz
            except ImportError:
                self.skipTest("缺 PyMuPDF")
            img = Path(td, "figure.png")
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 400), False)
            pix.clear_with(0x6699CC)
            pix.save(str(img))
            pdf = Path(td, "embedded.pdf")
            doc = fitz.open()
            page = doc.new_page(width=600, height=800)
            page.insert_image(fitz.Rect(50, 80, 450, 480), filename=str(img))
            page.insert_text((50, 510), "Fig. 1 Local scientific result")
            doc.save(str(pdf))
            doc.close()
            figs = extract_images(str(pdf), td, max_images=3)
            self.assertGreaterEqual(len(figs), 1)
            self.assertTrue(Path(figs[0]["image"]).exists())
            self.assertGreater(figs[0]["pixels"], 10000)

    def test_full_page_scan_is_not_treated_as_uploadable_figure(self):
        from paperbrain.vision import extract_images
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(extract_images("demo/scanned_sample.pdf", td, max_images=3), [])

    def test_extract_txt_pdf_empty(self):
        from paperbrain.vision import extract_images
        with tempfile.TemporaryDirectory() as td:
            figs = extract_images("demo/sample_paper.pdf", td)
            self.assertEqual(figs, [])

    def test_describe_parses_stream(self):
        from paperbrain import vision
        from paperbrain import llm
        fixture = ('{"type":"step_start","sessionID":"ses_x"}\n'
                   '{"type":"text","sessionID":"ses_x","part":{"type":"text","text":"两条曲线"}}\n')

        class R:
            returncode = 0
            stdout = fixture
            stderr = ""
        llm.reset_usage()
        with patch.dict(os.environ, {"PAPERBRAIN_CLOUD_ALLOWED": "1"}):
            with patch.object(vision, "_prepare_privacy_safe_png",
                              return_value="/tmp/pb-safe-fake.png"):
                with patch.object(vision.subprocess, "run", return_value=R()):
                    with patch.object(vision, "_cleanup_session") as cl:
                        r = vision.describe_figure("/tmp/fake.png", model="m/x")
                        self.assertIn("两条曲线", r["analysis"])
                        self.assertTrue(r["metadata_stripped"])
                        cl.assert_called_once_with("ses_x")
        usage = llm.usage_snapshot()
        self.assertEqual(usage["calls"], 1)
        self.assertGreater(usage["by_stage"]["vision_text"], 0)
        llm.reset_usage()

    def test_describe_failure_closed(self):
        from paperbrain import vision

        class R:
            returncode = 1
            stdout = ""
            stderr = "boom"
        with patch.dict(os.environ, {"PAPERBRAIN_CLOUD_ALLOWED": "1"}):
            with patch.object(vision, "_prepare_privacy_safe_png",
                              return_value="/tmp/pb-safe-fake.png"):
                with patch.object(vision.subprocess, "run", return_value=R()):
                    with patch.object(vision, "_cleanup_session"):
                        with self.assertRaises(RuntimeError):
                            vision.describe_figure("/tmp/fake.png", model="m/x")

    def test_vision_gating(self):
        from paperbrain import vision
        os.environ.pop("PAPERBRAIN_VISION", None)
        os.environ.pop("PAPERBRAIN_CLOUD_ALLOWED", None)
        self.assertFalse(vision.vision_enabled())
        os.environ["PAPERBRAIN_VISION"] = "1"
        try:
            self.assertFalse(vision.vision_enabled())
            os.environ["PAPERBRAIN_CLOUD_ALLOWED"] = "1"
            self.assertTrue(vision.vision_enabled())
        finally:
            os.environ.pop("PAPERBRAIN_VISION", None)
            os.environ.pop("PAPERBRAIN_CLOUD_ALLOWED", None)

    def test_describe_requires_separate_cloud_authorization(self):
        from paperbrain import vision
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PAPERBRAIN_CLOUD_ALLOWED", None)
            with self.assertRaisesRegex(RuntimeError, "未授权"):
                vision.describe_figure("/tmp/does-not-matter.png", model="m/x")

    def test_sixth_effective_vision_send_is_refused_before_subprocess(self):
        from paperbrain import llm, vision
        llm.reset_usage()
        for _ in range(5):
            llm.record_external_call("vision_text", "prompt", "result", vision_images=1)
        with patch.dict(os.environ, {"PAPERBRAIN_CLOUD_ALLOWED": "1"}):
            with patch.object(vision, "_prepare_privacy_safe_png",
                              return_value="/tmp/pb-safe-fake.png"), \
                    patch.object(vision.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "预算超限"):
                    vision.describe_figure("/tmp/fake.png", model="m/x", retries=0)
                run.assert_not_called()
        llm.reset_usage()


if __name__ == "__main__":
    unittest.main()
