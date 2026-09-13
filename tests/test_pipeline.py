"""Pipeline 端到端冒烟（未完成 50 对标定时必须阻止 CLEAN）。"""
import shutil
import unittest
from pathlib import Path

from paperbrain.pipeline import run_paper


class TestPipeline(unittest.TestCase):
    def test_demo_requires_calibration_review(self):
        out = Path("out/test_smoke")
        if out.exists():
            shutil.rmtree(out)
        r = run_paper("demo/sample_paper.txt", "2024_NeurIPS_01", str(out),
                      confirm_outline=True)
        self.assertTrue(r["budget_ok"])
        self.assertEqual(r["verify"], "NEEDS_REVIEW")
        self.assertFalse(r["is_clean"])
        self.assertFalse(r["calibration_verified"])
        for fn in ["sections.json", "passes.json", "memory.json", "outline_v1.json",
                   "draft.md", "verify.json", "ledger.csv", "report.md", "paperbrain.db"]:
            self.assertTrue((out / fn).exists(), fn)

    def test_scanned_pdf_ocr_recovered(self):
        out = Path("out/test_scan")
        if out.exists():
            shutil.rmtree(out)
        r = run_paper("demo/scanned_sample.pdf", "scan01", str(out),
                      confirm_outline=True)
        self.assertEqual(r["route"], "ocr_recovered")
        self.assertEqual(r["verify"], "NEEDS_REVIEW")

    def test_default_run_stops_before_unconfirmed_generation(self):
        out = Path("out/test_unconfirmed")
        if out.exists():
            shutil.rmtree(out)
        r = run_paper("demo/sample_paper.txt", "UNCONFIRMED", str(out), use_llm=False)
        self.assertEqual(r["verify"], "AWAITING_CONFIRMATION")
        self.assertTrue(r["awaiting_confirmation"])
        self.assertFalse((out / "draft.md").exists())


if __name__ == "__main__":
    unittest.main()
