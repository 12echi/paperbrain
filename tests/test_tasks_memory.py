"""解读类别与学习记忆测试."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path


class TestTasks(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_labels(self):
        from paperbrain.tasks import task_labels
        ids = [t["id"] for t in task_labels()]
        for want in ("full_read", "outline", "method", "figures", "review", "memory"):
            self.assertIn(want, ids)

    def test_outline_task(self):
        from paperbrain.tasks import run_task
        r = run_task("outline", "demo/sample_paper.txt", "P1", self.out, use_llm=False)
        self.assertEqual(r["task"], "outline")
        self.assertTrue(r["markdown"])
        self.assertIn("速览大纲", r["markdown"])
        self.assertTrue((Path(self.out) / "report.md").exists())

    def test_memory_task(self):
        from paperbrain.tasks import run_task
        r = run_task("memory", "demo/sample_paper.txt", "P2", self.out, use_llm=False)
        self.assertIn("学习记忆", r["markdown"])
        self.assertIn("关键实体", r["markdown"])

    def test_offline_review_does_not_masquerade_as_completed_peer_review(self):
        import json
        from paperbrain.tasks import run_task
        r = run_task("review", "demo/sample_paper.txt", "PR1", self.out, use_llm=False)
        self.assertEqual(r["verify"], "DIRTY")
        self.assertFalse(r["is_clean"])
        self.assertIn("离线规则模式", r["markdown"])
        verify = json.loads((Path(self.out) / "verify.json").read_text(encoding="utf-8"))
        self.assertTrue(any(row["citation"] == "[REVIEW_NOT_GENERATED]"
                            for row in verify["report"]))

    def test_missing_method_does_not_fall_back_to_unrelated_sections(self):
        import json
        from paperbrain.pipeline import generate_from_outline, run_outline
        from paperbrain.tasks import _method_outline
        out = Path(self.out, "missing-method")
        first = run_outline("demo/sample_paper.txt", "NOMETHOD", str(out), use_llm=False)
        no_method = {**first["outline"], "sections": [
            {"h1": "Background", "h2": "Context", "chunks": ["NOMETHOD_Sec1_C001"],
             "claims": ["background only"]},
        ]}

        method_outline = _method_outline(no_method)
        result = generate_from_outline(str(out), "NOMETHOD", method_outline,
                                       semantic_scorer=lambda claim, source: 1.0,
                                       use_llm=False)

        self.assertEqual(len(method_outline["sections"]), 1)
        self.assertEqual(method_outline["sections"][0]["chunks"], [])
        self.assertEqual(result["verify"], "DIRTY")
        self.assertIn("缺少可引用原文", (out / "draft.md").read_text(encoding="utf-8"))
        verify = json.loads((out / "verify.json").read_text(encoding="utf-8"))
        self.assertTrue(any(row["citation"] == "[MISSING_SECTION_EVIDENCE]"
                            for row in verify["report"]))

    def test_full_read_task(self):
        from paperbrain.tasks import run_task
        # standard: 分章纪要 + 深读；深读成果自身也必须给出验真状态。
        r = run_task("full_read", "demo/sample_paper.txt", "P3", self.out, use_llm=False)
        self.assertEqual(r["task"], "full_read")
        self.assertEqual(r["verify"], "DIRTY")
        self.assertFalse(r["is_clean"])
        self.assertTrue(r["markdown"])
        self.assertEqual(r["depth"], "standard")

    def test_full_read_depths(self):
        from paperbrain.tasks import run_task
        fast = run_task("full_read", "demo/sample_paper.txt", "P4F", self.out,
                        use_llm=False, depth="fast")
        self.assertIn("速览大纲", fast["markdown"])
        deep = run_task("full_read", "demo/sample_paper.txt", "P4D", self.out,
                        use_llm=False, depth="deep")
        self.assertEqual(deep["verify"], "AWAITING_CONFIRMATION")
        self.assertTrue(deep["awaiting_confirmation"])
        self.assertIn("sections", deep["outline"])
        self.assertFalse((Path(self.out) / "draft.md").exists())


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_search_delete(self):
        from paperbrain import memory_store as ms
        ms.save("2024_X_01", "full_read", title="质子 RBE 研究",
                summary="DBSCAN 用于 DNA 损伤", key_points=["要点A"],
                entities=[{"name": "DBSCAN", "type": "Algorithm/Model"}])
        ms.save("2024_Y_02", "outline", title="碳离子", summary="LET 谱")
        self.assertEqual(len(ms.search("")), 2)
        self.assertEqual(len(ms.search("DBSCAN")), 1)
        self.assertEqual(len(ms.search("碳离子")), 1)
        got = ms.get("2024_X_01")
        self.assertEqual(got[0]["title"], "质子 RBE 研究")
        # 幂等: 同 paper+task REPLACE
        ms.save("2024_X_01", "full_read", title="改题", summary="x")
        self.assertEqual(len(ms.search("")), 2)
        rid = [m for m in ms.search("") if m["paper_id"] == "2024_X_01"][0]["id"]
        ms.delete(rid)
        self.assertEqual(len(ms.search("")), 1)

    def test_save_from_out(self):
        import json
        from paperbrain import memory_store as ms
        d = self.tmp + "/run"
        Path(d).mkdir()
        (Path(d) / "passes.json").write_text(json.dumps({"pass1": "问题：A。贡献：B。"}), encoding="utf-8")
        (Path(d) / "memory.json").write_text(json.dumps(
            {"entities": [{"name": "Geant4", "type": "Algorithm/Model"}], "relations": []}), encoding="utf-8")
        (Path(d) / "outline_v1.json").write_text(json.dumps({"paper_id": "P"}), encoding="utf-8")
        (Path(d) / "sections.json").write_text(json.dumps([{"text": "Title About Radiation\nmore"}]), encoding="utf-8")
        r = ms.save_from_out(d, "P", "full_read")
        self.assertTrue(r["ok"])
        item = ms.get("P")[0]
        self.assertIn("Geant4", json.dumps(item["entities"], ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
