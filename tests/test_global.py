"""全局层测试: 中央配置 / 全局上下文 / 跨论文知识网络."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path


class TestConfig(unittest.TestCase):
    def test_reads_env_runtime(self):
        from paperbrain import config
        os.environ["PAPERBRAIN_PROVIDER"] = "opencode-cli"
        os.environ["PAPERBRAIN_ALL_MODEL"] = "1"
        os.environ["PAPERBRAIN_CONTEXT_CHARS"] = "1234"
        try:
            self.assertEqual(config.provider(), "opencode-cli")
            self.assertTrue(config.all_model())
            self.assertEqual(config.context_chars(), 1234)
        finally:
            for k in ("PAPERBRAIN_PROVIDER", "PAPERBRAIN_ALL_MODEL", "PAPERBRAIN_CONTEXT_CHARS"):
                os.environ.pop(k, None)
        self.assertEqual(config.provider(), "https")

    def test_flags(self):
        from paperbrain import config
        os.environ.pop("PAPERBRAIN_VISION", None)
        self.assertFalse(config.vision_enabled())
        os.environ["PAPERBRAIN_VISION"] = "true"
        try:
            self.assertTrue(config.vision_enabled())
        finally:
            os.environ.pop("PAPERBRAIN_VISION", None)


class TestContext(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        from paperbrain.pipeline import run_outline
        run_outline("demo/sample_paper.txt", "CTX01", self.out, use_llm=False)

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_section_markers_and_entities(self):
        from paperbrain.context import build_paper_context
        ctx = build_paper_context(self.out, budget=4000)
        self.assertIn("### Sec", ctx)
        self.assertIn("abstract", ctx.lower())
        self.assertLessEqual(len(ctx), 4000 + 800)

    def test_focus_pulls_relevant(self):
        # 造一个长方法节, 关注句埋在尾部, 小预算下也应被优先带入
        d = tempfile.mkdtemp()
        try:
            secs = [
                {"name": "method", "sec": "2",
                 "text": ("无关内容。" * 400) + "本文采用 tiling 注意力机制降低显存占用。" + ("废话。" * 400),
                 "confidence": 0.9, "downgrade_pass1_only": False,
                 "chunks": [{"chunk_id": "P_Sec2_C001", "text": "x"}]},
            ]
            Path(d, "sections.json").write_text(json.dumps(secs, ensure_ascii=False), encoding="utf-8")
            Path(d, "memory.json").write_text(json.dumps({"entities": []}), encoding="utf-8")
            from paperbrain.context import build_paper_context
            ctx = build_paper_context(d, budget=300, focus="tiling 注意力如何降低显存")
            self.assertIn("tiling", ctx)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestField(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cross_paper_network(self):
        from paperbrain import memory_store as ms
        ms.add_notes("P_A", "full_read", [
            {"kind": "洞见", "content": "DBSCAN 聚类用于定量 DNA 损伤", "source": "x"},
            {"kind": "entity", "content": "Geant4 蒙特卡洛模拟", "source": "e"}])
        ms.add_notes("P_B", "full_read", [
            {"kind": "洞见", "content": "DBSCAN 阈值敏感性分析 DNA 损伤", "source": "x"}])
        ms.build_concepts("P_A")
        ms.build_concepts("P_B")
        r = ms.link_papers(min_shared=1)
        self.assertGreaterEqual(r["edges"], 1)
        fm = ms.field_map()
        self.assertTrue(fm["edges"])
        terms = [c["term"].lower() for c in fm["shared_concepts"]]
        self.assertTrue(any("dbscan" in t for t in terms))
        self.assertTrue(fm["clusters"])


if __name__ == "__main__":
    unittest.main()
