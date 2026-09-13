"""知识网络 (原子笔记+链接+召回) 测试."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path


class TestNotes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_links_and_recall(self):
        from paperbrain import memory_store as ms
        ms.add_notes("P1", "full_read", [
            {"kind": "核心洞见", "content": "DBSCAN 聚类用于 DNA 损伤能量沉积建模", "source": "洞见"},
            {"kind": "局限", "content": "DBSCAN 阈值 17.5 eV 的普适性未验证", "source": "局限"},
            {"kind": "entity", "content": "Geant4 蒙特卡洛径迹模拟", "source": "entity"},
        ])
        notes = ms.search_notes("")
        self.assertGreaterEqual(len(notes), 3)
        # DBSCAN 两条应互链
        dbscan = [n for n in notes if "DBSCAN" in n["content"]]
        self.assertTrue(any(n["links"] for n in dbscan))
        r = ms.recall("DBSCAN 聚类", k=3)
        self.assertTrue(r["seeds"])

    def test_build_notes_from_out(self):
        from paperbrain import memory_store as ms
        d = Path(self.tmp) / "run"
        d.mkdir()
        (d / "deepread.md").write_text(
            "# 深度解读 · P\n\n## 一句话主张\n它主张 X 能降低内存。\n\n"
            "## 局限与威胁效度\n单点结果无显著性。\n", encoding="utf-8")
        (d / "deepread_synthesis.json").write_text(json.dumps(
            {"argument_map": [{"claim": "更快", "evidence": "2点", "limitation": "无方差"}]},
            ensure_ascii=False), encoding="utf-8")
        (d / "memory.json").write_text(json.dumps(
            {"entities": [{"name": "FlashAttention-v2", "type": "Algorithm/Model"}]},
            ensure_ascii=False), encoding="utf-8")
        res = ms.build_notes_from_out(str(d), "P", "full_read")
        self.assertGreaterEqual(res["added"], 3)
        # 无原文上下文时模型生成主张必须降为 candidate：可审计存在，但默认检索不可见。
        kinds = {n["kind"] for n in ms.search_notes("", include_pending=True)}
        self.assertIn("claim", kinds)
        self.assertIn("entity", kinds)
        entity = next(n for n in ms.search_notes("", include_pending=True)
                      if n["kind"] == "entity")
        self.assertEqual(entity["status"], "candidate")
        default_kinds = {n["kind"] for n in ms.search_notes("")}
        self.assertNotIn("claim", default_kinds)
        self.assertNotIn("entity", default_kinds)


if __name__ == "__main__":
    unittest.main()
