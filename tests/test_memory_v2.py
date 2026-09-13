"""记忆层 v2 测试: FTS5 检索 / 别名消解 / 巩固去重 / 问记忆 / 写时记忆卡回退."""
import os
import shutil
import tempfile
import unittest


class TestMemoryV2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"
        from paperbrain import memory_store as ms
        self.ms = ms
        ms.add_notes("P_A", "full_read", [
            {"kind": "核心洞见", "content": "DBSCAN 聚类把能量沉积点成团, 对应 DNA 双链断裂损伤 DSB", "anchor": "Sec2"},
            {"kind": "局限", "content": "17.5 eV 阈值引自文献未做敏感性分析", "anchor": "Sec3"},
        ])
        ms.add_notes("P_B", "full_read", [
            {"kind": "核心洞见", "content": "束流品质用归一化 QoB 表征, 与 DSB 产额线性相关", "anchor": "Sec1"},
        ])

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fts_cjk_two_char(self):
        # 2 字中文词必须能命中 (bigram 扩展)
        hits = self.ms.search_notes("损伤")
        self.assertTrue(any("损伤" in h["content"] or "DSB" in h["content"] for h in hits))
        hits2 = self.ms.search_notes("阈值")
        self.assertTrue(hits2)

    def test_fts_english_and_ranking_field(self):
        hits = self.ms.search_notes("DBSCAN")
        self.assertTrue(hits)
        self.assertIn("score", hits[0])  # BM25 排序分

    def test_alias_resolution(self):
        self.ms.add_alias("DNA 双链断裂", "DSB")
        self.assertEqual(self.ms.resolve_alias("DNA 双链断裂"), "DSB")
        self.ms.build_concepts("P_A")
        import sqlite3
        con = sqlite3.connect(self.ms.db_path())
        got = [r[0] for r in con.execute("SELECT concept FROM concepts WHERE paper_id='P_A'")]
        con.close()
        self.assertIn("DSB", got)

    def test_consolidate_dedup(self):
        self.ms.add_notes("P_A", "dup", [
            {"kind": "claim", "content": "重复主张一模一样的内容占位"},
            {"kind": "claim", "content": "重复主张一模一样的内容占位"},
            {"kind": "claim", "content": "重复主张一模一样的内容占位"},
        ])
        r = self.ms.consolidate("P_A")
        self.assertGreaterEqual(r["removed"], 2)
        kinds = [n["content"] for n in self.ms.search_notes("重复主张")]
        self.assertEqual(len(kinds), 1)

    def test_ask_memory_offline_digest(self):
        r = self.ms.ask_memory("DBSCAN 是怎么定义损伤的", compose=False)
        self.assertIn("命", r["answer"])
        self.assertTrue(r["used"])

    def test_memory_card_missing_ctx(self):
        d = tempfile.mkdtemp()
        try:
            r = self.ms.memory_card_from_out(d, "P_X")
            self.assertFalse(r.get("ok"))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_memory_stats(self):
        st = self.ms.memory_stats()
        self.assertGreaterEqual(st["notes"], 3)
        self.assertIn("aliases", st)


if __name__ == "__main__":
    unittest.main()
