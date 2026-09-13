"""记忆 v3: 混合检索 RRF / 画像加权 / 写入质检门 / 待裁决 (embedding 以 mock 注入, 不联网)."""
import os
import shutil
import tempfile
import unittest


class TestMemoryV3(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"
        from paperbrain import memory_store as ms
        self.ms = ms
        ms.add_notes("P_A", "full_read", [
            {"kind": "claim", "content": "QoB 可替代 LET 作为统一束流品质度量指标", "anchor": "Sec1"},
            {"kind": "claim", "content": "DBSCAN 用 ε 与最小点数做成团判定", "anchor": "Sec2"},
        ])

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mock_embed(self):
        from paperbrain import embeddings
        orig = (embeddings.available, embeddings.embed_one, embeddings.model)
        # 简易 2 维"语义"向量: 按关键词投影, 使 beam/LET 同向
        def fake_one(t):
            t = (t or "").lower()
            return [1.0 if ("let" in t or "品质" in t or "beam" in t or "qob" in t) else 0.0,
                    1.0 if "dbscan" in t else 0.0]
        embeddings.available = lambda: True
        embeddings.embed_one = fake_one
        embeddings.model = lambda: "mock-emb"
        self.addCleanup(lambda: (setattr(embeddings, "available", orig[0]),
                                 setattr(embeddings, "embed_one", orig[1]),
                                 setattr(embeddings, "model", orig[2])))

    def test_hybrid_rrf_vector_recall(self):
        self._mock_embed()
        # 写入 mock 向量 (直接落库)
        from paperbrain import embeddings
        con = self.ms._conn()
        for nid, content in con.execute("SELECT id, content FROM notes"):
            v = embeddings.embed_one(content)
            con.execute("REPLACE INTO vectors VALUES(?,?,?,?,?,?)",
                        ("note", nid, "mock-emb", len(v), "h", embeddings.to_blob(v)))
        con.commit()
        con.close()
        from paperbrain import vector_store
        original = (vector_store.available, vector_store.count,
                    vector_store.upsert, vector_store.search)
        vector_store.available = lambda: True
        vector_store.count = lambda *a: 1
        vector_store.upsert = lambda *a: {"ok": True, "indexed": 0}
        vector_store.search = lambda model, qv, n, allowed_ids=None: [
            row[0] for row in self.ms._conn().execute(
                "SELECT id FROM notes WHERE content LIKE '%QoB%'").fetchall()]
        self.addCleanup(lambda: (setattr(vector_store, "available", original[0]),
                                 setattr(vector_store, "count", original[1]),
                                 setattr(vector_store, "upsert", original[2]),
                                 setattr(vector_store, "search", original[3])))
        # 查询词与 DBSCAN 笔记无字面交集, 但向量同向 -> 应被召回
        hits = self.ms.search_notes("beam quality metric replacement", limit=2)
        joined = " ".join(h["content"] for h in hits)
        self.assertIn("QoB", joined)

    def test_profile_weight_known_only(self):
        self.ms.build_concepts("P_A")
        hit = self.ms._bump_query_preferences("两篇论文对 LET 与 QoB 的看法")
        low = {h.lower() for h in hit}
        self.assertIn("let", low)
        # 噪声 2-gram 不应入库
        terms = [p["term"] for p in self.ms.list_preferences()]
        self.assertNotIn("两篇", terms)

    def test_write_gate_lexical(self):
        # 主张与原文无重合 -> candidate
        ok = self.ms._lexical_support("量子纠缠可加速蒙特卡洛模拟一万倍", "DBSCAN 聚类用于 DNA 损伤")
        self.assertFalse(ok)
        ok2 = self.ms._lexical_support("DBSCAN 用于 DNA 损伤聚类", "DBSCAN 聚类用于 DNA 损伤")
        self.assertTrue(ok2)

    def test_decide_note(self):
        notes = self.ms.search_notes("DBSCAN")
        nid = notes[0]["id"]
        self.ms._set_status(nid, "candidate")
        self.assertEqual(len(self.ms.pending_decisions("candidate")), 1)
        self.ms.decide_note(nid, True)
        self.assertEqual(len(self.ms.pending_decisions("candidate")), 0)

    def test_contested_status(self):
        notes = self.ms.search_notes("QoB")
        contested_id = notes[0]["id"]
        self.ms._set_status(contested_id, "contested")
        self.assertEqual(len(self.ms.pending_decisions("contested")), 1)
        self.assertEqual(self.ms.memory_stats()["contested_notes"], 1)
        asked = self.ms.ask_memory("QoB", compose=False, expand=False, rerank=False)
        self.assertNotIn(contested_id, asked["used"])


if __name__ == "__main__":
    unittest.main()
