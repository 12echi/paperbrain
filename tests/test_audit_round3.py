"""第三轮审计回归测试 (critic findings 固化): 幽灵正例 / 悬空链 / 删除安全 / 质检门 / 熔断 / 配置缓存。"""
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
import urllib.error

import paperbrain.llm as llm


class _DB:
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"
        from paperbrain import memory_store as ms
        self.ms = ms

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _con(self):
        return sqlite3.connect(self.tmp + "/mem.sqlite")


class TestGcAndPurge(_DB, unittest.TestCase):
    def test_gc_prunes_dangling_and_orphans(self):
        a = self.ms.add_notes("P", "t", [{"kind": "claim", "content": "AAA 记忆甲内容"}])["ids"][0]
        b = self.ms.add_notes("P", "t", [{"kind": "claim", "content": "BBB 记忆乙内容"}])["ids"][0]
        con = self._con()
        con.execute("UPDATE notes SET links=? WHERE id=?", (json.dumps([b, 99999]), a))
        con.execute("INSERT OR REPLACE INTO vectors VALUES('note',99999,'m',2,'h',x'00')")
        con.execute("INSERT OR REPLACE INTO note_links VALUES(99999,?, 'supports')", (a,))
        con.commit()
        con.close()
        r = self.ms.gc_memory()
        self.assertGreaterEqual(r["links_pruned"], 1)
        self.assertGreaterEqual(r["vectors_removed"], 1)
        self.assertGreaterEqual(r["note_links_removed"], 1)
        con = self._con()
        links = json.loads(con.execute("SELECT links FROM notes WHERE id=?", (a,)).fetchone()[0])
        con.close()
        self.assertEqual(links, [b])  # 仅剔除死 id

    def test_delete_note_cleans_refs_and_vectors(self):
        a = self.ms.add_notes("P", "t", [{"kind": "claim", "content": "CCC 待删笔记内容"}])["ids"][0]
        b = self.ms.add_notes("P", "t", [{"kind": "claim", "content": "DDD 引用者内容" + "CC" * 3}])["ids"][0]
        con = self._con()
        con.execute("UPDATE notes SET links=? WHERE id=?", (json.dumps([a]), b))
        con.execute("INSERT OR REPLACE INTO vectors VALUES('note',?,'m',2,'h',x'00')", (a,))
        con.commit()
        con.close()
        self.assertTrue(self.ms.delete_note(a)["ok"])
        con = self._con()
        links = json.loads(con.execute("SELECT links FROM notes WHERE id=?", (b,)).fetchone()[0])
        vecs = con.execute("SELECT COUNT(*) FROM vectors WHERE owner_id=?", (a,)).fetchone()[0]
        con.close()
        self.assertEqual(links, [])
        self.assertEqual(vecs, 0)

    def test_invalidate_missing_returns_not_ok(self):
        self.assertFalse(self.ms.invalidate_note(99999999)["ok"])


class TestDeleteSafety(_DB, unittest.TestCase):
    def test_build_notes_failure_preserves_old(self):
        self.ms.add_notes("P1", "t", [{"kind": "claim", "content": "旧笔记必须保留"}])
        out = self.tmp + "/out"
        os.makedirs(out, exist_ok=True)
        with open(out + "/deepread.md", "w", encoding="utf-8") as f:
            f.write("## 一句话结论\n新结论内容很长足够入库\n## 核心洞见\n新洞见内容也很长足够\n")
        real = self.ms.add_notes

        def boom(*a, **k):
            raise sqlite3.OperationalError("simulated failure")
        self.ms.add_notes = boom
        try:
            with self.assertRaises(Exception):
                self.ms.build_notes_from_out(out, "P1", "full_read")
        finally:
            self.ms.add_notes = real
        con = self._con()
        n = con.execute("SELECT COUNT(*) FROM notes WHERE paper_id='P1'").fetchone()[0]
        con.close()
        self.assertEqual(n, 1, "add_notes 失败时旧笔记必须保留 (先插后删事务)")


class TestReflectionGate(_DB, unittest.TestCase):
    def setUp(self):
        super().setUp()
        for i in range(6):
            self.ms.add_notes("P1", "t", [{"kind": "claim", "content": f"记忆条目 {i}: 阈值与损伤"}])

    def test_no_evidence_insight_is_candidate(self):
        orig, origv = llm.reflect_notes, llm.verify_claims
        try:
            llm.reflect_notes = lambda ctx, n=3: '[{"insight":"没有证据的断言","evidence":[]}]'
            llm.verify_claims = lambda t, c: (_ for _ in ()).throw(AssertionError("不应调用验证"))
            r = self.ms.reflect("P1")
            self.assertTrue(r["ok"])
            self.assertEqual(r["verified"], 0)
            con = self._con()
            st = con.execute("SELECT status FROM notes WHERE kind='反思' ORDER BY id DESC LIMIT 1").fetchone()[0]
            con.close()
            self.assertEqual(st, "candidate", "无证据洞见不得 active")
        finally:
            llm.reflect_notes, llm.verify_claims = orig, origv


class TestVerifyClaimsOffset(_DB, unittest.TestCase):
    def test_complete_0based_and_1based(self):
        ctx = "DBSCAN 聚类 用于 能谱"
        llm.verify_claims = lambda t, c: json.dumps([{"i": 0, "ok": True}, {"i": 1, "ok": False}])
        self.assertEqual(self.ms._verify_claims(["甲", "乙"], ctx), [True, False])
        llm.verify_claims = lambda t, c: json.dumps([{"i": 1, "ok": True}, {"i": 2, "ok": True}])
        self.assertEqual(self.ms._verify_claims(["甲", "乙"], ctx), [True, True])

    def test_incomplete_not_shifted(self):
        ctx = "DBSCAN 聚类 用于 能谱"
        llm.verify_claims = lambda t, c: json.dumps([{"i": 1, "ok": True}])
        got = self.ms._verify_claims(["甲", "乙"], ctx)
        expected = [self.ms._lexical_support("甲", ctx), self.ms._lexical_support("乙", ctx)]
        self.assertEqual(got, expected, "不完整响应不得做 1-based 偏移推断")

    def test_lexical_support_empty_keywords_false(self):
        self.assertFalse(self.ms._lexical_support("  ", "任何原文"))

    def tearDown(self):
        import paperbrain.llm as _l
        _l.verify_claims = getattr(_l, "verify_claims", _l.verify_claims)
        super().tearDown()


class TestBatchSuppressionGuard(_DB, unittest.TestCase):
    def test_non_dict_item_no_leak(self):
        from paperbrain.batch import run_batch
        r = run_batch([42], self.tmp + "/bo", use_llm=False)
        self.assertFalse(self.ms._FIELD_REFLECT_SUPPRESS)
        self.assertEqual(r["failed"], 1)  # 非法输入按失败记录, 不抛穿


class TestEvalPosValidation(_DB, unittest.TestCase):
    def test_build_cases_filters_ghost_ids(self):
        a = self.ms.add_notes("P", "t", [{"kind": "claim",
                                          "content": "GHOSTTEST 悬空引用 测试 内容足够长"}])["ids"][0]
        con = self._con()
        con.execute("UPDATE notes SET links=? WHERE id=?", (json.dumps([99999, a]), a))
        con.commit()
        con.close()
        import tools.eval_retrieval as ev
        con = self.ms._conn()
        con.row_factory = sqlite3.Row
        cases = ev._build_cases(con, 5)
        con.close()
        for c in cases:
            self.assertNotIn(99999, c["pos"])


class TestEmbedHttpErrors(unittest.TestCase):
    def setUp(self):
        os.environ["PAPERBRAIN_EMBED_BASE_URL"] = "http://127.0.0.1:1/v1"
        os.environ["PAPERBRAIN_EMBED_API_KEY"] = "sk-test"
        from paperbrain import embeddings
        self.emb = embeddings
        embeddings._reset_breaker()
        self.real = embeddings.urllib.request.urlopen

    def tearDown(self):
        self.emb.urllib.request.urlopen = self.real
        self.emb._reset_breaker()
        os.environ.pop("PAPERBRAIN_EMBED_BASE_URL", None)
        os.environ.pop("PAPERBRAIN_EMBED_API_KEY", None)

    def _raise(self, code):
        def f(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, "err", {}, None)
        self.emb.urllib.request.urlopen = f

    def test_4xx_does_not_trip_breaker(self):
        self._raise(400)
        out = self.emb.embed_texts(["超长或模型名错误的情形"])
        self.assertTrue(all(v is None for v in out))
        self.assertFalse(self.emb.breaker_open(), "4xx 是客户端问题, 不得熔断")

    def test_5xx_trips_breaker(self):
        self._raise(503)
        self.emb.embed_texts(["服务端故障"])
        self.assertTrue(self.emb.breaker_open(), "5xx 应熔断保护")


if __name__ == "__main__":
    unittest.main()
