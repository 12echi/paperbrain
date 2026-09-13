"""学习方法 v3 测试: 三因子检索(重要性/近因) + 访问强化 + 反思层 + PPR 多跳 (模型 mock)."""
import json
import os
import shutil
import tempfile
import unittest

import paperbrain.llm as llm


class TestThreeFactor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"
        from paperbrain import memory_store as ms
        self.ms = ms

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_importance_by_kind(self):
        self.assertGreater(self.ms._importance("核心洞见", "x"), self.ms._importance("entity", "x"))
        self.assertGreater(self.ms._importance("局限与威胁效度", "x"), self.ms._importance("note", "x"))
        self.assertGreater(self.ms._importance("证据强度", "提升 2 点"), self.ms._importance("证据强度", "提升"))

    def test_recency_decay(self):
        import time as _t
        now = _t.time()
        fresh = self.ms._recency(now, now)
        old = self.ms._recency(now - 3600 * 24 * 7, now)  # 7 天前
        self.assertGreater(fresh, old)
        self.assertAlmostEqual(old, 0.5, places=1)

    def test_scoring_factors(self):
        s = self.ms._final_score
        # 同等相关: 重要性/近因越高分越高
        self.assertGreater(s(0.5, 0.9, 0.5), s(0.5, 0.3, 0.5))
        self.assertGreater(s(0.5, 0.5, 0.9), s(0.5, 0.5, 0.1))
        # 相关性占主导: 0.25 的相关优势 > 非相关项最大加成 (0.20)
        self.assertGreater(s(0.95, 0.3, 0.1), s(0.70, 0.95, 1.0))
        # 偏好乘子生效
        self.assertGreater(s(0.5, 0.5, 0.5, 1.2), s(0.5, 0.5, 0.5, 1.0))

    def test_lexical_confidence_gate(self):
        r = self.ms.add_notes("P", "t", [{"kind": "claim",
                                          "content": "DBSCAN 聚类 用于中子能谱 束流品质"}])
        nid = r["ids"][0]
        con = self.ms._conn()
        hi = self.ms._lexical_confidence("DBSCAN 中子能谱 束流品质", [nid], con)
        lo = self.ms._lexical_confidence("为什么吃饭会让人心情变好呢", [nid], con)
        empty = self.ms._lexical_confidence("任意", [], con)
        con.close()
        self.assertGreater(hi, 0.9)
        self.assertLess(lo, 0.2)
        self.assertEqual(empty, 0.0)

    def test_lex_trust_segmentation(self):
        lt = self.ms._lex_trust
        self.assertEqual(lt(0.0), 0.0)
        self.assertEqual(lt(0.58), 0.0)
        self.assertEqual(lt(0.6), 0.0)
        self.assertEqual(lt(1.0), 1.0)
        self.assertAlmostEqual(lt(0.8), 0.5)

    def test_touch_reinforcement(self):
        self.ms.add_notes("P", "t", [{"kind": "claim", "content": "QoB 可替代 LET 作为束流品质"}])
        self.ms.search_notes("QoB LET 束流品质", limit=5, touch=True)
        con = self.ms._conn()
        cnt = con.execute("SELECT access_count, last_access FROM notes").fetchone()
        con.close()
        self.assertGreaterEqual(cnt[0], 1)
        self.assertTrue(cnt[1])

    def test_ask_touches_picked(self):
        for i in range(6):
            self.ms.add_notes("P", "t", [{"kind": "claim",
                                          "content": f"结论 {i}: DBSCAN 与 DNA 损伤 LET 的关系"}])
        self.ms.ask_memory("DBSCAN LET", k=3, compose=False, expand=False, rerank=False)
        con = self.ms._conn()
        touched = con.execute("SELECT COUNT(*) FROM notes WHERE COALESCE(access_count,0)>0").fetchone()[0]
        con.close()
        self.assertGreater(touched, 0)


class TestReflection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"
        from paperbrain import memory_store as ms
        self.ms = ms
        for i in range(6):
            ms.add_notes("P1", "t", [{"kind": "核心洞见" if i % 2 else "claim",
                                      "content": f"记忆条目 {i}: DBSCAN 的 ε 与 DNA 损伤阈值"}])

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reflect_insufficient(self):
        ms2_tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = ms2_tmp + "/m.sqlite"
        from paperbrain import memory_store as msx
        self.ms.add_notes("P2", "t", [{"kind": "claim", "content": "only one"}])
        r = msx.reflect("P2")
        self.assertFalse(r["ok"])
        shutil.rmtree(ms2_tmp, ignore_errors=True)
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"

    def test_reflect_with_mock_model(self):
        orig, origv = llm.reflect_notes, llm.verify_claims
        try:
            llm.reflect_notes = lambda ctx, n=3: (
                '[{"insight":"ε 的选取与损伤阈值耦合, 需联合标定","evidence":[1,2]},'
                '{"insight":"不同论文的阈值假设不一致, 结果不可直接比较","evidence":[3,4]}]')
            llm.verify_claims = lambda texts, ctx: json.dumps(
                [{"i": i, "ok": True} for i in range(len(texts))])
            r = self.ms.reflect("P1")
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["insights"], 2)
            self.assertEqual(r["verified"], 2)
            hits = self.ms.search_notes("联合标定 阈值假设", limit=5)
            self.assertTrue(any(h["kind"] == "反思" for h in hits))
            con = self.ms._conn()
            nlinks = con.execute("SELECT COUNT(*) FROM note_links WHERE rel='supports'").fetchone()[0]
            con.close()
            self.assertGreaterEqual(nlinks, 1)  # 证据链
        finally:
            llm.reflect_notes, llm.verify_claims = orig, origv

    def test_reflect_unverified_downgraded(self):
        orig, origv = llm.reflect_notes, llm.verify_claims
        try:
            llm.reflect_notes = lambda ctx, n=3: ('[{"insight":"无据猜测的结论","evidence":[1,2]}]')
            llm.verify_claims = lambda texts, ctx: json.dumps(
                [{"i": i, "ok": False} for i in range(len(texts))])
            r = self.ms.reflect("P1")
            self.assertTrue(r["ok"])
            self.assertEqual(r["verified"], 0)
            con = self.ms._conn()
            st = con.execute("SELECT status FROM notes WHERE kind='反思' ORDER BY id DESC LIMIT 1").fetchone()[0]
            con.close()
            self.assertEqual(st, "candidate")  # 证据不足 → 不冒充已验证
        finally:
            llm.reflect_notes, llm.verify_claims = orig, origv

    def test_reflect_global_needs_two_papers(self):
        orig, origv = llm.reflect_notes, llm.verify_claims
        try:
            llm.reflect_notes = lambda ctx, n=5: ('[{"insight":"两篇论文间存在共同的阈值敏感性","evidence":[1,3]}]')
            llm.verify_claims = lambda texts, ctx: json.dumps(
                [{"i": i, "ok": True} for i in range(len(texts))])
            r0 = self.ms.reflect_global()
            self.assertFalse(r0["ok"], "单篇库不应触发领域反思")
            for i in range(5):
                self.ms.add_notes("P2", "t", [{"kind": "claim", "content": f"另一篇结论 {i}: 阈值与效率"}])
            r = self.ms.reflect_global()
            self.assertTrue(r["ok"], r)
            self.assertEqual(r["scope"], "field")
            con = self.ms._conn()
            pid = con.execute("SELECT paper_id FROM notes WHERE kind='领域反思' ORDER BY id DESC LIMIT 1").fetchone()[0]
            con.close()
            self.assertEqual(pid, "__field__")
        finally:
            llm.reflect_notes, llm.verify_claims = orig, origv


class TestGraphAndTime(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"
        from paperbrain import memory_store as ms
        self.ms = ms

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ppr_multihop(self):
        # A—B—C 链 (内容互不相似, 避免 add_notes 自动建链): 查询只命中 A, PPR 两步应把 C 扩散出来
        ra = self.ms.add_notes("P", "t", [{"kind": "claim", "content": "ALPHAONLY 苹果园 土壤 酸碱度"}])
        a = ra["ids"][0]
        rb = self.ms.add_notes("P", "t", [{"kind": "claim", "content": "BETAONLY 香蕉 冷链 运输"}])
        b = rb["ids"][0]
        rc = self.ms.add_notes("P", "t", [{"kind": "claim", "content": "GAMMAONLY 樱桃 采摘 机械"}])
        c = rc["ids"][0]
        self.assertEqual(self.ms._conn().execute("SELECT links FROM notes WHERE id=?", (a,)).fetchone()[0],
                         "[]", "测试前提: 不允许自动建链")
        self.ms.link_notes(a, b, "extends")
        self.ms.link_notes(b, c, "extends")
        p1 = self.ms.graph_expand({a: 1.0}, steps=1)
        self.assertLess(p1.get(c, 0.0), 0.02, "一跳不应到达 C")
        ppr = self.ms.graph_expand({a: 1.0}, steps=2)
        self.assertGreater(ppr.get(c, 0.0), 0.02, f"两跳未扩散到 C: {ppr}")
        self.assertGreater(ppr.get(b, 0.0), ppr.get(c, 0.0))  # 近跳权重高于远跳

    def test_invalidate_hides_note(self):
        r = self.ms.add_notes("P", "t", [{"kind": "claim", "content": "INVALIDMARK 应被撤回的主张"}])
        nid = r["ids"][0]
        self.assertTrue(self.ms.search_notes("INVALIDMARK", limit=5))
        self.ms.invalidate_note(nid)
        self.assertFalse(self.ms.search_notes("INVALIDMARK", limit=5))  # 留档但不参与检索
        con = self.ms._conn()
        still = con.execute("SELECT COUNT(*) FROM notes WHERE id=?", (nid,)).fetchone()[0]
        con.close()
        self.assertEqual(still, 1)  # 历史不删除


class TestBatchSuppression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"
        from paperbrain import memory_store as ms
        self.ms = ms
        for pid, nn in (("P1", 6), ("P2", 3), ("P3", 3)):
            for i in range(nn):
                ms.add_notes(pid, "t", [{"kind": "claim",
                                         "content": f"{pid} 结论{i}: 阈值敏感性与剂量学量"}])

    def tearDown(self):
        self.ms.suppress_field_reflection(False)
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_batch_suppresses_field_reflection(self):
        orig, origv = llm.reflect_notes, llm.verify_claims
        outdir = self.tmp + "/out"
        os.makedirs(outdir, exist_ok=True)
        try:
            llm.reflect_notes = lambda ctx, n=5: ('[{"insight":"跨论文综合洞见Z","evidence":[1,2]}]')
            llm.verify_claims = lambda texts, ctx: json.dumps(
                [{"i": i, "ok": True} for i in range(len(texts))])
            self.ms.suppress_field_reflection(True)
            out = self.ms.finalize_memory(outdir, "P1", "full_read")
            self.assertNotIn("field_reflection", out)
            self.ms.suppress_field_reflection(False)
            out2 = self.ms.finalize_memory(outdir, "P1", "full_read")
            self.assertIn("field_reflection", out2)
            self.assertTrue(out2["field_reflection"]["ok"], out2)
        finally:
            llm.reflect_notes, llm.verify_claims = orig, origv

    def test_run_batch_resets_suppression(self):
        from paperbrain.batch import run_batch
        r = run_batch([], self.tmp + "/batchout", use_llm=False)
        self.assertFalse(self.ms._FIELD_REFLECT_SUPPRESS, "run_batch 结束必须复位抑制标志")
        self.assertEqual(r["done"], 0)
        self.assertIn("field_reflection", r)


if __name__ == "__main__":
    unittest.main()
