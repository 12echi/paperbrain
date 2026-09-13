"""审计修复回归: 质检门可检索性 / 画像词边界 / 去重无死链 / 矛盾定位 / 服务器硬化."""
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path


class TestAuditFixes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"
        from paperbrain import memory_store as ms
        self.ms = ms
        ms.add_notes("P1", "t", [
            {"kind": "claim", "content": "活跃主张 alpha 关于损伤", "status": "active"},
            {"kind": "claim", "content": "未验证主张 beta 特殊词", "status": "candidate"},
            {"kind": "claim", "content": "被驳回主张 gamma", "status": "rejected"},
            {"kind": "claim", "content": "有争议主张 delta", "status": "contested"},
        ])

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    # D1/D2: 质检门
    def test_candidate_and_rejected_not_retrievable(self):
        for q in ("beta 特殊词", "gamma"):
            hits = self.ms.search_notes(q, limit=10)
            self.assertFalse(any(q.split()[0] in h["content"] for h in hits), q)
        allhits = self.ms.search_notes("", limit=100)
        joined = " ".join(h["content"] for h in allhits)
        self.assertNotIn("beta 特殊词", joined)
        self.assertNotIn("gamma", joined)
        self.assertIn("alpha", joined)  # active 正常
        self.assertIn("delta", joined)  # contested 可见

    def test_contested_marked_in_results(self):
        hits = self.ms.search_notes("delta", limit=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["status"], "contested")

    def test_include_pending_override(self):
        hits = self.ms.search_notes("", limit=100, include_pending=True)
        self.assertTrue(any("beta" in h["content"] for h in hits))

    def test_recall_neighbors_respect_gate(self):
        # 邻居一跳扩展也必须过质检门 (修复: 原 recall 按 id 盲取, candidate 泄漏)
        self.ms.add_notes("P9", "t", [
            {"kind": "claim", "content": "邻居候选 A 关于 DBSCAN 聚类", "status": "active"},
            {"kind": "claim", "content": "邻居候选 B 关于 DBSCAN 聚类的未验证说法", "status": "candidate"},
        ])
        con = self.ms._conn()
        active = con.execute("SELECT id FROM notes WHERE content LIKE '邻居候选 A%'").fetchone()[0]
        cand = con.execute("SELECT id FROM notes WHERE content LIKE '邻居候选 B%'").fetchone()[0]
        # 手工把两条互链
        con.execute("UPDATE notes SET links=? WHERE id=?", (json.dumps([cand]), active))
        con.execute("UPDATE notes SET links=? WHERE id=?", (json.dumps([active]), cand))
        con.commit()
        con.close()
        nb = self.ms.neighbors_of([cand])  # 直接请求 candidate 作邻居
        self.assertEqual(nb, [])
        # recall 路径同样过滤
        rec = self.ms.recall("DBSCAN 聚类", k=10)
        self.assertTrue(all(n["status"] in ("active", "contested") for n in rec["neighbors"]))

    # D3: 画像词边界
    def test_profile_word_boundary(self):
        self.ms.bump_preference("let", 1.0)
        self.ms.add_notes("P2", "t", [
            {"kind": "claim", "content": "we delete and complete the work", "status": "active"}])
        con = self.ms._conn()
        rid = con.execute("SELECT id FROM notes WHERE content LIKE '%delete%'").fetchone()[0]
        self.assertEqual(self.ms._profile_boost(con, [rid]), {})  # 'let' 不应命中 delete
        con.close()

    # D4: 去重无死链
    def test_consolidate_no_dead_links(self):
        self.ms.add_notes("P3", "dup", [
            {"kind": "claim", "content": "重复内容去重测试 ABC"},
            {"kind": "claim", "content": "重复内容去重测试 ABC"},
            {"kind": "claim", "content": "重复内容去重测试 ABC"},
        ])
        r = self.ms.consolidate("P3")
        self.assertGreaterEqual(r["removed"], 2)
        con = self.ms._conn()
        ids = {row[0] for row in con.execute("SELECT id FROM notes")}
        orphan_links = [x for x in con.execute("SELECT src,dst FROM note_links")
                        if x[0] not in ids or x[1] not in ids]
        dead_json = 0
        for (lj,) in con.execute("SELECT links FROM notes"):
            try:
                arr = json.loads(lj or "[]")
            except Exception:
                arr = []
            dead_json += sum(1 for i in arr if i not in ids)
        orphan_vec = [x for x in con.execute("SELECT owner_id FROM vectors")
                      if x[0] not in ids]
        con.close()
        self.assertEqual(orphan_links, [])
        self.assertEqual(dead_json, 0)
        self.assertEqual(orphan_vec, [])

    # D5: 矛盾定位排除自身
    def test_find_conflict_excludes_self(self):
        claim = "既有观点认为 LET 足以表征生物效应"
        self.ms.add_notes("P4", "card", [{"kind": "claim", "content": claim, "status": "contested"}])
        other = self.ms._find_conflict(claim)
        self.assertTrue(other is None or other["content"].strip() != claim.strip())


class TestServerHardening(unittest.TestCase):
    def test_path_whitelist(self):
        import paperbrain.server as srv
        self.assertFalse(srv._path_allowed(Path("/etc/passwd")))
        self.assertFalse(srv._path_allowed(Path("/usr/bin/env")))
        self.assertTrue(srv._path_allowed(Path.home() / "Downloads" / "a.pdf"))
        self.assertTrue(srv._path_allowed(Path("/tmp/x.pdf")))

    def test_jobs_cleanup(self):
        import paperbrain.server as srv
        with srv.JOBS_LOCK:
            srv.JOBS.clear()
            srv.JOBS["old"] = {"done": True, "t0": time.time() - 7200}
            srv.JOBS["new"] = {"done": False, "t0": time.time()}
        dropped = srv._jobs_cleanup(ttl_done=3600)
        self.assertGreaterEqual(dropped, 1)
        with srv.JOBS_LOCK:
            self.assertNotIn("old", srv.JOBS)
            self.assertIn("new", srv.JOBS)
            srv.JOBS.clear()


if __name__ == "__main__":
    unittest.main()
