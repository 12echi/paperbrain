"""逐行逻辑审查修复的回归测试 (B1 索引错位 / B2 画像词边界 / B3 中文实体相似度 / 注解与缓存)."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import paperbrain.llm as llm


class TestLogicFixes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_MEMORY_DB"] = self.tmp + "/mem.sqlite"
        from paperbrain import memory_store as ms
        self.ms = ms

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_MEMORY_DB", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    # B1: 模型返回 1-based 序号时必须正确对齐, 且全 False 也要采信
    def test_verify_claims_one_based(self):
        orig = llm.verify_claims
        try:
            llm.verify_claims = lambda texts, ctx: '[{"i":1,"ok":true},{"i":2,"ok":false},{"i":3,"ok":true}]'
            r = self.ms._verify_claims(["a", "b", "c"], "ctx")
            self.assertEqual(r, [True, False, True])
        finally:
            llm.verify_claims = orig

    def test_verify_claims_all_false_trusted(self):
        orig = llm.verify_claims
        try:
            llm.verify_claims = lambda texts, ctx: '[{"i":0,"ok":false},{"i":1,"ok":false}]'
            # 不能因为"全 False"就回退词法 (词法可能误判为支持)
            r = self.ms._verify_claims(["DBSCAN 聚类用于损伤", "DNA 修复"], "DBSCAN 聚类用于损伤")
            self.assertEqual(r, [False, False])
        finally:
            llm.verify_claims = orig

    def test_verify_claims_zero_based_ok(self):
        orig = llm.verify_claims
        try:
            llm.verify_claims = lambda texts, ctx: '[{"i":0,"ok":true},{"i":1,"ok":true}]'
            self.assertEqual(self.ms._verify_claims(["x", "y"], "x y"), [True, True])
        finally:
            llm.verify_claims = orig

    # B2: 画像词边界 (let 不得命中 delete/complete)
    def test_bump_preferences_word_boundary(self):
        self.ms.add_alias("let", "LET_metric")
        hit = self.ms._bump_query_preferences("we delete and complete this")
        self.assertNotIn("let", [h for h in hit])
        hit2 = self.ms._bump_query_preferences("what does LET mean")
        self.assertIn("let", hit2)

    # B3: 中文实体相似度 (以前 [a-z0-9]+ 丢弃中文 -> 永不合并)
    def test_lexical_sim_cjk(self):
        from paperbrain.graph import lexical_sim
        s = lexical_sim("闪速放疗", "闪速放疗技术")
        self.assertGreater(s, 0.4)
        # 完全无关的中文应当低
        self.assertLess(lexical_sim("闪速放疗", "质子蒙特卡洛"), 0.2)
        # ASCII 行为保持
        self.assertGreaterEqual(lexical_sim("ImageNet-1k", "ImageNet 1k"), 0.85)


class TestStaticInvariants(unittest.TestCase):
    def test_safe_pid_and_under(self):
        from pathlib import Path
        from paperbrain.server import _safe_pid, _under
        self.assertEqual(_safe_pid("../.."), "paper")
        self.assertEqual(_safe_pid("/etc/passwd"), "etc_passwd")
        self.assertEqual(_safe_pid("2024_NeurIPS_01"), "2024_NeurIPS_01")
        self.assertEqual(_safe_pid(""), "paper")
        self.assertFalse(_under(Path("/tmp/../../etc/passwd"), Path("/tmp")))
        self.assertTrue(_under(Path("/tmp/a/b.txt"), Path("/tmp")))

    def test_figures_no_fake_iou(self):
        from paperbrain.figures import split_figure
        import tempfile, os
        try:
            import cv2, numpy as np
        except Exception:
            self.skipTest("缺cv2")
        img = 255 * np.ones((200, 400), dtype="uint8")
        img[20:90, 20:190] = 0
        img[20:90, 210:380] = 0
        fd, p = tempfile.mkstemp(suffix=".png"); os.close(fd)
        cv2.imwrite(p, img)
        try:
            r = split_figure(p, "Fig. (a) left. (b) right.")
            self.assertTrue(r["split"])
            self.assertGreater(r["iou"], 0.5)  # 必须是真实计算且过硬阈值
        finally:
            os.unlink(p)

    def test_extract_claim_annotation(self):
        import inspect
        from paperbrain.verifier import extract_claim
        ann = inspect.signature(extract_claim).return_annotation
        self.assertIn("Dict", str(ann))

    def test_embedding_cache_bounded(self):
        src = Path("paperbrain/embeddings.py").read_text(encoding="utf-8")
        self.assertIn("_Q_CACHE.clear()", src)

    def test_js_keyword_filter_allows_two_char(self):
        import paperbrain.server as srv
        self.assertIn("k.length>1", srv.PAGE)
        self.assertNotIn("k.length>2", srv.PAGE)


if __name__ == "__main__":
    unittest.main()
