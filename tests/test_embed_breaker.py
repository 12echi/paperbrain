"""Embedding 熔断保护测试: 端点不可达时必须快速失败, 不许挂住检索/评测。"""
import os
import time
import unittest


class TestEmbedBreaker(unittest.TestCase):
    def setUp(self):
        os.environ["PAPERBRAIN_EMBED_BASE_URL"] = "http://127.0.0.1:1/v1"  # 必然拒绝连接
        os.environ["PAPERBRAIN_EMBED_API_KEY"] = "sk-test"
        from paperbrain import embeddings
        self.emb = embeddings
        embeddings._reset_breaker()

    def tearDown(self):
        self.emb._reset_breaker()
        os.environ.pop("PAPERBRAIN_EMBED_BASE_URL", None)
        os.environ.pop("PAPERBRAIN_EMBED_API_KEY", None)

    def test_fast_fail_and_circuit_break(self):
        from paperbrain import config
        self.assertTrue(config.embed_enabled())
        self.assertFalse(self.emb.breaker_open(), "初始不应处于熔断状态")
        t0 = time.time()
        out1 = self.emb.embed_texts(["探针"])
        first = time.time() - t0
        self.assertTrue(all(v is None for v in out1))
        self.assertLess(first, 10, "连接拒绝应快速失败, 不得等长超时")
        self.assertTrue(self.emb.breaker_open(), "失败后必须熔断")
        t0 = time.time()
        out2 = self.emb.embed_texts(["第二次"])
        self.assertLess(time.time() - t0, 0.2, "熔断期内必须立即返回")
        self.assertTrue(all(v is None for v in out2))
        self.emb._reset_breaker()
        self.assertFalse(self.emb.breaker_open())


if __name__ == "__main__":
    unittest.main()
