"""检索指标测试 (tools/eval_retrieval.py 的核心, 手算校验)。"""
import math
import unittest

from paperbrain.retrieval import recall_at_k, mrr, ndcg_at_k


class TestMetrics(unittest.TestCase):
    def test_recall(self):
        self.assertAlmostEqual(recall_at_k(["a", "b", "c"], {"b", "c"}, k=2), 0.5)
        self.assertAlmostEqual(recall_at_k(["b", "c"], {"b", "c"}, k=2), 1.0)
        self.assertAlmostEqual(recall_at_k(["a", "b"], {"b", "c"}, k=5), 0.5)
        self.assertAlmostEqual(recall_at_k(["a"], set(), k=3), 1.0)  # 无正例不罚

    def test_mrr(self):
        self.assertAlmostEqual(mrr(["a", "b", "c"], {"b"}), 0.5)
        self.assertAlmostEqual(mrr(["a", "b", "c"], {"a"}), 1.0)
        self.assertAlmostEqual(mrr(["a", "b"], {"z"}), 0.0)

    def test_ndcg(self):
        got = ndcg_at_k(["a", "b", "c", "d"], {"b", "d"}, k=4)
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(got, dcg / idcg, places=6)
        self.assertAlmostEqual(ndcg_at_k(["b", "d", "a"], {"b", "d"}, k=3), 1.0)
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "c", "d"], {"b", "d"}, k=1), 0.0)
        self.assertAlmostEqual(ndcg_at_k(["a"], set(), k=3), 1.0)


if __name__ == "__main__":
    unittest.main()
