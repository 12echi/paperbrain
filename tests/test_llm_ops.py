"""全模型语义算子测试 (mock llm.chat, 不烧额度)."""
import unittest

import paperbrain.llm as llm
import paperbrain.llm_ops as ops


class TestLLMOps(unittest.TestCase):
    def setUp(self):
        ops.reset_cache()
        self._orig = llm.chat
        self._orig_ops_chat = ops._chat
        self.calls = 0

        def fake(messages, **kw):
            self.calls += 1
            c = messages[0]["content"]
            if "结构解析器" in c:
                return ('```json\n[{"name":"abstract","sec":"0","title":"Abstract"},'
                        '{"name":"method","sec":"2","title":"2. Methods"},'
                        '{"name":"conclusion","sec":"5","title":"5. Conclusion"}]\n```')
            if "知识图谱" in c:
                return ('{"entities":[{"name":"DBSCAN","type":"Algorithm/Model"},'
                        '{"name":"model","type":"Algorithm/Model"}],'
                        '"relations":[{"from":"DBSCAN","rel":"Evaluated_On","to":"DNA damage"},'
                        '{"from":"DBSCAN","rel":"BAD_REL","to":"X"}]}')
            if "0-100" in c:
                return "87"
            return ""
        llm.chat = fake
        ops._chat = lambda p, max_tokens=1500: fake([{"role": "user", "content": p}])

    def tearDown(self):
        llm.chat = self._orig
        ops._chat = self._orig_ops_chat

    def test_segment_locates_titles(self):
        text = "Abstract\nabs.\n2. Methods\nuse DBSCAN.\n5. Conclusion\ndone."
        seg = ops.segment(text, "P_1")
        self.assertIsNotNone(seg)
        names = [s["name"] for s in seg]
        self.assertEqual(names[:2], ["abstract", "method"])
        self.assertIn("conclusion", names)
        # 每个 chunk_id 前缀正确
        self.assertTrue(seg[0]["chunks"][0]["chunk_id"].startswith("P_1_Sec0_"))

    def test_segment_fallback_on_bad_json(self):
        ops._chat = lambda p, max_tokens=1500: "抱歉我做不到"
        self.assertIsNone(ops.segment("Abstract\nx", "P_1"))

    def test_extract_graph_rejects_invalid_batch_then_retries_atomically(self):
        calls = []
        bad = ('{"entities":[{"name":"DBSCAN","type":"Algorithm/Model"}],'
               '"relations":[{"from":"DBSCAN","rel":"BAD_REL","to":"X"}]}')
        good = ('{"entities":[{"name":"DBSCAN","type":"Algorithm/Model"},'
                '{"name":"DNA damage","type":"Problem/Task"}],'
                '"relations":[{"from":"DBSCAN","rel":"Requires","to":"DNA damage"}]}')
        def retrying(prompt, max_tokens=1500):
            calls.append(prompt)
            return bad if len(calls) == 1 else good
        ops._chat = retrying
        ents, rels = ops.extract_graph("whatever")
        self.assertEqual([e["name"] for e in ents], ["DBSCAN", "DNA damage"])
        self.assertEqual(rels, [{"from": "DBSCAN", "rel": "Requires",
                                 "to": "DNA damage"}])
        self.assertEqual(ops.last_graph_status()["status"], "RETRY_ACCEPTED")
        self.assertEqual(len(calls), 2)

    def test_extract_graph_two_invalid_batches_enter_manual_review(self):
        ops._chat = lambda *a, **k: '{"entities":[],"relations":[{"rel":"BAD"}]}'
        self.assertEqual(ops.extract_graph("whatever"), ([], []))
        status = ops.last_graph_status()
        self.assertEqual(status["status"], "MANUAL_REVIEW")
        self.assertEqual(status["attempts"], 2)

    def test_extract_graph_empty_payload_is_not_accepted(self):
        ops._chat = lambda *a, **k: '{"entities":[],"relations":[]}'
        self.assertEqual(ops.extract_graph("whatever"), ([], []))
        self.assertEqual(ops.last_graph_status()["status"], "MANUAL_REVIEW")

    def test_nli_cache_and_parse(self):
        s1 = ops.nli_score("DBSCAN models DNA damage", "DBSCAN used to model DNA damage")
        s2 = ops.nli_score("DBSCAN models DNA damage", "DBSCAN used to model DNA damage")
        self.assertAlmostEqual(s1, 0.87)
        self.assertEqual(s1, s2)
        self.assertEqual(self.calls, 1)  # 首次调模型, 第二次命中缓存

    def test_enabled_flag(self):
        import os
        os.environ.pop("PAPERBRAIN_ALL_MODEL", None)
        self.assertFalse(ops.enabled())
        os.environ["PAPERBRAIN_ALL_MODEL"] = "1"
        try:
            self.assertTrue(ops.enabled())
        finally:
            os.environ.pop("PAPERBRAIN_ALL_MODEL", None)


if __name__ == "__main__":
    unittest.main()
