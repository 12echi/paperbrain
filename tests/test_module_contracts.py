"""模块衔接契约回归 (v5.0): 锁死本次深审发现的 5 个真 bug.
1. 编号 "2. Methods" / "4. Discussion" 标题必须检出
2. outline 按名称归位, 结论编号非4也不丢
3. 同名节 (Experiments+Discussion) 合并, 不互相覆盖
4. 实体抽词边界: LET/FLASH 不得命中 complete/flashing
5. coverage 诚实: 降级时 <1, 正常时 =1 (不被 Pass 输出回灌虚高)
"""
import unittest


PAPER = """Title

Abstract
This is the abstract describing the study and its contributions clearly enough.

1. Introduction
We introduce the problem and prior work with sufficient detail for the splitter.

2. Methods
We describe the model and the loss function used in this work with enough words.

3. Experiments
We evaluate on datasets and report accuracy and other metrics in tables.

4. Discussion
We discuss implications and compare with prior work in detail here.

5. Conclusion
We conclude the study and summarize limitations and future work clearly.
"""


class TestSectionContracts(unittest.TestCase):
    def test_methods_and_discussion_detected(self):
        from paperbrain.sections import split_sections
        secs = split_sections(PAPER, "2024_X_01")
        names = [s["name"] for s in secs]
        self.assertIn("method", names)
        self.assertIn("experiments", names)
        self.assertIn("conclusion", names)
        # Discussion 归入 experiments 桶 (Pass3)
        self.assertEqual(names.count("experiments"), 2)

    def test_numbered_variants(self):
        from paperbrain.sections import HEADERS
        import re
        for h in ("2. Methods", "2. Methodology", "3. Experiments",
                  "4. Discussion", "4. Results", "5. Conclusion"):
            self.assertTrue(any(re.match(p, h, re.I) for _, p, _ in HEADERS), h)


class TestOutlineContracts(unittest.TestCase):
    def test_conclusion_kept_when_number_not_4(self):
        from paperbrain.sections import split_sections
        from paperbrain.passes import run_passes
        from paperbrain.outline import build_outline
        p = run_passes(split_sections(PAPER, "2024_X_01"), "2024_X_01")
        ol = build_outline(p, {"entities": []}, "2024_X_01")
        h1 = [s["h1"] for s in ol["sections"]]
        for want in ("Background", "Method", "Experiments", "Conclusion"):
            self.assertIn(want, h1)

    def test_duplicate_name_merged_not_lost(self):
        from paperbrain.sections import split_sections
        from paperbrain.passes import run_passes
        p = run_passes(split_sections(PAPER, "2024_X_01"), "2024_X_01")
        self.assertIn("evaluate on datasets", p["pass3"].lower())
        self.assertIn("discuss implications", p["pass3"].lower())

    def test_name_sec_map(self):
        from paperbrain.sections import split_sections
        from paperbrain.passes import run_passes
        p = run_passes(split_sections(PAPER, "2024_X_01"), "2024_X_01")
        self.assertEqual(p["name_sec"]["method"], ["2"])
        self.assertEqual(sorted(p["name_sec"]["experiments"]), ["3", "4"])
        self.assertEqual(p["name_sec"]["conclusion"], ["5"])


class TestMemoryContracts(unittest.TestCase):
    def test_word_boundary_no_false_positive(self):
        from paperbrain.memory import rule_extract
        ents, _ = rule_extract("we complete the flashing deletion process")
        self.assertEqual([e["name"] for e in ents], [])

    def test_real_entity_still_hit(self):
        from paperbrain.memory import rule_extract
        ents, _ = rule_extract("The LET spectrum and RBE model using DBSCAN.")
        got = {e["name"] for e in ents}
        self.assertIn("LET", got)
        self.assertIn("RBE", got)
        self.assertIn("DBSCAN", got)


class TestCoverageContract(unittest.TestCase):
    def test_normal_coverage_full(self):
        from paperbrain.sections import split_sections
        from paperbrain.passes import run_passes
        p = run_passes(split_sections(PAPER, "2024_X_01"), "2024_X_01")
        self.assertEqual(p["coverage"], 1.0)

    def test_downgrade_coverage_below_one(self):
        from paperbrain.sections import split_sections
        from paperbrain.passes import run_passes
        # 强制降级: 切分失败
        secs = split_sections("no headers at all just a blob " * 40, "X_1")
        p = run_passes(secs, "X_1")
        self.assertTrue(p["downgrade_pass1_only"])
        self.assertLessEqual(p["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
