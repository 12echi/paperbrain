"""
PaperBrain Copilot - End-to-End Acceptance Test Suite (AC1 - AC6)
Derived strictly from ORIGINAL_REQUEST.md and PROJECT.md specifications.

Acceptance Criteria:
- AC1: Full regression runner verifying every test module except this orchestrator itself.
- AC2: Test vector scientific figure extraction: creates synthetic pure vector PDF
       (PyMuPDF drawing paths, e.g. line/bar chart without bitmap XObjects),
       invokes vision.extract_images, and asserts len(figs) > 0 and files exist.
- AC3: Test Span coordinate [start, end] alignment: provides academic text with
       equations ($E=mc^2$), decimals (Sec 3.2, 98.5%), and abbreviations (e.g., Fig. 1),
       splits spans, and asserts text[start:end].strip() == sent with 0 drift.
- AC4: Test concurrency and session isolation: launches 2 independent client threads
       calling server.py HTTP endpoints concurrently with distinct payloads,
       asserting isolated session directories, distinct outputs, no cross-talk, and stable 200 OK responses.
- AC5: Test NLI model invocation convergence: configures CitationVerifierV5 with
       a Spy/Mock semantic scorer, verifies a draft containing citations across multiple chunks,
       and asserts scorer.call_count <= 2 per citation.
- AC6: Test verification caching: runs CitationVerifierV5 twice on identical citation
       and context, asserts 100% cache hit rate and call_count delta == 0 on the second call.
"""

import io
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try PyMuPDF
try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


# ==============================================================================
# AC1: Regression Runner for Existing Suite
# ==============================================================================

EXISTING_TEST_MODULES = sorted(
    path.stem for path in Path(__file__).parent.glob("test_*.py")
    if path.stem != Path(__file__).stem
)


class TestAC1RegressionSuite(unittest.TestCase):
    """AC1: 全套回归测试 - 验证 verifier 门禁及当前全部其他测试模块."""

    def test_ac1_verifier_v5_suite(self):
        """AC1.1: 独立验证 tests/test_verifier_v5.py 全部门禁用例 100% 通过."""
        suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_verifier_v5")
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=0)
        result = runner.run(suite)
        output = stream.getvalue()

        self.assertEqual(
            len(result.failures), 0,
            f"test_verifier_v5 has failures:\n{output}"
        )
        self.assertEqual(
            len(result.errors), 0,
            f"test_verifier_v5 has errors:\n{output}"
        )
        self.assertGreaterEqual(
            result.testsRun, 15,
            f"Expected at least 15 tests in test_verifier_v5, ran {result.testsRun}"
        )

    def test_ac1_full_suite_modules_regression(self):
        """AC1.2: 动态遍历全部其他测试模块，断言 0 失败、0 错误、无回归."""
        suite = unittest.TestSuite()
        loader = unittest.defaultTestLoader

        loaded_modules = 0
        for mod_name in EXISTING_TEST_MODULES:
            try:
                mod_suite = loader.loadTestsFromName(f"tests.{mod_name}")
                suite.addTest(mod_suite)
                loaded_modules += 1
            except Exception as e:
                self.fail(f"Failed to load test module tests.{mod_name}: {e}")

        self.assertEqual(loaded_modules, len(EXISTING_TEST_MODULES),
                         "Must load every current test module except this orchestrator")

        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=0)
        result = runner.run(suite)
        output = stream.getvalue()

        failure_details = []
        if result.failures:
            for test_case, tb in result.failures:
                failure_details.append(f"FAILURE in {test_case}: {tb}")
        if result.errors:
            for test_case, tb in result.errors:
                failure_details.append(f"ERROR in {test_case}: {tb}")

        self.assertEqual(
            len(result.failures), 0,
            f"Regression failures detected in existing suite ({len(result.failures)}):\n" +
            "\n".join(failure_details[:5])
        )
        self.assertEqual(
            len(result.errors), 0,
            f"Regression errors detected in existing suite ({len(result.errors)}):\n" +
            "\n".join(failure_details[:5])
        )
        self.assertGreater(
            result.testsRun, 50,
            f"Total executed tests across all modules should be > 50, got {result.testsRun}"
        )


# ==============================================================================
# AC2: Vector Scientific Figure Extraction (Pure Vector PDF)
# ==============================================================================

class TestAC2VectorFigureExtraction(unittest.TestCase):
    """AC2: 矢量科研图表提取 - 针对纯矢量 PDF (无位图 XObject), 提取图表并落盘有效图像."""

    def _create_pure_vector_pdf(self, pdf_path: Path):
        """使用 PyMuPDF 纯矢量算子 (lines, rects, curves) 绘制学术图表, 不嵌入任何位图."""
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # 插入图表标题与说明文本
        page.insert_text(
            (72, 80),
            "Figure 1: Performance Benchmark and Accuracy Comparison",
            fontsize=12
        )
        page.insert_text(
            (72, 98),
            "Comparison between Baseline and Proposed Model across multiple epochs.",
            fontsize=9
        )

        # 纯矢量路径绘制折线图与柱状图
        shape = page.new_shape()

        # 坐标轴
        shape.draw_line(fitz.Point(72, 320), fitz.Point(450, 320))
        shape.draw_line(fitz.Point(72, 320), fitz.Point(72, 120))

        # 柱状图条目 (Rectangles)
        bars = [
            fitz.Rect(100, 220, 135, 320),
            fitz.Rect(160, 180, 195, 320),
            fitz.Rect(220, 150, 255, 320),
            fitz.Rect(280, 130, 315, 320),
            fitz.Rect(340, 140, 375, 320),
        ]
        for b in bars:
            shape.draw_rect(b)

        # 折线图趋势线 (Polyline / Curves)
        line_pts = [
            fitz.Point(117, 210),
            fitz.Point(177, 170),
            fitz.Point(237, 140),
            fitz.Point(297, 125),
            fitz.Point(357, 135),
        ]
        for i in range(len(line_pts) - 1):
            shape.draw_line(line_pts[i], line_pts[i + 1])

        # 绘制数据散点点阵
        for pt in line_pts:
            shape.draw_circle(pt, 3)

        shape.finish(color=(0.1, 0.2, 0.6), fill=(0.3, 0.5, 0.8), stroke_opacity=1.0)
        shape.commit()

        # 底部正文文本
        page.insert_text(
            (72, 360),
            "As depicted in Figure 1, the proposed method demonstrates significant throughput gains.",
            fontsize=10
        )

        doc.save(str(pdf_path))
        doc.close()

    def test_ac2_pure_vector_figure_extracted_and_rasterized(self):
        """验证纯矢量学术 PDF (get_images 为空) 中图表被检测并局部光栅化输出有效 PNG."""
        if not HAS_FITZ:
            self.skipTest("PyMuPDF (fitz) is required for vector PDF generation and extraction")

        from paperbrain.vision import extract_images

        with tempfile.TemporaryDirectory() as td:
            pdf_path = Path(td) / "vector_chart_sample.pdf"
            self._create_pure_vector_pdf(pdf_path)

            # 严格前置断言: 该 PDF 绝无任何内嵌位图 XObject
            verify_doc = fitz.open(str(pdf_path))
            raw_images = verify_doc[0].get_images()
            verify_doc.close()
            self.assertEqual(
                len(raw_images), 0,
                "Precondition violated: Test PDF must contain 0 raster image XObjects"
            )

            # 执行图表提取
            figs = extract_images(str(pdf_path), td, max_images=5)

            # 核心断言: 纯矢量图表必须被提取为有效图像文件且数量 > 0
            self.assertGreater(
                len(figs), 0,
                "AC2 Failure: Vector figures were not extracted (len(figs) == 0). "
                "extract_images() must detect vector drawing layouts and rasterize them via page.get_pixmap()."
            )

            for fig in figs:
                img_path = Path(fig["image"])
                self.assertTrue(
                    img_path.exists(),
                    f"Extracted figure file does not exist on disk: {img_path}"
                )
                self.assertGreater(
                    img_path.stat().st_size, 0,
                    f"Extracted figure file is empty: {img_path}"
                )
                self.assertGreater(
                    fig.get("pixels", 0), 1000,
                    f"Extracted figure has insufficient pixel area: {fig}"
                )


# ==============================================================================
# AC3: Span Coordinate [start, end] Alignment (Zero Drift)
# ==============================================================================

class TestAC3SpanCoordinateAlignment(unittest.TestCase):
    """AC3: 引文 Span 坐标零偏移 - 数学公式、小数与缩写词切分后 text[start:end] 严格对齐."""

    def test_ac3_span_zero_drift_equations_and_decimals(self):
        """验证包含数学公式 ($E=mc^2$)、小数 (Sec 3.2, 98.5%) 及缩写 (Fig. 1) 的文本 Span 零偏移."""
        from paperbrain.text import split_spans

        test_text = (
            "In Sec. 3.2, we evaluate the baseline convergence. "
            "As shown in Fig. 1, our method achieves a 98.5% BLEU-4 gain (p < 0.01) on the test set. "
            "Furthermore, the relativistic energy relation $E=mc^2$ holds under standard assumptions. "
            "Baseline comparisons (e.g., Model A vs. Model B) demonstrate substantial stability gains. "
            "Finally, et al. verified these findings in Tab. 2.4."
        )

        spans = split_spans(test_text)
        self.assertGreater(len(spans), 0, "split_spans must return non-empty spans")

        total_drift = 0
        drift_errors = []

        for idx, (sent, start, end) in enumerate(spans):
            slice_raw = test_text[start:end]
            slice_stripped = slice_raw.strip()
            drift = abs(len(slice_stripped) - len(sent))
            total_drift += drift

            if slice_stripped != sent:
                drift_errors.append(
                    f"Span #{idx} drift detected:\n"
                    f"  Expected sent : '{sent}' (len={len(sent)})\n"
                    f"  Actual slice  : '{slice_stripped}' (len={len(slice_stripped)})\n"
                    f"  Raw slice [{start}:{end}]: '{slice_raw}'"
                )

        self.assertEqual(
            len(drift_errors), 0,
            f"AC3 Failure: Span coordinate drift detected across {len(drift_errors)} sentences:\n" +
            "\n".join(drift_errors)
        )
        self.assertEqual(
            total_drift, 0,
            f"AC3 Failure: Total coordinate drift must be exactly 0, got {total_drift}"
        )

    def test_ac3_protect_preserves_string_length_1_to_1(self):
        """验证 protect() 标点保护机制必须维持 1:1 字符长度，禁止字符串占位符膨胀."""
        from paperbrain.text import protect

        sample = "Section 3.2 shows 98.5% accuracy in Fig. 1 and Tab. 2 vs. baseline (e.g., p < 0.01)."
        protected = protect(sample)

        self.assertEqual(
            len(protected), len(sample),
            f"AC3 Contract Failure: protect(s) expanded length from {len(sample)} to {len(protected)}. "
            f"Multi-character tokens like <DOT> (+4) or <P> (+2) must be replaced with 1:1 Unicode sentinels."
        )


# ==============================================================================
# AC4: Concurrency and Session Isolation
# ==============================================================================

class TestAC4ConcurrencyAndSessionIsolation(unittest.TestCase):
    """AC4: 并发安全与会话隔离 - 2 个独立客户端并发请求 server.py，会话隔离完整，状态无污染."""

    def setUp(self):
        import paperbrain.server as _srvmod
        self.srv = HTTPServer(("127.0.0.1", 0), _srvmod.H)
        self.port = self.srv.server_address[1]
        self.srv_thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.srv_thread.start()
        self.test_dirs = []

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        for d in self.test_dirs:
            shutil.rmtree(d, ignore_errors=True)

    def _post_json(self, path: str, payload: dict) -> Tuple[int, dict]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def test_ac4_concurrent_two_client_sessions_isolation(self):
        """启动 2 个独立客户端并发请求 /api/outline，断言无跨请求状态污染且稳定 200 OK."""
        pid_a = "session_client_alpha_01"
        pid_b = "session_client_beta_02"
        self.test_dirs.extend([Path("out/web") / pid_a, Path("out/web") / pid_b])

        text_a = (
            "Abstract\n"
            "This paper explores quantum computing algorithms for linear systems. " * 8 +
            "\nMethods\n"
            "We employ the HHL algorithm with quantum phase estimation. " * 10
        )
        text_b = (
            "Abstract\n"
            "This study investigates neural network pruning for edge devices. " * 8 +
            "\nMethods\n"
            "We utilize magnitude-based iterative weight pruning. " * 10
        )

        results = {}
        errors = []
        barrier = threading.Barrier(2)

        def client_worker(client_name: str, pid: str, text: str):
            try:
                barrier.wait(timeout=10)
                status, resp = self._post_json("/api/outline", {
                    "paper_id": pid,
                    "text": text,
                    "session_id": f"sess_{client_name}"
                })
                results[client_name] = (status, resp)
            except Exception as e:
                errors.append(f"Client {client_name} error: {e}")

        t_a = threading.Thread(target=client_worker, args=("alpha", pid_a, text_a))
        t_b = threading.Thread(target=client_worker, args=("beta", pid_b, text_b))

        t_a.start()
        t_b.start()
        t_a.join(timeout=30)
        t_b.join(timeout=30)

        self.assertEqual(len(errors), 0, f"Concurrent clients encountered exceptions: {errors}")
        self.assertIn("alpha", results)
        self.assertIn("beta", results)

        status_a, resp_a = results["alpha"]
        status_b, resp_b = results["beta"]

        # 断言 1: 均稳定返回 200 OK
        self.assertEqual(status_a, 200, "Client Alpha must receive HTTP 200")
        self.assertEqual(status_b, 200, "Client Beta must receive HTTP 200")
        self.assertTrue(resp_a.get("ok"), f"Client Alpha failed: {resp_a}")
        self.assertTrue(resp_b.get("ok"), f"Client Beta failed: {resp_b}")

        # 断言 2: 会话与 paper_id 严格隔离
        self.assertEqual(resp_a.get("paper_id"), pid_a)
        self.assertEqual(resp_b.get("paper_id"), pid_b)

        # 断言 3: 输出内容互不串扰
        outline_a_str = json.dumps(resp_a.get("outline", ""))
        outline_b_str = json.dumps(resp_b.get("outline", ""))
        self.assertNotEqual(outline_a_str, outline_b_str, "Concurrent outputs must not be identical")

        # 断言 4: 磁盘隔离目录独立存在且数据无交叉污染
        dir_a = Path("out/web") / pid_a
        dir_b = Path("out/web") / pid_b
        self.assertTrue(dir_a.exists(), f"Client Alpha workspace missing: {dir_a}")
        self.assertTrue(dir_b.exists(), f"Client Beta workspace missing: {dir_b}")

        input_a = (dir_a / "input.txt").read_text(encoding="utf-8")
        input_b = (dir_b / "input.txt").read_text(encoding="utf-8")
        self.assertIn("quantum computing", input_a)
        self.assertNotIn("pruning", input_a)
        self.assertIn("neural network pruning", input_b)
        self.assertNotIn("quantum computing", input_b)


# ==============================================================================
# AC5: NLI Model Invocation Convergence (<= 2 Calls Per Citation)
# ==============================================================================

class SpyScorer:
    """Spy semantic scorer tracking invocation count and arguments."""

    def __init__(self, fixed_score: float = 0.90):
        self.call_count = 0
        self.calls: List[Tuple[str, str]] = []
        self.fixed_score = fixed_score

    def __call__(self, claim: str, src: str) -> float:
        self.call_count += 1
        self.calls.append((claim, src))
        return self.fixed_score


class TestAC5NLIInvocationConvergence(unittest.TestCase):
    """AC5: NLI 模型调用收敛 - 两阶段粗排过滤至 Top-2，单引文实际模型调用严格 <= 2 次."""

    def test_ac5_single_citation_nli_calls_capped_at_two(self):
        """验证单引文验证流程中，包含 8 个候选分块的长节内，NLI 模型调用严格 <= 2 次."""
        from paperbrain.verifier import CitationVerifierV5

        gt_chunks = {
            "2024_NeurIPS_01_3.2": "Tiling attention into on-chip SRAM eliminates off-chip memory access."
        }
        # 构造包含 8 个候选分块的真实长章节
        section_chunks = {
            "2024_NeurIPS_01_3.2": [
                "Chunk 0: Overview of GPU memory hierarchy and thread scheduling.",
                "Chunk 1: Tiling attention into on-chip SRAM eliminates off-chip memory access.",
                "Chunk 2: Mathematical derivation of online softmax rescaling factor.",
                "Chunk 3: Numerical stability proofs for half-precision floating point.",
                "Chunk 4: Backward pass recomputation versus intermediate caching.",
                "Chunk 5: Kernel fusion implementation details in CUDA and Triton.",
                "Chunk 6: Experimental latency benchmarks on A100 SXM4 architectures.",
                "Chunk 7: Conclusion and discussions on future architectural co-design."
            ]
        }
        fig_index = {"2024_NeurIPS_01_3.2": {"figs": [], "tabs": []}}

        spy = SpyScorer(fixed_score=0.92)
        verifier = CitationVerifierV5(
            gt_chunks,
            fig_index,
            semantic_scorer=spy,
            section_chunks=section_chunks
        )

        draft = "Our architecture tiles attention into on-chip SRAM [Ref: 2024_NeurIPS_01, Sec 3.2]."
        report = verifier.verify_draft(draft)

        self.assertEqual(report["status"], "CLEAN")
        self.assertTrue(report["is_clean"])

        # 核心断言: 单引文 NLI 调用次数严格 <= 2 次
        self.assertLessEqual(
            spy.call_count, 2,
            f"AC5 Failure: NLI scorer was invoked {spy.call_count} times for a single citation. "
            f"Two-stage coarse ranking must prune candidate chunks to Top-2 before NLI invocation (<= 2 calls)."
        )

    def test_ac5_multi_citation_linear_convergence(self):
        """验证多引文验证流程中，NLI 调用总次数 <= 2 * 引文总数."""
        from paperbrain.verifier import CitationVerifierV5

        gt_chunks = {
            "2024_NeurIPS_01_3.2": "Tiling attention into on-chip SRAM reduces memory traffic.",
            "2024_NeurIPS_01_4.1": "We evaluate throughput on LLaMA-7B reaching 2.4x speedup.",
            "2024_NeurIPS_01_5.0": "Ablation confirms backward recomputation saves 50% peak memory."
        }
        section_chunks = {
            "2024_NeurIPS_01_3.2": [f"Chunk 3.2-{i} on kernel tiling details" for i in range(6)],
            "2024_NeurIPS_01_4.1": [f"Chunk 4.1-{i} on benchmarking setup and throughput" for i in range(6)],
            "2024_NeurIPS_01_5.0": [f"Chunk 5.0-{i} on ablation memory reduction" for i in range(6)],
        }
        fig_index = {
            "2024_NeurIPS_01_3.2": {"figs": [], "tabs": []},
            "2024_NeurIPS_01_4.1": {"figs": [], "tabs": []},
            "2024_NeurIPS_01_5.0": {"figs": [], "tabs": []},
        }

        spy = SpyScorer(fixed_score=0.91)
        verifier = CitationVerifierV5(
            gt_chunks,
            fig_index,
            semantic_scorer=spy,
            section_chunks=section_chunks
        )

        draft = (
            "We tile attention into SRAM [Ref: 2024_NeurIPS_01, Sec 3.2]. "
            "Evaluations demonstrate 2.4x speedup [Ref: 2024_NeurIPS_01, Sec 4.1]. "
            "Ablation proves 50% memory saving [Ref: 2024_NeurIPS_01, Sec 5.0]."
        )
        report = verifier.verify_draft(draft)

        self.assertEqual(report["status"], "CLEAN")
        num_citations = len(report.get("report", []))
        self.assertEqual(num_citations, 3, "Draft must contain exactly 3 citations")

        # 核心断言: 平均每条引文 <= 2 次调用
        max_allowed_calls = 2 * num_citations
        self.assertLessEqual(
            spy.call_count, max_allowed_calls,
            f"AC5 Failure: NLI scorer called {spy.call_count} times for {num_citations} citations "
            f"(allowed <= {max_allowed_calls}). Calls per citation = {spy.call_count / num_citations:.2f}"
        )


# ==============================================================================
# AC6: Verification Caching (100% Cache Hit on Duplicate Query)
# ==============================================================================

class TestAC6VerificationCaching(unittest.TestCase):
    """AC6: 引文验真缓存 - 相同引文与上下文二次验证，缓存命中率 100%，额外调用为 0."""

    def test_ac6_repeat_verification_100_percent_cache_hit(self):
        """验证对相同引文与上下文执行二次验证时，调用增量为 0 (100% 缓存命中)."""
        from paperbrain.verifier import CitationVerifierV5

        gt_chunks = {
            "2024_NeurIPS_01_3.2": "Tiling attention into on-chip SRAM eliminates off-chip memory access."
        }
        section_chunks = {
            "2024_NeurIPS_01_3.2": [
                "Chunk 1: Tiling attention into on-chip SRAM eliminates off-chip memory access.",
                "Chunk 2: Mathematical derivation of online softmax rescaling factor."
            ]
        }
        fig_index = {"2024_NeurIPS_01_3.2": {"figs": [], "tabs": []}}

        spy = SpyScorer(fixed_score=0.92)
        verifier = CitationVerifierV5(
            gt_chunks,
            fig_index,
            semantic_scorer=spy,
            section_chunks=section_chunks
        )

        draft = "Our architecture tiles attention into on-chip SRAM [Ref: 2024_NeurIPS_01, Sec 3.2]."

        # 第一次调用: 必须执行真实打分并沉淀缓存
        report1 = verifier.verify_draft(draft)
        first_call_count = spy.call_count
        self.assertGreater(
            first_call_count, 0,
            "First verification call must invoke the semantic scorer"
        )
        self.assertEqual(report1["status"], "CLEAN")

        # 第二次调用 (完全相同引文与上下文): 必须 100% 命中缓存
        report2 = verifier.verify_draft(draft)
        second_call_count = spy.call_count
        call_delta = second_call_count - first_call_count

        # 核心断言: 额外模型调用为 0
        self.assertEqual(
            call_delta, 0,
            f"AC6 Failure: Repeated verification did not hit cache. "
            f"Initial calls: {first_call_count}, Second total calls: {second_call_count}, Delta: {call_delta}. "
            f"Expected delta == 0 with 100% cache hit."
        )
        self.assertEqual(report2["status"], report1["status"])
        self.assertEqual(report2["is_clean"], report1["is_clean"])

    def test_ac6_cache_invalidation_on_context_change(self):
        """验证上下文或 Claim 发生变更时，缓存键变更，触发正常新评估而不是错误复用."""
        from paperbrain.verifier import CitationVerifierV5

        gt_chunks = {
            "2024_NeurIPS_01_3.2": "Tiling attention into on-chip SRAM eliminates off-chip memory access."
        }
        section_chunks = {
            "2024_NeurIPS_01_3.2": [
                "Chunk 1: Tiling attention into on-chip SRAM eliminates off-chip memory access."
            ]
        }
        fig_index = {"2024_NeurIPS_01_3.2": {"figs": [], "tabs": []}}

        spy = SpyScorer(fixed_score=0.92)
        verifier = CitationVerifierV5(
            gt_chunks,
            fig_index,
            semantic_scorer=spy,
            section_chunks=section_chunks
        )

        draft_a = "Claim A: SRAM tiling speeds up attention [Ref: 2024_NeurIPS_01, Sec 3.2]."
        report_a = verifier.verify_draft(draft_a)
        calls_after_a = spy.call_count

        draft_b = "Claim B: Different context with distinct statement [Ref: 2024_NeurIPS_01, Sec 3.2]."
        report_b = verifier.verify_draft(draft_b)
        calls_after_b = spy.call_count

        # 断言: 不同 Claim 上下文不应误中 draft_a 的缓存
        self.assertGreater(
            calls_after_b, calls_after_a,
            "Cache must be specific to (citation, context) tuple; different statement must trigger evaluation"
        )


if __name__ == "__main__":
    unittest.main()
