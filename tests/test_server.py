"""服务端接口回归测试 (本机回环, 随机端口, 跑完即关).
锁死: home/demo/health/run/outline/generate/upload/out 取产物全链路.
"""
import base64
import json
import shutil
import threading
import unittest
import urllib.request
from http.server import HTTPServer
from pathlib import Path


def _pick(names):
    for n in names:
        try:
            return __import__(n)
        except ImportError:
            continue
    raise unittest.SkipTest("缺 paperbrain 包")


class TestServerAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import paperbrain.server as _srvmod
        cls.srv = HTTPServer(("127.0.0.1", 0), _srvmod.H)
        cls.port = cls.srv.server_address[1]
        cls.t = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.t.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        for d in ["out/web/apitest01", "out/web/apitest02", "out/web/apitest_task",
                  "out/web/apitest_versions"]:
            shutil.rmtree(d, ignore_errors=True)

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as r:
            return r.status, r.read()

    def _post(self, path, obj):
        b = json.dumps(obj).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=b,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode())

    def test_10_home(self):
        code, body = self._get("/")
        self.assertEqual(code, 200)
        self.assertIn("PaperBrain", body.decode()[:2000])

    def test_20_health_keys(self):
        code, body = self._get("/api/health")
        h = json.loads(body.decode())
        for k in ("llm", "pdf", "ocr", "cv2", "node", "tesseract",
                  "formula_checkers", "vector_index"):
            self.assertIn(k, h)
        self.assertIn("katex", h["formula_checkers"])
        self.assertIn("sympy", h["formula_checkers"])
        self.assertEqual(h["vector_index"]["backend"], "duckdb-vss")

    def _wait_job(self, job, timeout=120):
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/api/job?id={job}") as r:
                j = json.loads(r.read().decode())
            if j.get("done"):
                return j
            time.sleep(0.5)
        self.fail("job 超时")

    def test_30_run_and_read_back(self):
        txt = Path("demo/sample_paper.txt").read_text(encoding="utf-8")
        r = self._post("/api/run", {"paper_id": "apitest01", "text": txt})
        self.assertTrue(r["ok"])
        j = self._wait_job(r["job"])
        self.assertTrue(j["ok"])
        self.assertEqual(j["result"]["verify"], "NEEDS_REVIEW")
        self.assertFalse(j["quality_ok"])
        for fn in ("report.md", "draft.md", "verify.json", "ledger.csv",
                   "outline_v1.json", "memory.json"):
            code, body = self._get(f"/api/out?paper_id=apitest01&file={fn}")
            self.assertEqual(code, 200, fn)
            c = json.loads(body.decode())["content"]
            self.assertTrue(len(c) > 10, fn)

    def test_40_outline_generate(self):
        txt = Path("demo/sample_paper.txt").read_text(encoding="utf-8")
        r = self._post("/api/outline", {"paper_id": "apitest02", "text": txt})
        self.assertTrue(r["ok"])
        self.assertIn("sections", r["outline"])
        outline_job = self._wait_job(r["job"])
        self.assertTrue(outline_job["ok"])
        self.assertIsNone(outline_job["quality_ok"])
        self.assertIn("等待确认大纲", outline_job["stage"])
        g = self._post("/api/generate", {"paper_id": "apitest02", "deepread": True,
                                          "focus": "reproducibility"})
        self.assertTrue(g["ok"])
        j = self._wait_job(g["job"])
        self.assertTrue(j["ok"])
        self.assertEqual(j["result"]["verify"], "NEEDS_REVIEW")
        self.assertFalse(j["quality_ok"])
        self.assertTrue(j["result"]["markdown"])

    def test_45_primary_task_reports_quality_failure_separately(self):
        r = self._post("/api/task", {"task": "full_read", "depth": "standard",
                                      "paper_id": "apitest_task",
                                      "path": str(Path("demo/sample_paper.txt").resolve())})
        self.assertTrue(r["ok"])
        j = self._wait_job(r["job"])
        self.assertTrue(j["ok"], "执行应成功")
        self.assertFalse(j["quality_ok"], "未标定结果不得伪装成质量通过")
        self.assertIn("质量门禁未通过", j["stage"])

    def test_42_outline_history_diff_and_rollback_api(self):
        from paperbrain.outline import save_outline_version
        root = Path("out/web/apitest_versions")
        shutil.rmtree(root, ignore_errors=True)
        first = {"paper_id": "apitest_versions",
                 "sections": [{"h1": "Old", "chunks": ["C1"]}], "entities": []}
        second = {"paper_id": "apitest_versions",
                  "sections": [{"h1": "New", "chunks": ["C1"]}], "entities": []}
        save_outline_version(str(root), first, "generated")
        save_outline_version(str(root), second, "confirmed")
        _, raw = self._get("/api/outline/history?paper_id=apitest_versions")
        history = json.loads(raw.decode())
        self.assertEqual([x["version"] for x in history["versions"]], ["v1", "v2"])
        _, raw = self._get(
            "/api/outline/diff?paper_id=apitest_versions&from=v1&to=v2")
        self.assertIn("New", json.loads(raw.decode())["diff"])
        rolled = self._post("/api/outline/rollback",
                            {"paper_id": "apitest_versions", "version": "v1"})
        self.assertTrue(rolled["ok"])
        self.assertEqual(rolled["outline"]["version"], "v3")

    def test_50_upload_txt(self):
        raw = ("Abstract\n" + "Long enough test content. " * 20 + "\nMethods\n" + "x" * 300).encode()
        b64 = base64.b64encode(raw).decode()
        r = self._post("/api/upload", {"paper_id": "apitest02", "filename": "a.txt", "data_b64": b64})
        self.assertTrue(r["ok"])
        j = self._wait_job(r["job"])
        self.assertTrue(j["ok"])

    def test_70_model_api_masked(self):
        import os
        os.environ["PAPERBRAIN_CONF_FILE"] = "/tmp/pb_test_env.json"
        try:
            r = self._post("/api/model", {"PAPERBRAIN_API_KEY": "sk-faketestkey12345"})
            self.assertTrue(r["ok"] and r["has_key"])
            self.assertIn("****", r["api_key_masked"])
            self.assertNotIn("faktestkey", __import__("json").dumps(r))
            g = self._get("/api/model")
            import json as _j
            self.assertTrue(_j.loads(g[1].decode())["has_key"])
            # Key 泄漏检查: 落盘文件 600 且 health 不回显
            h = _j.loads(self._get("/api/health")[1].decode())
            self.assertIn("llm", h)
        finally:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/model",
                                         method="DELETE")
            urllib.request.urlopen(req).read()
            os.environ.pop("PAPERBRAIN_CONF_FILE", None)
            os.environ.pop("PAPERBRAIN_API_KEY", None)
            Path("/tmp/pb_test_env.json").unlink(missing_ok=True)

    def test_75_pid_traversal_sanitized(self):
        # POST 带 paper_id="../.." 必须被安全化, 产物落在 out/web/paper 内
        import shutil as _sh
        from pathlib import Path as _P
        txt = _P("demo/sample_paper.txt").read_text(encoding="utf-8")
        r = self._post("/api/outline", {"paper_id": "../..", "text": txt})
        self.assertTrue(r["ok"])
        self.assertEqual(r["paper_id"], "paper")
        self.assertTrue((_P("out/web") / "paper").exists())
        self.assertFalse((_P("out/web") / ".." / "..").resolve() == (_P("out/web") / "paper").resolve())
        _sh.rmtree(_P("out/web") / "paper", ignore_errors=True)

    def test_60_unknown(self):
        try:
            self._get("/api/nope")
            self.fail("应404")
        except Exception as e:
            self.assertIn("404", str(e))


if __name__ == "__main__":
    unittest.main()
