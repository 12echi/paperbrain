"""批量处理测试: 收集/ID/断点续跑/失败隔离/HTTP 端点 (全部离线)."""
import base64
import csv
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


class TestBatchModule(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(dir="/tmp")  # /tmp 在路径白名单内
        self.out = tempfile.mkdtemp(dir="/tmp")
        for i in range(3):
            Path(self.tmp, f"paper{i}.txt").write_text(
                "Abstract\nThis is abstract for paper %d with enough words.\n"
                "1. Introduction\nIntro text about DBSCAN DNA damage LET.\n" % i,
                encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.out, ignore_errors=True)

    def test_collect_and_ids(self):
        from paperbrain.batch import collect_files, derive_paper_id
        files = collect_files(dirs=[self.tmp], pattern="*.txt")
        self.assertEqual(len(files), 3)
        self.assertEqual(derive_paper_id("/a/b/My Paper (1).pdf"), "My_Paper_1")

    def test_run_batch_offline_and_resume(self):
        from paperbrain.batch import collect_files, run_batch
        files = collect_files(dirs=[self.tmp], pattern="*.txt")
        items = [{"path": f} for f in files]
        prog = []
        r1 = run_batch(items, self.out, task="outline", depth="fast", use_llm=False,
                       on_progress=lambda i, n, row: prog.append((i, row["verify"])))
        self.assertEqual(r1["done"], 3)
        self.assertEqual(r1["failed"], 0)
        self.assertEqual(r1["quality_unverified"], 3)
        # 每篇先 RUNNING 后终态
        self.assertIn((1, "RUNNING"), prog)
        finals = [p for p in prog if p[1] != "RUNNING"]
        self.assertEqual([p[0] for p in finals], [1, 2, 3])
        self.assertEqual([p[1] for p in finals], ["NOT_RUN"] * 3)
        self.assertTrue(Path(r1["ledger"]).exists())
        with open(r1["ledger"], newline="", encoding="utf-8") as f:
            ledger_rows = list(csv.DictReader(f))
        self.assertEqual(len(ledger_rows), 3)
        self.assertEqual(set(ledger_rows[0]), {
            "source_sha256", "run_signature", "text_tokens", "vision_tokens", "total",
            "input_tokens", "output_tokens", "llm_calls", "cost"})
        self.assertNotIn("paper0.txt", Path(r1["ledger"]).read_text(encoding="utf-8"))
        # 断点续跑: 全部跳过
        r2 = run_batch(items, self.out, task="outline", depth="fast", use_llm=False)
        self.assertEqual(r2["skipped"], 3)
        self.assertEqual(r2["done"], 0)
        self.assertEqual(r2["quality_unverified"], 3)
        # force 重跑
        r3 = run_batch(items, self.out, task="outline", depth="fast", use_llm=False,
                       resume=False)
        self.assertEqual(r3["done"], 3)

    def test_run_batch_failure_isolated(self):
        from paperbrain.batch import run_batch
        bad = Path(self.tmp, "missing.txt")  # 不存在 -> 该篇 ERROR, 不影响其他
        good = Path(self.tmp, "paper0.txt")
        r = run_batch([{"path": str(bad)}, {"path": str(good)}], self.out,
                      task="outline", depth="fast", use_llm=False)
        self.assertEqual(r["failed"], 1)
        self.assertEqual(r["done"], 1)
        statuses = {row["paper_id"]: row["verify"] for row in r["results"]}
        self.assertIn("ERROR", statuses.values())

    def test_resume_is_task_aware(self):
        # 先跑 outline, 再跑 full_read: 不能因"有报告"就跳过 (修复 over-skip)
        from paperbrain.batch import collect_files, run_batch
        items = [{"path": f} for f in collect_files(dirs=[self.tmp], pattern="*.txt")]
        run_batch(items, self.out, task="outline", depth="fast", use_llm=False)
        r2 = run_batch(items, self.out, task="full_read", depth="fast", use_llm=False)
        self.assertEqual(r2["skipped"], 0)
        # 同任务同深度再来一次 -> 全跳过
        r3 = run_batch(items, self.out, task="full_read", depth="fast", use_llm=False)
        self.assertEqual(r3["skipped"], 3)

    def test_resume_rejects_legacy_and_changed_input_or_options(self):
        from paperbrain.batch import derive_paper_id, run_batch
        src = Path(self.tmp, "paper0.txt")
        pid = derive_paper_id(str(src))
        od = Path(self.out, pid)
        od.mkdir(parents=True)
        (od / "report.md").write_text("legacy CLEAN", encoding="utf-8")
        (od / "verify.json").write_text('{"status":"CLEAN"}', encoding="utf-8")

        # 无版本旧产物不能被当成已完成。
        first = run_batch([{"path": str(src)}], self.out, task="outline", depth="fast",
                          use_llm=False)
        self.assertEqual(first["done"], 1)
        self.assertEqual(first["skipped"], 0)
        again = run_batch([{"path": str(src)}], self.out, task="outline", depth="fast",
                          use_llm=False)
        self.assertEqual(again["skipped"], 1)

        # 合法标记不能掩盖被删除或篡改的产物。
        (od / "report.md").write_text("tampered", encoding="utf-8")
        tampered = run_batch([{"path": str(src)}], self.out, task="outline", depth="fast",
                             use_llm=False)
        self.assertEqual(tampered["done"], 1)
        self.assertEqual(tampered["skipped"], 0)

        # 同一路径内容变化，以及会改变产出的 focus 变化，都必须失效续跑缓存。
        src.write_text(src.read_text(encoding="utf-8") + "\nchanged evidence", encoding="utf-8")
        changed = run_batch([{"path": str(src)}], self.out, task="outline", depth="fast",
                            use_llm=False)
        self.assertEqual(changed["done"], 1)
        option_changed = run_batch([{"path": str(src)}], self.out, task="outline",
                                   depth="fast", focus="new focus", use_llm=False)
        self.assertEqual(option_changed["done"], 1)

    def test_resume_invalidates_when_graph_policy_changes_in_same_process(self):
        from unittest.mock import patch
        import paperbrain.batch as batch
        source = Path(self.tmp, "paper0.txt")
        policy = Path(self.tmp, "graph_policy.json")
        base = {"schema": "paperbrain-graph-policy-v1", "reviewed_at": None,
                "aliases": {}, "blacklist": []}
        policy.write_text(json.dumps(base), encoding="utf-8")
        with patch.dict(os.environ, {"PAPERBRAIN_GRAPH_POLICY": str(policy)}, clear=False):
            first = batch.run_batch([{"path": str(source)}], self.out, task="outline",
                                    depth="fast", use_llm=False)
            self.assertEqual(first["done"], 1)
            self.assertEqual(batch.run_batch([{"path": str(source)}], self.out, task="outline",
                                             depth="fast", use_llm=False)["skipped"], 1)
            base["blacklist"] = ["new generic"]
            policy.write_text(json.dumps(base), encoding="utf-8")
            changed = batch.run_batch([{"path": str(source)}], self.out, task="outline",
                                      depth="fast", use_llm=False)
            self.assertEqual(changed["done"], 1)
            self.assertEqual(changed["skipped"], 0)


class TestBatchHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import paperbrain.server as srvmod
        cls.tmp = tempfile.mkdtemp(dir="/tmp")
        for i in range(2):
            Path(cls.tmp, f"http{i}.txt").write_text(
                "Abstract\nabstract text %d with enough words here.\n"
                "1. Introduction\nintro about DBSCAN DNA LET.\n" % i, encoding="utf-8")
        # 隔离产物目录, 避免测试写脏真实 out/web (含 batch_ledger.csv)
        cls._orig_out = srvmod.OUT_ROOT
        srvmod.OUT_ROOT = Path(cls.tmp) / "out"
        cls.srv = HTTPServer(("127.0.0.1", 0), srvmod.H)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        import paperbrain.server as srvmod
        srvmod.OUT_ROOT = cls._orig_out
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        for i in range(2):
            shutil.rmtree(Path("out/web") / f"http{i}", ignore_errors=True)

    def _post(self, path, obj):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=json.dumps(obj).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=30) as r:
            return json.loads(r.read().decode())

    def test_batch_endpoint(self):
        paths = [str(Path(self.tmp, "http0.txt")), str(Path(self.tmp, "http1.txt"))]
        r = self._post("/api/batch", {"task": "outline", "depth": "fast", "paths": paths})
        self.assertTrue(r["ok"])
        self.assertEqual(r["total"], 2)
        jid = r["job"]
        for _ in range(120):
            j = self._get("/api/job?id=" + jid)
            if j.get("done"):
                break
            time.sleep(0.3)
        self.assertTrue(j.get("done"))
        self.assertTrue(j.get("ok"), j.get("error"))
        self.assertFalse(j.get("quality_ok"))
        self.assertEqual(j["result"]["done"], 2)
        self.assertEqual(len(j.get("files", [])), 2)

    def test_batch_rejects_outside_path(self):
        r = self._post("/api/batch", {"paths": ["/etc/passwd"]})
        self.assertFalse(r["ok"])
        self.assertIn("允许范围", r.get("error", ""))


if __name__ == "__main__":
    unittest.main()
