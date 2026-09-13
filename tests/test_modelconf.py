"""模型接入配置测试 (隔离配置文件, 不碰真Key)."""
import json
import os
import tempfile
import unittest


class TestModelConf(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["PAPERBRAIN_CONF_FILE"] = self.tmp + "/env.json"
        for k in ("PAPERBRAIN_API_KEY", "PAPERBRAIN_BASE_URL", "PAPERBRAIN_MODEL",
                  "PAPERBRAIN_PROVIDER", "PAPERBRAIN_CLOUD_ALLOWED"):
            os.environ.pop(k, None)

    def tearDown(self):
        os.environ.pop("PAPERBRAIN_CONF_FILE", None)
        for k in ("PAPERBRAIN_API_KEY", "PAPERBRAIN_BASE_URL", "PAPERBRAIN_MODEL",
                  "PAPERBRAIN_PROVIDER", "PAPERBRAIN_CLOUD_ALLOWED"):
            os.environ.pop(k, None)
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_masked_clear(self):
        from paperbrain import modelconf
        st = modelconf.save({"PAPERBRAIN_API_KEY": "sk-test1234567890",
                             "PAPERBRAIN_MODEL": "my-model"})
        self.assertTrue(st["has_key"])
        self.assertNotIn("sk-test1234567890", json.dumps(st))
        self.assertIn("****", st["api_key_masked"])
        self.assertEqual(os.environ["PAPERBRAIN_MODEL"], "my-model")
        # 落盘权限 600
        import stat
        mode = oct(os.stat(self.tmp + "/env.json").st_mode & 0o777)
        self.assertEqual(mode, "0o600")
        # 空保存不覆盖已有 Key
        st2 = modelconf.save({"PAPERBRAIN_API_KEY": ""})
        self.assertTrue(st2["has_key"])
        st3 = modelconf.clear()
        self.assertFalse(st3["has_key"])
        self.assertNotIn("PAPERBRAIN_API_KEY", os.environ)

    def test_no_key_test_fails_closed(self):
        from paperbrain import modelconf
        r = modelconf.test_connection()
        self.assertFalse(r["ok"])
        self.assertNotIn("sk-", json.dumps(r))

    def test_occli_parse_fixture(self):
        # 不真调: 用录制的 --format json 事件流测解析
        from paperbrain.llm import occli_chat
        import paperbrain.llm as _m
        fixture = ('{"type":"step_start","part":{"id":"a"}}\n'
                   '{"type":"text","part":{"type":"text","text":"你好"}}\n'
                   '{"type":"text","part":{"type":"text","text":"世界"}}\n'
                   'not-json-line\n')
        real_run = _m.subprocess.run

        class R:
            returncode = 0
            stdout = fixture
            stderr = ""
        _m.subprocess.run = lambda *a, **k: R()
        try:
            self.assertEqual(occli_chat([{"role": "user", "content": "hi"}], model="x/y"), "你好世界")
        finally:
            _m.subprocess.run = real_run

    def test_provider_default_https(self):
        os.environ.pop("PAPERBRAIN_PROVIDER", None)
        from paperbrain.llm import provider
        self.assertEqual(provider(), "https")

    def test_import_opencode_no_key_copy(self):
        import pathlib
        from paperbrain import modelconf
        if not pathlib.Path("/Users/liaosheng/.local/share/opencode/auth.json").exists():
            self.skipTest("无opencode环境")
        r = modelconf.import_opencode("muse-spark-1.3-contributor")
        self.assertTrue(r.get("ok"), r.get("error"))
        self.assertEqual(os.environ.get("PAPERBRAIN_BASE_URL"),
                         "https://opencode.ai/zen/go/v1")
        saved = json.loads(__import__("pathlib").Path(self.tmp, "env.json").read_text(encoding="utf-8"))
        blob = json.dumps(saved)
        self.assertNotIn("sk-", blob)  # Key 不得落本文件
        self.assertTrue(saved.get("opencode_go"))
        self.assertNotIn("PAPERBRAIN_CLOUD_ALLOWED", saved)
        self.assertFalse(r["cloud_allowed"], "导入凭据不得隐式授权上传论文内容")
        modelconf.clear()

    def test_config_env_file_fallback(self):
        """CLI/工具 不走 modelconf.load() 也能读 env.json 的 PAPERBRAIN_* 键; 环境变量优先;
        非 PAPERBRAIN_ 前缀键不得进入配置。"""
        from paperbrain import config
        with open(self.tmp + "/env.json", "w", encoding="utf-8") as f:
            json.dump({"PAPERBRAIN_EMBED_BASE_URL": "http://x:1/v1",
                       "PAPERBRAIN_EMBED_API_KEY": "sk-lm-test",
                       "model": "unprefixed-should-not-leak"}, f)
        os.environ.pop("PAPERBRAIN_EMBED_BASE_URL", None)
        os.environ["PAPERBRAIN_TEST"] = "0"  # 放行兜底 (测试进程默认禁用)
        try:
            self.assertEqual(config.embed_base_url(), "http://x:1/v1")
            self.assertTrue(config.embed_enabled())
            os.environ["PAPERBRAIN_EMBED_BASE_URL"] = "http://y:2/v1"
            self.assertEqual(config.embed_base_url(), "http://y:2/v1")  # 环境变量优先
            os.environ.pop("PAPERBRAIN_EMBED_BASE_URL", None)
            self.assertEqual(config.env("model", "d"), "d")  # 非 PAPERBRAIN_ 前缀不读取
        finally:
            os.environ.pop("PAPERBRAIN_TEST", None)

    def test_config_fallback_blocked_under_unittest(self):
        from paperbrain import config
        with open(self.tmp + "/env.json", "w", encoding="utf-8") as f:
            json.dump({"PAPERBRAIN_EMBED_BASE_URL": "http://x:1/v1"}, f)
        os.environ.pop("PAPERBRAIN_EMBED_BASE_URL", None)
        os.environ.pop("PAPERBRAIN_TEST", None)  # 未显式设置: unittest 进程自动禁用
        self.assertEqual(config.embed_base_url(), "")

    def test_flag_reads_env_file_fallback(self):
        from paperbrain import config
        with open(self.tmp + "/env.json", "w", encoding="utf-8") as f:
            json.dump({"PAPERBRAIN_REFLECT": "0", "PAPERBRAIN_VISION": "1"}, f)
        os.environ["PAPERBRAIN_TEST"] = "0"  # 放行兜底
        try:
            self.assertFalse(config.reflect_enabled(), "文件里 REFLECT=0 必须生效")
            self.assertTrue(config.vision_enabled(), "文件里 VISION=1 必须生效")
        finally:
            os.environ.pop("PAPERBRAIN_TEST", None)

    def test_clear_drops_stale_cache(self):
        from paperbrain import config
        with open(self.tmp + "/env.json", "w", encoding="utf-8") as f:
            json.dump({"PAPERBRAIN_EMBED_BASE_URL": "http://stale:9/v1",
                       "PAPERBRAIN_EMBED_API_KEY": "sk-stale-secret"}, f)
        os.environ["PAPERBRAIN_TEST"] = "0"
        os.environ.pop("PAPERBRAIN_EMBED_BASE_URL", None)
        try:
            self.assertEqual(config.embed_base_url(), "http://stale:9/v1")
            os.unlink(self.tmp + "/env.json")
            self.assertEqual(config.embed_base_url(), "", "文件删除后不得返回旧地址")
            self.assertEqual(config.embed_api_key(), "", "文件删除后不得返回旧 Key")
        finally:
            os.environ.pop("PAPERBRAIN_TEST", None)


if __name__ == "__main__":
    unittest.main()
