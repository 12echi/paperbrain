"""前端页面脚本健康检查: 防止转义/引号问题导致整页 JS 失效 (点击无反应)."""
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import paperbrain.server as srv


class TestPage(unittest.TestCase):
    def test_script_extractable(self):
        m = re.search(r"<script>([\s\S]*?)</script>", srv.PAGE)
        self.assertIsNotNone(m)
        self.assertGreater(len(m.group(1)), 1000)

    def test_js_syntax_valid(self):
        if not shutil.which("node"):
            self.skipTest("无 node，跳过 JS 语法检查")
        m = re.search(r"<script>([\s\S]*?)</script>", srv.PAGE)
        fd, path = tempfile.mkstemp(suffix=".js")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(m.group(1))
        try:
            r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr[:500])
        finally:
            os.unlink(path)

    def test_required_elements_present(self):
        for el in ("bImport", "bRun", "tasks", "drop", "file", "path", "pid",
                   "tabs", "view", "mems", "mq"):
            self.assertIn('id="%s"' % el, srv.PAGE, el)
        # 四个部分标题
        for title in ("模型设置", "文件导入", "解读任务", "自动学习记忆"):
            self.assertIn(title, srv.PAGE)

    def test_no_paste_box(self):
        # 需求: 去掉粘贴框, 只允许文件导入
        self.assertNotIn("论文正文", srv.PAGE)
        self.assertNotIn('id="txt"', srv.PAGE)

    def test_batch_wiring(self):
        # 「开始解读」必须能自动识别批量选择 (曾 bug: 选了多文件仍弹"请先选择")
        self.assertIn("function submitBatch", srv.PAGE)
        self.assertIn("submitBatch(this)", srv.PAGE)
        self.assertIn("window.__manyFiles", srv.PAGE)
        for el in ("bBatch", "bpaths", "files", "dropMany", "bfiles"):
            self.assertIn('id="%s"' % el, srv.PAGE, el)

    def test_no_raw_emoji(self):
        # 需求: 去 emoji, 防字体缺字显示成乱码
        bad = [c for c in srv.PAGE if ord(c) > 0x1F000]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
