# PaperBrain

本地运行的文献精读工作台，支持文档导入、分章解读、引文检查与跨论文记忆检索。可选连接 OpenAI 兼容模型接口或本机 opencode 配置。

> 当前为开发版本，尚未完成生产验收。模型回答、引文检查和自动记忆复核都不保证 100% 准确；研究结论仍需核对原文。

## 快速开始

需要 Python 3.9+，在项目根目录执行：

```bash
git clone https://github.com/12echi/paperbrain.git
cd paperbrain
python3 -m venv .venv
source .venv/bin/activate
python3 -m paperbrain.server --port 8000
```

打开 http://127.0.0.1:8000 。TXT / Markdown 基础流程无需额外 Python 依赖；PDF 解析可安装 `python3 -m pip install pymupdf==1.26.5`。其他可选依赖见 `pyproject.toml`：OCR 需要系统 Tesseract，公式检查需要 Node/KaTeX 与 SymPy，向量检索需要 DuckDB VSS。工具缺失时相关能力会降级。

## 模型与数据

- 在页面中配置模型，并单独授权云端发送后，才启用模型调用；未授权时使用离线规则，不等价于模型精读。
- 自动记忆复核基于原文证据和模型判断。可选联网复核查询 Crossref 文献摘要，并非完整互联网事实核查；证据不足时保留待复核状态。
- 模型服务可能收费。授权后论文片段会发送到配置的服务；联网复核另需开启相应选项。
- 运行产物、上传文档和数据库位于 `out/`，不随仓库发布。模型配置保存在用户目录下的 `~/.config/paperbrain/env.json`，不得提交凭据。
- 服务没有账户认证，仅用于可信本机环境，不要直接暴露至公网。

## 开发与验收

```bash
python3 -m unittest discover -s tests -v
python3 tools/release_gate.py
```

单元测试通过不代表生产验收通过。发布门禁还需要标注数据、引文标定及人工评审等材料，详见 `AGENTS.md` 和 v5.0 执行方案。`demo/` 是人工构造的测试材料，其中论文名称、指标与结论不能作为真实研究证据。

主要目录：`paperbrain/` 为应用核心与网页，`tests/` 为测试，`tools/` 为评测和门禁工具。Docker 配置提供可选运行方式，本次公开发布不代表已验证 Docker 构建。

## 许可

尚未选择开源许可证。公共可见不等于已授予任意使用、修改或分发许可；第三方依赖遵循各自许可证。
