<div align="center">

# PaperBrain · 文献精读工作台

**简体中文** · [English](README.md)

</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](pyproject.toml)

在本机导入论文，按章节解读、检查引用，再把带来源的笔记组织成可检索的学习记忆。支持可选的远端模型调用，不要求安装大型本地语言模型。

**项目状态：开发中，尚未完成生产验收。** 模型输出、引文评分和自动记忆复核不保证 100% 准确。请用于辅助阅读，不要替代原文核对、实验复现或人工同行评审。

## 目录

- [能做什么](#能做什么)
- [快速开始](#快速开始)
- [可选工具链](#可选工具链)
- [模型配置与授权](#模型配置与授权)
- [使用流程](#使用流程)
- [自动学习与复核机制](#自动学习与复核机制)
- [批量命令行](#批量命令行)
- [输出与数据位置](#输出与数据位置)
- [Docker 运行](#docker-运行)
- [常见问题](#常见问题)
- [开发与验收](#开发与验收)
- [安全与隐私](#安全与隐私)
- [许可证](#许可证)

## 能做什么

| 功能 | 当前实现与边界 |
| --- | --- |
| 文档导入 | PDF、TXT、Markdown；支持拖拽、多选和本机路径。PDF/OCR 依赖可选工具链。 |
| 分章阅读 | 按摘要、引言、方法、实验等组织内容，提取纪要与引用片段；低置信解析会降级。 |
| 多任务解读 | 全文精读、速览大纲、方法拆解、图表识别、审稿批判、记忆沉淀。模型不可用时能力受限。 |
| 大纲与草稿 | 先生成大纲，确认后再生成草稿；深度批处理也不绕过确认步骤。 |
| 引文检查 | 检查来源标识、图表引用及语义支持；缺证据或工具不可用时保留待复核状态。 |
| 学习记忆 | 保存笔记、实体和关系，支持跨论文检索及带 `[M#]` 来源标记的问答。 |
| 自动复核 | 对待验证或争议笔记重新取证并调用模型复核，可选查询 Crossref 摘要补证。 |
| 本地优先 | 核心采用 Python 标准库和 SQLite；模型、视觉和向量服务按需配置。 |

```text
导入文档 → 解析与分章 → 解读与来源绑定 → 引文/主张检查
                                      ├─ 证据支持 → 有效记忆 → 检索与问答
                                      ├─ 证据不足 → 待复核
                                      └─ 证据反驳 → 拒绝，保留审计记录
```

这里的“支持”是系统根据取得的证据作出的判断，不等价于已证明主张客观正确。

## 快速开始

需要 **Python 3.9 或更新版本**。以下为 macOS / Linux shell 命令；Windows 兼容性尚未完成验证。

```bash
git clone https://github.com/12echi/paperbrain.git
cd paperbrain
python3 -m venv .venv
source .venv/bin/activate
python3 -m paperbrain.server --port 8000
```

打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。保持终端运行，按 `Ctrl+C` 停止服务。后续启动时，先进入仓库并激活 `.venv`。

TXT / Markdown 的基础离线流程无需额外 Python 包。首次试用可以导入 `demo/sample_paper.txt`；该文件是**人工构造的测试材料**，不是可用于引用的真实论文。

PDF 用户在已激活的虚拟环境中安装：

```bash
python3 -m pip install pymupdf==1.26.5
```

工具检测入口：[http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)。页面能打开不代表所有解析、公式或模型能力都已就绪。

## 可选工具链

按实际需求安装，不必一次装齐。依赖声明见 [pyproject.toml](pyproject.toml)。

| 用途 | 依赖 | 缺失时的影响 |
| --- | --- | --- |
| PDF 文本与图片解析 | PyMuPDF | PDF 能力降级，不能视为完整解析。 |
| 扫描件 OCR | Tesseract；中文材料需要中文语言数据 | 扫描页可能无法提取有效文本。 |
| 公式检查 | Node.js + KaTeX、SymPy + ANTLR | 相关公式标记为未验证或待复核。 |
| 图像处理 | OpenCV | 部分图像分析能力受限。 |
| 向量检索 | Embedding 接口 + DuckDB VSS 扩展 | 无法使用向量后端时退回词法检索。 |

示例安装命令（macOS 已安装 Homebrew 的情况）：

```bash
# 扫描件与中文 OCR
brew install tesseract tesseract-lang

# 公式检查
brew install node
npm install -g katex@0.18.7
python3 -m pip install sympy==1.14.0 'antlr4-python3-runtime==4.11.*'

# 按需安装图像、向量和评测依赖
python3 -m pip install opencv-python-headless duckdb==1.4.5 rapidfuzz==3.13.0
```

包安装成功不代表 DuckDB 的 VSS 扩展已可加载，也不代表 Embedding 服务已配置。KaTeX 使用非常规全局安装路径时，也可能无法被检测到。以健康检查和实际运行报告为准。

## 模型配置与授权

### 网页配置

1. 展开“模型设置”，填写服务商提供的 Base URL、API Key、文本模型名称；读图时还需可用的视觉模型。
2. 单独勾选云端发送授权，确认允许论文片段及所选配图发送到该服务。
3. 保存配置并测试连通，再开始解读。

也可以通过页面的 opencode 接入入口复用本机已保存的相应服务凭据。这不包含免费模型额度，也不意味着任意模型都可用，仍取决于本机配置与服务商。

**“已配置但未授权”不是报错**：它表示远端模型发送尚未获准。未授权时使用离线规则，离线摘录不等价于模型精读或审稿。

### 环境变量

在启动服务的同一个终端设置。下面的地址和模型名是占位值，需要替换：

```bash
export PAPERBRAIN_BASE_URL='https://your-provider.example/v1'
export PAPERBRAIN_MODEL='your-text-model'
# API Key 建议通过页面输入，避免写入 shell 历史。
# 如果从环境注入，请设置 PAPERBRAIN_API_KEY。
export PAPERBRAIN_CLOUD_ALLOWED=1
python3 -m paperbrain.server --port 8000
```

项目不会自动加载 `.env` 文件。通常配置优先级为：**进程环境变量 → 本地配置文件 → 代码默认值**。环境变量存在时，修改配置文件可能不会覆盖它。

| 变量 | 作用 / 默认行为 |
| --- | --- |
| `PAPERBRAIN_BASE_URL` | OpenAI 兼容接口地址，应与模型和凭据属于同一服务。 |
| `PAPERBRAIN_API_KEY` | 文本模型接口凭据，不要提交到 Git。 |
| `PAPERBRAIN_MODEL` | 文本模型名称，需使用服务商支持的标识。 |
| `PAPERBRAIN_CLOUD_ALLOWED` | 远端模型发送授权，默认关闭。 |
| `PAPERBRAIN_WEB_VERIFY` | Crossref 补证查询，默认关闭；主张文本会用于外部查询。 |
| `PAPERBRAIN_VL_MODEL` | 视觉模型名称；另需在运行选项中启用读图。 |
| `PAPERBRAIN_ALL_MODEL` | 模型参与分章、图谱抽取、语义判断等阶段，默认关闭。 |
| `PAPERBRAIN_EMBED_BASE_URL` | 可选 Embedding 接口，未配置时不启用。 |
| `PAPERBRAIN_EMBED_API_KEY` / `PAPERBRAIN_EMBED_MODEL` | Embedding 凭据与模型。 |
| `PAPERBRAIN_REFLECT` | 自动反思开关，默认开启；模型不可用时能力受限。 |
| `PAPERBRAIN_CONF_FILE` / `PAPERBRAIN_MEMORY_DB` | 覆盖配置文件或记忆数据库位置。 |

完整配置见 [paperbrain/config.py](paperbrain/config.py)。本机回环地址的 Embedding 服务可无需云端授权；如果地址实际经 SSH 隧道转发到远端，数据仍会离开本机，需自行确认目标与权限。

## 使用流程

1. **导入材料**：拖入 PDF / TXT / Markdown，或展开高级导入输入本机路径。网页上传单文件限制为 20 MB；多文件进入批量流程。
2. **选择目标**：选全文精读、方法拆解或其他任务，可填写关注点。低置信材料应先检查原文解析质量。
3. **选择深度**：`fast`、`standard`、`deep` 控制处理深度，具体阶段取决于任务类别。更深不保证更准确，通常增加模型调用和等待时间。
4. **查看结果**：阅读精读成果、完整分析和事实核验；“更多产物”中查看运行报告、大纲、草稿或图谱。不是每个任务都会生成所有文件。
5. **确认大纲**：涉及草稿生成时，需确认大纲后再生成相应草稿。
6. **检索记忆**：解读后自动提取笔记；提问时核对 `[M#]` 来源，不要只看答案正文。

`CLEAN` 表示通过了当前检查链，不是“论文已被证明正确”。`NEEDS_REVIEW`、`NO_CITATION`、`NOT_RUN` 分别表示需要复核、缺少引文或未运行对应检查，不能当作通过。

## 自动学习与复核机制

系统保存来源绑定，并区分 `active`（有效）、`candidate`（待验证）、`contested`（争议）和 `rejected`（拒绝）等状态。正常可信问答排除待验证、争议和拒绝笔记。

自动复核实现在 [paperbrain/adjudication.py](paperbrain/adjudication.py)：

1. 服务启动、模型配置保存或学习收尾时，在获得云端授权的情况下触发后台复核。
2. 从待验证/争议笔记中取一小批，查找对应论文的 `state.json` 原文片段。
3. 要求模型给出“支持 / 反驳 / 无法判断”和逐字证据；程序检查证据标识与引文是否存在于输入片段中。
4. 原文判断不明确且开启联网补证时，查询 Crossref，以带 DOI 的可用摘要补充证据。
5. 对明确支持或反驳的结论再次调用模型复核；意见不一致时保留待复核状态。
6. 支持的内容可进入有效记忆；被反驳的内容标为拒绝、退出有效记忆，**保留审计记录而非物理删除**。

### 当前限制

- 默认每次最多处理 **8 条**，24 小时内尝试过的条目暂不重复处理；不是持续运行直至清空所有积压的后台服务。
- 缺少原论文材料时，不会仅凭联网搜索强行入库。目前原文查找依赖默认 `out/web/` 布局，自定义输出路径可能导致无法绑定。
- Crossref 摘要检索不是全文网络搜索；不少记录没有摘要，搜索结果也不一定相关。
- 两次模型判断不是两个独立事实来源，更不是准确率证明。
- 引文逐字匹配只能证明文字来自给定材料，不能证明材料本身可靠，也不能完全排除模型错误解释。
- 保留待复核内容是正常结果；没有证据不能推导出“错误”，不能自动删除所有不确定内容。
- 自动复核会额外消耗模型调用。未授权、接口失败或证据不足时，不应期待自动吸收完成。

## 批量命令行

在项目根目录、已激活虚拟环境的终端执行：

```bash
# 使用两个合成 TXT 示例
python3 tools/batch.py demo/sample_paper.txt demo/sample_paper2.txt \
  --out out/batch --task full_read --depth fast

# 替换为自己的论文目录
python3 tools/batch.py /path/to/papers \
  --pattern '*.pdf' --out out/batch --task full_read --depth standard

python3 tools/batch.py --help
```

同输入、代码和选项且产物哈希完整时可跳过已完成论文。`--force` 会忽略续跑判断并重新处理，使用前确认输出目录和成本。CLI 自定义输出目录的结果不保证被当前自动复核的默认路径查找发现。

## 输出与数据位置

| 默认位置 | 内容 |
| --- | --- |
| `out/web/` | 网页任务产物、上传材料及任务状态。 |
| `out/batch/` | 命令行批量任务默认输出。 |
| `out/memory.sqlite` | 跨论文笔记、关系与复核记录。 |
| `out/memory.sqlite.vectors.duckdb` | 默认向量数据库，按需生成。 |
| `~/.config/paperbrain/env.json` | 模型配置，可能包含凭据。 |

常见产物包括 `state.json`、`report.md`、`deepread.md`、`deepread_full.md`、`deepread_verify.json`、`outline_v1.json`、`draft.md`、`verify.json` 和 `memory.json`，以实际任务输出为准。

备份时建议先停止服务，再备份 `out/` 和必要配置；配置备份应按敏感文件保管。不要只保留记忆数据库而丢弃原文与状态文件，否则复核可能失去证据。Git 忽略规则不等于加密或访问控制。

## Docker 运行

仓库提供 [Dockerfile](Dockerfile) 和 [docker-compose.yml](docker-compose.yml)，作为可选运行方式：

```bash
docker compose up --build
# 停止容器
docker compose down
```

默认绑定本机 `127.0.0.1:8000`，持久化挂载 `./out`，容器限额为 3 GB 内存 / 2 CPU。构建会下载 Python 包、系统工具、KaTeX 与 DuckDB 扩展，需要网络且可能失败。

**本次文档更新未验证 Docker 构建。** 主机模型配置不会自动挂载到容器；在容器页面保存的配置若未额外持久化，重建后可能丢失。容器不能直接读取任意主机论文路径，应使用网页上传或设置受控挂载。

## 常见问题

| 现象 | 检查方法 |
| --- | --- |
| 网页打不开 | 确认启动终端仍在运行；端口冲突时使用 `--port 8001` 并打开对应地址。 |
| 模型已配置但不调用 | 检查独立云端授权、凭据、接口地址、模型名和服务额度；不要把密钥贴入 Issue。 |
| 内容像简单摘录 | 查看是否处于离线规则模式、工具降级或解析置信度不足。 |
| PDF 没有正文 | 检查 PyMuPDF；扫描件检查 Tesseract 与对应语言数据。 |
| 公式总是待验证 | 检查 Node/KaTeX 和 SymPy/ANTLR，不要删除警告伪装通过。 |
| 可信记忆数量为 0 | 可能只有待复核笔记，或原文、模型授权、证据不满足条件，不等于没有导入论文。 |
| 隔离区没有全部消失 | 自动复核有批量上限和重试间隔；缺证据或冲突条目会保留。 |
| 修改设置后仍用旧值 | 检查启动进程中的环境变量是否覆盖配置文件。 |
| 向量检索不可用 | 同时检查 Embedding 服务与 DuckDB VSS，安装包不意味着扩展就绪。 |

## 开发与验收

```text
paperbrain/     应用核心、网页服务、解析、模型接口与记忆复核
tests/          单元与集成测试
tools/          批处理、评测、标定与发布门禁工具
demo/           合成测试材料，不是真实研究证据
AGENTS.md       开发约束与模块契约
```

本地检查命令：

```bash
PAPERBRAIN_TEST=1 PAPERBRAIN_CLOUD_ALLOWED=0 python3 -m unittest discover -s tests -v
python3 tools/release_gate.py
```

初次公开提交 `4a77ba4` 在维护者本机通过了 294 项单元测试。这不是持续集成徽章，也不代表所有平台、外部模型或 Docker 环境通过验证。

生产验收还需要标注数据、引文标定、资源测试和人工评审等材料。缺少材料时门禁失败是预期行为，不应降低阈值或跳过检查来宣称完成。当前开发约束与模块契约见 [AGENTS.md](AGENTS.md)；其中的指标是验收目标，不代表已达成的性能承诺。

贡献时请提供最小复现、脱敏日志、Python/系统版本和测试结果。不要提交真实论文全集、API Key、记忆数据库或用户配置。部分辅助脚本仍有维护者环境假设，例如 `tools/embed_tunnel.sh`；不要未经检查直接运行，也不要将其当作通用安装步骤。

## 安全与隐私

- 服务没有用户账户认证，仅用于可信本机环境。不要直接暴露到局域网或公网；公开代码不等于适合公开部署。
- 授权模型后，相关论文文本或配图会发送到配置的服务；开启联网复核后，主张文本会用于 Crossref 查询。请先确认材料允许外传。
- 不要将 API Key 写入 README、截图、公开日志或 Issue。`.gitignore` 不会阻止已跟踪文件里的密钥被提交。
- 论文的版权和保密义务不会因项目采用 MIT 许可证而改变。
- 合成示例的名称、指标、结论和错误公式仅用于测试，不能作为科学事实引用。

## 许可证

本项目采用 [MIT License](LICENSE)：允许商业使用、修改、分发及闭源集成，但分发软件副本或重要部分时须保留版权声明和许可声明。软件按原样提供，不提供担保。

许可证文本参考 [Open Source Initiative 的 MIT 许可证](https://opensource.org/license/mit)。第三方依赖、外部模型服务和用户导入的论文分别遵循自身许可或服务条款。
