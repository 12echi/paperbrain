# AGENTS.md — PaperBrain Copilot

> 生产级仓库：唯一验收依据是 `学术文献全通读与论文写作系统_落地执行方案v5.0.txt`（v4.0/v3.0 仅留底）；代码与测试必须落实并阻断未满足的 v5 门禁，不得反过来绕过规范。实现入口：`paperbrain/{ids,verifier,budget,graph,vector_store,preflight,sections,passes,memory,outline,pipeline}.py`。界面：`python3 -m paperbrain.server --port 8000`（`/api/health` 看工具状态，steps ①出大纲②确认生成）。门禁：`python3 -m unittest discover -s tests -v` 全过，最终统一跑 `python3 tools/release_gate.py`；M1/M3/M5 分别可用 `tools/predict_m1.py` / `tools/predict_m3.py` / `tools/predict_m5.py` 生成带源 SHA/案例与策略 SHA 的预测；统一门禁会从 M1 golden 的源路径现场解析，并现场重生成确定性的 M3/M5 预测，不信任任何外部 pred。其余量化工具为 `tools/calibrate.py`、`tools/validate_m6.py` 与 `tools/validate_output_quality.py`。未提供 30 篇 M1 golden、M3/M5 golden、50 对引文标定、50 篇 M6 样本或 M8 双人工评审报告时只能输出未验收/待复核，不能宣称生产完成。

## 项目意图与架构（5 阶段，见方案第二章）

1. `预处理分流`：PyMuPDF 检测文本可提取率 → 标准解析 / OCR 管道；过滤非法公式，切分 `(a)(b)(c)` 复合子图。
2. `分章投喂`：拆为 `[Intro/Abstract] [Methods] [Experiments/Figures]`，并发跑 Pass1-3。
3. `记忆沉淀`：LightRAG + SQLite / DuckDB，只存白名单实体/关系。
4. `大纲生成`：STORM 式三级大纲 → 用户确认 → 按小节分块生成 + 全局过渡句。
5. `事实门禁`：正则提取引文标记 → 比对检索片段 → 不合格标 `⚠️ [待核实引文]`，绝不静默放行。

## 模块衔接契约（数据流，改任一环先看这里）
```
preflight ── PreflightResult(route,sha,text) ─▶ split_sections ── sections[{name,sec,text,confidence,chunks[{chunk_id,text}]}]
   │                                                                        │
   └(pdf) OCR兜底/表格清点/公式门禁/可选VL读图                              ▼
                                                    run_passes(summarize_fn) ── passes{pass1..4, ground_truth{chunk_id→text},
                                                                                 fig_index, name_sec, pass_tokens, coverage}
                                                                                          │
                          build_memory(sections) ── SQLite(chunks/entities/relations) ────┤
                                                                                          ▼
                          build_outline(passes,memory) ── outline{sections[{h1,h2,h3,chunks,claims}],entities}
                                                                                          │
   用户确认/编辑 ─▶ draft_from_outline(generate_fn 或 draft_override) ── draft.md ─▶ consistency.polish
                                                                                          ▼
        CitationVerifierV5(gt, fig_index, scorer, section_chunks) ── verify.json(status/report)
                                                                                          ▼
                          budget.check_budget ── ledger.csv + report.md
```

关键契约（改动易错点）：
- `sections.name` 必须是 `abstract|intro|method|experiments|conclusion|full`；`sec` 是论文原编号（0/1/2/3/4 或 5/6，因文而异）。**outline 按 `name` 归位（`passes.name_sec`），不得硬编码节号**。
- 同名节（如 Experiments 3 + Discussion 4）在 `run_passes` 内**必须按 name 合并**，否则 dict 覆盖丢整节。
- `ground_truth` 键为 `{Paper}_Sec{sec}_C{nnn}`（chunk）与 `{Paper}_{sec}`（节粗键）；verifier 打分优先用 `section_chunks` 逐块取最强。
- 外部模型草稿走 `generate_from_outline(..., draft_override=...)`（`model=muse-session`），跳过内置 RAG。
- 自动过渡只在草稿 `## h1 / h2` 标题序列与已确认 outline 逐项一致时注入，并须跟随草稿主体语言生成；标题缺失、重复或错序必须保持原稿并令 `transition_complete=false`，禁止按位置猜测绑定。
- `deepread.md` 与 `deepread_full.md` 必须分别验真并聚合为失败关闭状态；系统生成笔记只有在对应产物为 `CLEAN` 且主张级复核也通过时才可标 `active`，缺失/旧版验真记录一律进入 `candidate`。
- 任一低置信章节触发全篇 `downgrade` 后只允许 Pass1 与待复核概览；后续即使生成草稿且引文打分很高，也必须由 `[LOW_CONFIDENCE_EXTRACTION]` 阻断 CLEAN。
- `review` 任务必须使用专用审稿生成，区分证据支持的优点、效度威胁与材料无法回答的问题；任一审稿小节未完成专用生成时，以 `[REVIEW_NOT_GENERATED]` 阻断，离线摘录不得冒充同行评审。
- 确认后大纲的 `paper_id` 必须与当前论文一致，所有 ChunkID 必须存在，h1/h2 身份必须唯一；部分有效、部分无效的 ChunkID 也不得静默忽略，分别由结构门禁明确阻断。
- 系统生成实体笔记仅在图谱策略有效且未到复审期、实体名称以词边界真实出现于源文时标 `active`；策略待复核、无源绑定或仅为更长标识符子串时一律 `candidate`。
- 深读全局综合 JSON 必须先按 `argument_map/positioning/field_view` 契约清洗，引用提示须注入真实 paper_id；warning 只是诊断元数据，所有模型阶段无有效正文时必须回退完整离线骨架，禁止把 warning 当成成功内容。
- M8 产出质量须以当前源文件与产物 SHA 绑定的至少20案例双人工独立评审证明，覆盖 full_read≥10、method≥5、review≥5；事实准确、证据可追溯、学习目标、洞察、可操作性、语言一致性六维均分≥4/5，存在阻断问题即失败。可用 `tools/validate_output_quality.py --init-manifest <清单> --output <评审模板>` 计算哈希并生成待填模板；模板默认 `REVIEW_REQUIRED`，不能直接放行。
- SQLite 写入幂等：chunks/entities 用 REPLACE，relations 先按 paper_id 删除再插。

## 硬约束（方案第一章 6 陷阱，不要绕过）

- **解析**：三轨路由 >90%标准 / 50-90% Hybrid / <50%全量OCR；Hybrid 只对文本率<70%的页补 OCR，表格线缺失由 Caption 锚定局部 text-table 解析；LaTeX 须过 `katex` 主校验+`sympy`辅校验，双fail打 `$$[FORMULA_UNVERIFIED]$$`；复合图仅 IoU>0.5 且标签数==块数才拆，否则保留整图+`⚠️ [需人工拆图]`；章节低置信降级只跑 Pass1。
- **Token 预算**：双顶 TEXT≤12k / VISION≤6k / TOTAL≤18k（含system/JSON/重试），见 `budget.check_budget`。Pass1 ≤2.5k；Pass2 ≤3.5k；Pass3 ≤4k；Pass4 只读纪要+引用链 ≤2k，无引用链拒绝执行。缓存命中复跑 ≤2k，键为5元组`(pdf_sha,section_hash,prompt_version,model_version,embedding_version)`。
- **图谱 Schema**：只允许 6 实体 `Algorithm/Model, Dataset/Benchmark, Evaluation Metric, Theoretical Component, Problem/Task, Limitation/Artifact` + 5 关系 `Improves_Upon, Evaluated_On, Contradicts, Vulnerable_To, Requires`。入库前强制归一+别名合并(cos>0.93)+黑名单，泛词率<1%，Schema外0容忍。
- **引文格式**：统一 `[Ref: <PaperID>, Sec <X.Y>]` / Fig / Tab（如 `[Ref: 2024_NeurIPS_01, Sec 3.2]`）。必须跑 `CitationVerifierV5.verify_draft()`：存在性+Fig/Tab精确+语义蕴含三过才PASS（embedding链用0.82，离线词法链用`thresholds.json`标定值）；单引用多图直接UNVERIFIED；全/半角括号兼容；无scorer/低分段报`NEEDS_REVIEW`不冒充PASS；零引用报 `NO_CITATION` 不冒充CLEAN。
- **长文生成**：禁止一次性数千字直出。先出三级大纲+每节ChunkID清单（版本化可回滚）→ 等用户确认 → 按小节生成，全局一致性Agent做术语/去重/过渡/矛盾检查。
- **Mac 本地资源**：统一内存 16/24GB上限，本地常驻 ≤3GB、整机峰值≤8GB，Docker加 `--memory=3g --cpus=2` 懒启动；多模态走云端VL需显式授权开关；存储只用SQLite/DuckDB；禁默认装7B-VL/MinerU全量。

## 验收阈值（方案第八章 v5.0 门禁）

- 两栏+公式 PDF 还原率 ≥98%，表格 ≥99%，图表必须绑定 Caption+正文上下文段（30篇golden集）；矛盾检测召回≥90%且负例误报率≤10%。
- 单篇 Token ≤12k；泛词率<1%，别名合并≥97%；引文三阶段全过；本地常驻 ≤3GB、50篇P95无崩。

## 开发约定（生产级：`pyproject.toml` + stdlib-only）

- Python ≥3.9，核心零重依赖：`python3 -m unittest discover -s tests -v` 必须全过才算完成。PDF 解析是可选依赖（`pip install pymupdf`，`preflight` 懒导入，缺失时 PDF 降级只跑 Pass1）；公式/图表/OCR 工具链（node+katex、sympy、opencv、tesseract）缺失时对应模块降级标记、不硬扛；测试本身保持 stdlib-only。
- 真 LLM 精读需环境变量 `PAPERBRAIN_API_KEY`（+可选`PAPERBRAIN_BASE_URL/PAPERBRAIN_MODEL`）以及独立显式授权 `PAPERBRAIN_CLOUD_ALLOWED=1`；任一缺失均回退离线规则式并在报告如实记录。也可走界面模型接入卡（配置文件 600 权限，可测试/清除）；一键复用本机 opencode 时走 `opencode-go`，Key 现读 auth.json 不复制。VL 还需 `PAPERBRAIN_VISION=1`，上传前强制重编码 PNG 并剥离 EXIF/XMP/文本元数据，单篇最多5图、用后删除 scratch。`tools/calibrate.py --pairs <50对以上标注集>` 才能生成可放行阈值；仓库内 demo 阈值仅诊断，改打分器必须重标定。
- 会话模型（Muse）为默认主力：`run_outline` → `export_prompts` → 会话中撰写 → `generate_from_outline(..., draft_override=...)` 验真（报告记 `model=muse-session`）；UI 走③④按钮；`import_answers` 只收草稿不收无源断言。
- 全模型模式（零本地重模型）：`PAPERBRAIN_ALL_MODEL=1` 后，分章/图谱抽取/NLI蕴含/纪要/草稿全由已接入模型（opencode-go CLI 或 OpenAI兼容API）完成，见 `paperbrain/llm_ops.py`；分章/图谱失败可规则回退并记录，NLI 失败必须 `NEEDS_REVIEW`，禁止换评分器后静默放行。**本机实测 8GB RAM / 43GB 可用磁盘，不装 MinerU/torch，全部语义活走模型调用**（会话复用后约8~10s/次）。
- 向量层（`embeddings.py` + `vector_store.py`）：外部 OpenAI 兼容 `/embeddings`；float32 BLOB 仅作可重建缓存，真实召回必须走 DuckDB VSS/HNSW，后端不可用时退 BM25，禁止 Python 全表余弦。内容哈希幂等，`finalize_memory` 自动 `embed_notes`；端点不可用自动降级纯 BM25（熔断：连接/超时/5xx 后 120s 跳过，4xx 不熔断）。
- 新增重型本地模型（如 7B-VL、MinerU 全量）前先说明内存与成本影响；向量检索必须显式声明 `sqlite-vss` / DuckDB VSS 扩展，不默认可用。
- 云端调用单独隔离、可 mock（`semantic_scorer` 可注入）；`paperbrain/` 按 ids/verifier/budget/graph/retrieval/llm/text/formulas/figures/ocr 分文件，不混写；批量处理 `paperbrain/batch.py`（`run_batch` 串行+续跑+逐篇 `finalize_memory`，批量期**抑制领域反思**、收尾统一 `reflect_global()` 一次，UI「批量解读」/`POST /api/batch`/CLI `tools/batch.py` 三入口共用；输出 `out/web/batch_ledger.csv`）；性能基准 `python3 tools/bench.py [rounds]`（离线，改动记忆/检索/验真后跑它对比，勿凭感觉优化）；检索评测双集（冻结快照 `out/eval_snapshot.sqlite`，改检索/权重必跑）：①关键词精查 `python3 tools/eval_retrieval.py --db out/eval_snapshot.sqlite`；②模糊语义 `python3 tools/eval_retrieval.py --db out/eval_snapshot.sqlite --questions out/eval_questions.json [--langs zh|en]`（问题集由 `tools/gen_eval_questions.py --n 20 --en` 一次性生成并缓存；评测不依赖模型/网络）。
- 切分：裸标题（Methods/Results/Discussion）+ 卷首语捕获 + 文末元数据截断（`split_sections(..., meta)` 看 `dropped_tail`）；公式先 `check_formulas` 再 `mark_text` 入库；引用必须命中 gt key，无源小节跳过并打标不编造。
- 界面（`server.py` PAGE）四部分：①模型设置 ②文件导入（拖拽/文件选择/本机绝对路径，**无粘贴框**；路径限 用户目录/tmp/Volumes 白名单）③解读类别（含**深度三档** fast/standard/deep）④学习记忆（问记忆/生成反思/领域反思/待裁决/偏好/知识网络/领域地图）；类别注册在 `tasks.py`（full_read/outline/method/figures/review/memory），调度 `run_task(depth=...)`（standard=纪要+深读；deep=再加草稿+验真）；记忆库见下；PDF/txt 文本统一过 `text.sanitize_text` 防乱码。
- 深度精读（A+B+C，`deepread.py`）：穿透式深读（一句话主张/Gap/洞见/方法/证据强度/隐藏假设/局限/可复现/延伸设想）+ 多视角拷问（`llm.py` 的 `gen_questions`/`answer_questions`）+ 全局综合（argument_map/positioning/field_view）；`full_read` 产出 `deepread.md` 为成果正文。产物 `deepread_synthesis.json` 便于机器读。
- 记忆网络 v2（`memory_store.py`，五层）：L1 笔记 `notes`(+anchor/status/confidence/importance/last_access/invalid_at)；L2 `concepts`+`aliases`（模型写时消解）；L3 类型化 `note_links`(supports/contradicts/extends) + `paper_edges`；L4 检索 = **自适应融合 × 三因子 × 图多跳**（FTS5/BM25 + 向量 RRF；`_lexical_confidence`/`_lex_trust` 分段：conf>0.6 信词面不用向量、三因子/画像满权；conf≤0.6 向量权重升至 1.0、候选池收缩到 12、重要性/近因/偏好衰减为 0。**评测（修正幽灵正例后）**：模糊 Recall 0.183→0.309（+69%）、MRR 0.373→0.586、nDCG 0.237→0.413，与最优纯 RRF 持平；关键词 0.498/0.950/0.717，较裸 BM25 0.510/0.988/0.741 略低 2-4%（重要性/近因平局裁决的代价，口径见 `out/eval_*.json`）；终分=相关0.80(min-max)+信任×(重要性0.12+近因0.08)，半衰期7天、访问强化 `touch_notes`；`graph_expand` PPR 懒惰随机游走 2 跳（模糊 MRR +0.2%）；`invalidate_note` 双时间失效留档不检索；删除/重建会同步清理 note_links/vectors/悬空 links（反向链封顶 24），存量可用 `gc_memory()` 回收；实体/rejected/superseded/已失效自动过滤）；L5 `ask_memory(q)` = 查询扩展→检索→模型重排→MMR→一跳图→带 [M#] 合成（离线回退摘要）、`reflect()`/`reflect_global()` **反思层**（Generative Agents：综合洞见写回 note + supports 证据链，`_verify_claims` 质检门不过或无证据一律降 candidate；跨论文领域反思默认每新增 3 篇自动触发+meta 防重，`PAPERBRAIN_REFLECT=0` 可关）、`consolidate()`、`memory_stats()`；写时质检门 `_verify_claims`（模型批量 NLI + 词法兜底，响应不完整不做 1-based 偏移推断）→ 不足者 `candidate`；矛盾主张标双方 `contested`；`pending_decisions/decide_note` 待裁决闭环。**改检索权重必须重跑评测**（见下）。
- 全局层（`config.py` / `context.py` / `memory_store` 跨论文）：所有环境变量/路径/限额统一走 `config.py`（运行时读取；`PAPERBRAIN_*` 缺失时兜底读 `~/.config/paperbrain/env.json`，含布尔开关，环境变量优先，文件删除/`modelconf.clear` 后缓存失效，`PAPERBRAIN_TEST=1`/unittest 进程禁用兜底）；模型喂料统一走 `context.build_paper_context`（按节配额 + 关注点相关，替代各处 `[:8000]` 硬截断）；`build_concepts`/`link_papers`/`field_map` 从笔记提炼概念（过泛词黑名单）并建立跨论文边与连通簇，`/api/field` 输出领域地图；收尾统一 `finalize_memory(out_dir,pid,task)`（产物为空时**保留旧笔记**，不误删）。
