# 数据目录索引

本目录是原始工作目录的完整副本，包括标签、LLM 请求与回复记录、运行日志和中间缓存。其中的代码与原始项目的目录层次一致，所以不能直接在这里运行（见根目录 README 的"复现"一节）。

## data/H2：合成数据与真实数据的生成

H2 生成合成数据（POOL 5000、TEST 1000，模板不重叠）和真实数据（DFT 与 CHGNet 标签，TRAIN 与 HOLDOUT 各 2000）。

| 文件或目录 | 内容 |
|---|---|
| `synthetic_records_private.json.gz` | 合成数据的标签（POOL 与 TEST），用 CHGNet 弛豫后计算的形成能 |
| `real_records_private.json.gz` | 真实数据的标签（DFT 与 CHGNet），TRAIN 与 HOLDOUT |
| `split_manifest.json` | 训练、验证、测试的划分清单 |
| `frozen_plan.json.gz` | 冻结的生成计划 |
| `formal_attempts.jsonl`、`completion_attempts.jsonl` | 生成与补全的全部尝试记录 |
| `worker_recovery_events.jsonl` | 运行中工作进程的恢复事件 |
| `benchmark_*`、`gpu_memory_*`、`serial_repeat_*`、`calibration_*` | 资源基准与重复性检查的结果 |
| `*.log`、`*_stdout.log`、`*_stderr.log` | 各次运行的日志 |
| `protocol_revision1_prior/`、`protocol_revision2_prior/`、`resource_revision0/` | 正式运行前的旧版本协议和资源修订，包含 `h2_*.py`、`p1_*.py` 等脚本与其输出 |
| `cache/` | matplotlib 的字体缓存 |

## data/H3：记忆效应与零样本特征

H3 的结果说明在 `docs/H3_results_README.md`。H3 的原始数据主要在 `data/H2` 和 H4 目录中。

## data/H4：LLM 特征循环与对照

| 文件或目录 | 内容 |
|---|---|
| `phase4_1/` | 阶段 4.1：一次性 LLM 特征（O）。包括候选的特征代码、最终模型、终局评分标记和结果 |
| `phase4_2/` | 阶段 4.2：循环 LLM 特征（L0）。包括每轮的请求、回复、选择结果、报告 |
| `phase4_3/` | 阶段 4.3：循环加记忆（L1）。包括作废的首轮调用（`voided_original_round1_20261031/`）、修订后的运行（`memory_20000_revision_20261031/`、`manual_terminal_resume_20261031/`）、终局评分与报告 |
| `phase4_4/` | 阶段 4.4：对照与迁移。包括 CGCNN 复用审计（`audit/cgcnn_reuse_audit.json`）、合成与真实的终局结果、报告 |
| `src/`、`tests/` | H4 的代码与测试，原始工作目录中的版本 |
| `protocol/` | 模型目录、特征结构、LLM 指令文本 |
| `transport/`、`software_transport/`、`software_stats/`、`software_l0_stats/`、`software_l0_driver/` | LLM 调用的传输记录与软件测试记录 |
| `frozen_config.json`、`notification_state.json`、`monitor_automation.json`、`process_identity.json`、`transport_derivative_audit.json` | 运行状态与审计文件 |

每个阶段的 `report/` 目录下有完整报告、曲线图和特征源码清单。

## data/H4_feature_loop：H4 的公开目录

这是 H4 对外公开的目录，包括预测结果（`predictions.csv`，含 L1 追加的 3000 行）、指标（`metrics.json`）、运行日志（`run.log`）、配置（`config.json`）、LLM 请求与回复记录（`llm_calls.jsonl`）和状态说明（`README.md`）。

这个目录的文件哈希与早期受保护清单记录的不同，原因和处理方式见 `docs/PROTOCOL_REVISIONS.md` 的第 5 条。
