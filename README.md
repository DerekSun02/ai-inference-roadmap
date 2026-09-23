# AI Inference Engineer Learning and Benchmark Roadmap

这是 16 周 AI Inference Engineer 学习、源码阅读、Benchmark 和开源复盘的统一仓库。

路线依据：`AI_Inference_Engineer_16_Week_Roadmap.docx`。默认节奏为每周 12–15 小时；如果每周约 8 小时，则保持相同交付标准并延长到 22–24 周。

## 当前状态

- 阶段：第一阶段——建立共同语言
- 当前周：Week 1——请求到输出的完整链路
- 当前主题：收口 Week 1 的环境、显存计算、baseline 和 PR 复盘
- 理论进度：Week 1 最低要求已覆盖，并提前涉及 Week 2 的 GQA/KV 与 Week 9 的 TP/PP/DP
- 尚未完成：实际模型权重与 KV 显存验证、可复现 baseline、环境记录、两个既有 PR 复盘、周报记分卡

## 每周闭环

1. 用自己的话解释概念和代价。
2. 在 SGLang Omni 或 SGLang 中定位真实代码。
3. 设计只改变一个变量的实验。
4. 保存环境、命令、原始数据和解释。
5. 将结果沉淀为报告、PR 或求职证据。

## 目录

- `roadmap/`：16 周计划和阶段验收标准
- `notes/`：概念、源码调用链、计算题和课堂复盘
- `benchmarks/raw/`：原始实验输出，不手工修改
- `benchmarks/processed/`：清洗数据、汇总表和图表
- `modal/`：云端 GPU 实验的可复现环境与运行手册
- `deploy/`：容器、Kubernetes 与服务配置
- `dashboards/`：Prometheus 和 Grafana 资产
- `reports/`：周报、阶段报告和作品集材料
- `pr-notes/`：Issue、PR 方案和 review 复盘

## 工作在制品限制

任何时刻只保留：一个主动学习模块、一个主动实验、一个主动 PR。等待 review 的 PR 不计入主动 PR。

## 每节课的自动沉淀规则

每次理论学习或源码精读结束时，默认执行以下动作，不需要再次提醒：

1. 提炼本节的核心心智模型、关键状态变化和容易混淆的点。
2. 将内容补充到对应的 `notes/week-XX/` 文件，避免把聊天逐字复制进仓库。
3. 把已完成内容、遗留问题和下一学习点更新到本周进度文件。
4. 如果产生实验结论，同时更新实验记录和周报；原始数据只放在 `benchmarks/raw/`。
5. 检查 Markdown 和 Git diff，创建一个主题明确的小提交。

仓库是长期复习的事实来源；聊天用于互动讲解，笔记用于结构化保留结论。

## Benchmark Definition of Done

- 环境、commit 和模型版本可追踪
- 启动、压测和数据处理命令可复制
- 保存原始数据
- 明确输入输出长度、并发或 request rate
- 至少一次 warmup 和三次正式测量
- 结论区分事实、解释与猜测
- 写明局限和下一步最小验证实验
