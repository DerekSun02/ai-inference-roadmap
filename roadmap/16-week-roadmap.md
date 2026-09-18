# 16 周执行路线

## 总目标

把理论、源码、实验和开源贡献串成同一条证据链：学习概念，在 SGLang Omni 或 SGLang 中找到实现，通过实验建立数据，再沉淀为 PR、报告或作品集。

| 周 | 理论主题 | 项目与开源动作 | 验收 |
|---:|---|---|---|
| 1 | 请求到 token 或 audio chunk 的全链路 | 追踪 Omni API、Pipeline、Scheduler、Executor；建立 baseline 骨架 | 能画出数据流、进程、队列、Stage 和 GPU 边界 |
| 2 | Transformer、自回归推理、Prefill、Decode、KV cache | 定位执行入口和 KV 生命周期；计算权重与 KV 显存 | 解释 Prefill 偏计算、Decode 常偏带宽 |
| 3 | Batching、Scheduler 与 Cache | 比较 concurrency/request rate 对 TTFT、ITL 的影响 | 第一份可信 benchmark 报告 |
| 4 | Linux 并发与服务基础 | 进程、线程、asyncio、shared memory、relay；基础 metrics/replay | 第一阶段复盘和知识缺口表 |
| 5 | Paged KV 与 Prefix Cache | 对照 Radix Cache 与 PagedAttention；测试共享前缀 | 解释 cache policy 与 workload 的关系 |
| 6 | GPU 执行与性能模型 | PyTorch Profiler 或 Nsight Systems 首份 trace | 区分 CPU、kernel、memory、communication gap |
| 7 | 量化与数值精度 | 比较 BF16、FP8 或 INT8 的显存、吞吐和质量 | accuracy/performance trade-off 表 |
| 8 | Benchmark 方法论 | 固定 warmup、数据集、并发、GPU 状态并重复实验 | Benchmark Lab 可公开版本 |
| 9 | 模型并行与 Collective | TP/PP/DP/EP 与 NCCL；单卡和双卡比较 | 解释扩展效率不足 |
| 10 | 多阶段 Omni Pipeline | stage placement、relay、backpressure、streaming | stage latency、queue depth、TTFA、RTF 分解 |
| 11 | Kubernetes 与弹性 | 容器化、probe、rollout、队列指标伸缩 | Production Serving 骨架 |
| 12 | 可靠性与事件处理 | OOM、worker crash、slow request 故障注入 | runbook 与 postmortem |
| 13 | 端到端瓶颈定位 | 建立 hypothesis → profile → change → validate 循环 | 选定唯一优化目标 |
| 14 | 源码级优化 | scheduler/cache/observability/kernel 路径改进 | 可复现的性能或可靠性收益 |
| 15 | 系统设计与成本 | 容量模型、SLO、cost per token、trade-off | 架构图和容量模型 |
| 16 | 包装与投递 | 清理仓库、重跑 benchmark、整理 PR 和面试故事 | 项目页、简历 bullet 和演示 |

## Week 1 必须完成

- [x] 开始追踪请求从 Client 到音频输出的链路
- [x] 区分 Pipeline 与 Scheduler 的职责
- [x] 理解 Prefill 与 Decode 的基本边界
- [x] 理解 continuous batching 中请求如何进入和离开 `running_batch`
- [ ] 理解 KV cache 为什么节省重复计算
- [ ] 手算一次模型权重和 KV cache 显存量级
- [ ] 固定服务、模型、commit 和运行环境
- [ ] 单请求、固定并发、固定 request rate 三组 baseline
- [ ] 两个既有 PR 各完成一次复盘
- [ ] 完成 Week 1 周报和记分卡

## 进度原则

当前 Scheduler 和 continuous batching 的源码学习，是为了完成 Week 1 的“真实请求调用链”。它不等于已经进入 Week 3 的系统化 batching 实验。Week 3 会重新回到该主题，重点转向定量实验、策略比较、TTFT/ITL/吞吐和公平性。
