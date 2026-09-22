# Week 1 周报

## 本周一句话结论

已建立 Qwen3 TTS 请求从预处理、Scheduler、Prefill/Decode 到 Vocoder 的心智模型，能解释 continuous batching、KV Cache 地址映射、Radix prefix 复用和基础并行分片；实验基线与实际显存验证尚待完成。

## 已有交付物

- `notes/week-01/01-request-lifecycle.md`
- `notes/week-01/02-prefill-coalescing.md`
- `notes/week-01/03-continuous-batching.md`
- `notes/week-01/04-kv-cache.md`
- `notes/week-01/05-kv-cache-lifecycle.md`
- `notes/week-01/06-sglang-kv-addressing.md`
- `notes/week-01/07-parallel-overlap-request-slots-radix.md`
- `notes/week-01/08-kv-head-and-parallelism-numeric-example.md`

## 待完成交付物

- 完整源码调用链
- 实际模型权重与 KV cache 显存计算及 GPU 数量级验证
- Baseline 实验与原始数据
- 两个既有 PR 复盘

## 周度记分卡

| 维度 | 当前证据 | 得分 |
|---|---|---:|
| 理论 | Pipeline、Scheduler、Prefill/Decode、continuous batching 笔记 | 待自评/2 |
| 代码阅读 | 已定位部分 Scheduler 路径，调用链待补全 | 待自评/2 |
| 实验 | 尚未完成 baseline | 0/2 |
| 开源 | 既有 PR 尚未在本仓库复盘 | 待自评/2 |
| 表达 | 已形成三份结构化学习笔记 | 待自评/2 |
| 总分 | | 待填写/10 |
