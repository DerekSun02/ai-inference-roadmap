# Week 1 学习进度

## 已完成的课堂主题

1. Qwen3 TTS 请求链路和首段可播放音频的前置步骤。
2. Pipeline 与 Scheduler 的职责边界。
3. `prefill_coalesce_requests` 的配置传播路径和等待条件。
4. concurrency 与 Prefill batch size 的区别。
5. Prefill coalescing 对 QPS、TTFC 和 Decode batch 的影响。
6. `running_batch` 的动态变化、请求完成、加入与 batch compaction。

## 当前掌握程度

- 能说明 `concurrency=16` 表示系统中最多或目标存在多个并发请求，不代表每一轮恰好出现 16 个新 Prefill 请求。
- 能说明 coalescing 通过短暂等待增加 Prefill batch 的机会，因此可能提高吞吐但增加 TTFC。
- 能说明新请求先形成独立 Prefill batch，完成后才在安全边界 merge 到 `running_batch`。
- 能说明完成请求的 batch row 必须和 `seq_lens`、`req_pool_indices` 等逐行字段一起 compact。

## 下一步

1. 完成 continuous batching 状态练习。
2. 学习 KV cache 的内容、生命周期和显存公式。
3. 把当前源码链路补进 `01-request-lifecycle.md`。
4. 固定实验环境，为 Week 1 baseline 做准备。
