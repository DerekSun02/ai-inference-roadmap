# Week 1 学习进度

## 已完成的课堂主题

1. Qwen3 TTS 请求链路和首段可播放音频的前置步骤。
2. Pipeline 与 Scheduler 的职责边界。
3. `prefill_coalesce_requests` 的配置传播路径和等待条件。
4. concurrency 与 Prefill batch size 的区别。
5. Prefill coalescing 对 QPS、TTFC 和 Decode batch 的影响。
6. `running_batch` 的动态变化、请求完成、加入与 batch compaction。
7. 完成 finished、retract、Prefill merge 的状态推演练习。
8. 理解普通 retract 需要重建 KV，以及 Decode disaggregation 可通过 host backup 恢复 KV 的例外。
9. 理解没有 KV Cache 时为何每个 Decode step 会重新计算历史 token 的逐层 K、V。

## 当前掌握程度

- 能说明 `concurrency=16` 表示系统中最多或目标存在多个并发请求，不代表每一轮恰好出现 16 个新 Prefill 请求。
- 能说明 coalescing 通过短暂等待增加 Prefill batch 的机会，因此可能提高吞吐但增加 TTFC。
- 能说明新请求先形成独立 Prefill batch，完成后才在安全边界 merge 到 `running_batch`。
- 能说明完成请求的 batch row 必须和 `seq_lens`、`req_pool_indices` 等逐行字段一起 compact。
- 能推演请求完成、KV 压力 retract 和新 Prefill 请求 merge 后的 `running_batch` 与 `waiting_queue`。
- 能解释 causal mask 为什么保证旧 token 的 K、V 在追加未来 token 后保持不变。
- 能区分“保留 token 历史”和“保留历史中间 KV 结果”。

## 下一步

1. 理解为什么保存 K、V 而不是保存 Q。
2. 追踪 KV cache 从 Prefill 分配、Decode 追加到完成释放的生命周期。
3. 手算一次 KV cache bytes，并与实际 GPU memory 观察做数量级验证。
4. 把当前源码链路补进 `01-request-lifecycle.md`。
5. 固定实验环境，为 Week 1 baseline 做准备。
