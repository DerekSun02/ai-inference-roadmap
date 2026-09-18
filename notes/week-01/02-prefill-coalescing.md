# Prefill Coalescing 学习笔记

## 参数传播

- 参数：`prefill_coalesce_requests`
- 默认值：`0`，即默认不开启
- factory：`create_sglang_tts_engine_executor`
- builder：`Qwen3TtsEngineBuilder`
- Scheduler 传播：通过 `extra_scheduler_kwargs` 进入 generation engine builder，再在 scheduler 构造时传给 `OmniScheduler`

## 放行条件

在启用相应开关且允许 coalescing 的情况下，只要满足以下任一条件就不再等待：

- `waiting_queue` 数量达到 `prefill_coalesce_requests`
- 最老请求等待时间达到 `prefill_coalesce_wait_s`

否则返回空的 `NextBatchPlan`，保留现有 `running_batch`，暂不构造新的 Prefill batch。

## 为什么 concurrency 不等于 Prefill batch size

`concurrency=16` 描述系统同时存在或允许推进的请求规模。请求可能分散在：

- waiting queue
- 正在 Prefill
- 正在 Decode
- 等待下游 Stage 或输出

Prefill batch size 只统计同一调度时刻被选入 Prefill 的新请求，因此通常小于并发数，并会受请求到达时间、队列策略、token budget 和 KV 资源影响。

## 吞吐与延迟权衡

短暂等待可以把多个小 Prefill 合成较大的 Prefill batch，摊薄固定调度和 kernel launch 成本，可能提高 QPS。但它会增加 admission wait，因此 TTFC/TTFT 可能上升。

该策略应保持 opt-in，因为不同 workload 下未必产生净收益：等待 Prefill 请求可能推迟 admission，Prefill 抢占 step 也可能影响 Decode batch 的规模和连续性，最终可能只有延迟增加而没有吞吐改善。

## 时间线示例

```text
K = 2, T = 30 ms

0 ms:
  waiting_queue = [req3]
  running_batch = [req1, req2]
  → req3 未达到数量或超时条件，本轮继续 Decode

20 ms:
  waiting_queue = [req3, req4]
  running_batch = [req1, req2]
  → 达到 K，构造 new_prefill_batch=[req3, req4]
  → 本轮运行 Prefill；req3/req4 尚未进入 running_batch

下一安全调度边界:
  running_batch.merge_batch(new_prefill_batch)
  → [req1, req2, req3, req4]
```
