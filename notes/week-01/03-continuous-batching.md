# Continuous Batching 与 running batch

## 核心心智模型

`running_batch` 是当前仍有资格执行下一轮 Decode 的活跃请求集合，不是创建后不变的静态 batch。

```text
running_batch(t+1)
= running_batch(t)
- finished
- aborted
- retracted
+ newly_prefilled
```

## 一轮调度的简化过程

```text
接收请求
  → 处理 waiting queue
  → 合并上一轮完成 Prefill 的请求
  → 过滤 finished/aborted 请求
  → KV 不足时 retract 一部分请求
  → 选择新的 Prefill batch 或 running Decode batch
  → 执行一次 forward
  → 处理结果并更新请求状态
  → 下一轮
```

## 新请求如何加入

```text
waiting_queue
  → admission
  → 独立的 Prefill batch
  → Prefill 完成
  → 下一安全调度边界 merge
  → running_batch
```

请求不能从 waiting queue 直接跳进 Decode，因为它必须先完成 Prefill 并建立可供后续 Decode 使用的上下文和 KV cache。

## 请求完成与安全删除

GPU 的返回结果按执行 batch 的旧行号排列。结果处理会先更新 token，并把 EOS、达到最大长度或 abort 的请求标记为 finished。真正从 batch 中删除通常发生在结果已经安全消费后的调度边界，避免 GPU result row 和 request row 错位。

## Batch compaction

删除请求时必须用同一组 `keep_indices` 压紧所有逐 batch-row 字段。

```text
删除前:
row                0      1      2      3
reqs              req1   req2   req3   req4
seq_lens            20     31     18     42
req_pool_indices     7     11      3      9

req2 finished → keep_indices=[0,2,3]

删除后:
row                0      1      2
reqs              req1   req3   req4
seq_lens            20     18     42
req_pool_indices     7      3      9
```

不能只删除 Python `reqs` 列表，否则 `reqs[1]` 可能已经是 req3，但 `seq_lens[1]` 和 KV 映射仍属于 req2。

## 逻辑 compaction 与 KV 内存

Batch compaction 压紧的是 Scheduler 的活跃行视图，不一定搬动整块 KV tensor。`req_pool_indices` 将新的连续 batch row 映射到稳定、可能不连续的 KV pool slot。

## Retract

当 Decode 所需 KV memory 不足时，Scheduler 可以把部分活跃请求从 `running_batch` 移出、释放或调整资源，并重新放回 waiting queue。Retract 是资源压力下的重新排队，不等同于请求完成。

## 练习

初始：

```text
running_batch = [A, B, C]
waiting_queue = [D]
```

本轮 B 遇到 EOS、C 因 KV 不足被 retract、D 随后完成 Prefill。填写：

1. `filter_batch()` 后：
2. retract 后：
3. D merge 后：
