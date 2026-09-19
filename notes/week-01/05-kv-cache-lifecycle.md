# KV Cache 生命周期

## 核心对象

可以先把 Serving Engine 中的状态分成三层：

```text
batch row
  → request slot / req_pool_idx
  → 每个 token position 对应的 KV pool slot
  → KV pool slot 中每一层的 K/V tensor
```

Batch compaction 只改变第一层的活跃行排列。只要请求到 token slot 的映射仍正确，就不必因为 batch row 改变而搬动整块 KV Cache。

## 1. 请求进入 waiting queue

此时请求通常只有 prompt token、sampling 参数和业务 metadata，还没有占用完整的生成 KV。Scheduler 会检查 token budget、可用 KV slots、prefix cache 命中和 admission 条件。

## 2. Prefill admission 与分配

Scheduler 为请求分配 request slot，并为需要计算的 prompt token 分配 KV token slots。如果 Prefix/Radix Cache 命中一部分前缀，只需为未命中的后缀计算和分配新的 KV。

```text
prompt = [A,B,C,D]
prefix hit = [A,B]
Prefill/Extend only = [C,D]
```

模型逐层计算新 token 的 K/V，并把它们写入 KV pool；请求映射记录每个逻辑 token position 实际落在哪个物理 slot。

## 3. Prefill 完成

请求已经拥有完整上下文所需的 K/V，可以在安全调度边界 merge 到 `running_batch`。Prefill 最后一个位置的 logits 可以采样第一个输出 token，但该新输出 token 的 K/V 要在它作为下一次 Decode 输入时才写入 Cache。

## 4. 每个 Decode step 追加一个位置

对普通自回归 Decode，每个活跃请求每步大致执行：

```text
为新位置申请 1 个 KV token slot
→ 用最新 token 运行各层
→ 读取所有历史 KV
→ 写入最新 token 在每一层的 K/V
→ sequence length + 1
→ 产生下一 token 的 logits
```

因此 KV 占用随活跃序列长度近似线性增长。Continuous batching 中不同请求长度不同，各自通过映射找到自己的 KV slots。

## 5. Batch compaction

请求完成后，Scheduler 可以把 batch rows 从：

```text
[A, B, C] → [A, C]
```

压紧为两行。A、C 的 KV 不必跟着复制；新的 batch row 仍通过 `req_pool_indices` 找到各自 request slot，再找到历史 token 的 KV slots。

## 6. Finish

请求正常完成时：

- 输出结果被发送；
- request slot 被释放；
- 不再需要的 KV token slots 回到 allocator；
- 若启用 Prefix/Radix Cache，可复用前缀可能由 cache 保留，而不是立即全部释放。

因此“请求完成”和“每一个 KV byte 立即消失”不一定完全等价。

## 7. Abort

Abort 首先要安全地标记请求，避免 GPU 返回行和 request row 错位；在安全边界过滤该请求后，再释放 request 与 KV 资源。客户端断开、超时或内部错误可能触发这条路径。

## 8. Retract

KV 空间不足时，Scheduler 可以选择某个正在 Decode 的请求：

```text
running_batch 移除
→ 释放 GPU KV
→ 重置生成调度状态
→ 重新排队
```

普通路径恢复时要通过 Prefill/Extend 重建 KV。若 Prefix Cache 命中，只重算未命中后缀；Decode disaggregation 还可能把 KV 备份到 CPU，恢复时直接加载，避免重新计算。

## 生命周期总图

```text
waiting
  → admission / allocate request and token slots
  → prefill writes prompt KV
  → merge into running batch
  → decode allocates and appends one KV position per step
  ├─ finish → release or retain reusable prefix
  ├─ abort  → safe filtering → release
  └─ retract → release/offload → requeue → rebuild/restore
```

## 检查题

1. `running_batch` 中 C 从 row 2 compact 到 row 1 时，为什么通常不需要复制 C 的全部 KV？
2. 一个请求完成一次普通 Decode step 后，逻辑上新增的是几个 token position 的 KV？这一个 position 涉及多少层？
3. 启用 Prefix/Radix Cache 后，请求 finish 为什么不一定立即释放它的所有 KV？
