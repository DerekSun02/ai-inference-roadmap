# TP / DP / PP、Async Decode、Request Slot 与 Radix Cache

本文基于 `sglang==0.5.19` 和当前 `sglang-omni` checkout，连接四个概念：模型如何分布到 GPU、Omni 如何一步前瞻、请求如何获得地址表行、Radix Cache 如何把 token prefix 映射回离散 KV slots。

## 1. Rank、Worker 和并行组

- **GPU**：物理计算设备。
- **process / worker**：通常是控制一张 GPU 的进程。
- **rank**：一个分布式进程在通信组中的编号，例如 TP rank 0、TP rank 1。
- **local**：当前 rank 真正持有和计算的那部分参数、层、heads 或请求。

## 2. TP、DP、PP

### Tensor Parallelism（TP，张量并行）

把同一层的权重矩阵或 attention heads 横向拆到多张 GPU。一个请求的同一个 token 同时经过所有 TP ranks；中间结果通常需要 all-reduce / all-gather 等通信才能组成完整层输出。

例如模型有 8 个 KV heads，`TP=2`：

```text
TP rank 0: 本层的 KV heads 0..3
TP rank 1: 本层的 KV heads 4..7
```

所以每个 rank 上的 KV Cache shape 使用 `num_kv_heads_local=4`。TP 主要解决单张 GPU 放不下一个模型或希望用多 GPU 加速一次 forward，但每个请求会占用整个 TP group。

### Data Parallelism（DP，数据并行）

每个 DP replica 持有一份完整模型，或者持有一整个相同的 TP group。不同请求被路由到不同 replica，replica 之间通常不需要在每个 inference layer 上通信。

```text
DP replica 0: 请求 A、C
DP replica 1: 请求 B、D
```

DP 主要扩大总吞吐和服务容量；代价是模型权重被复制多份。SGLang 的 DP attention 等高级模式会让某些 attention / MoE 组件的行为更特殊，但入门阶段先使用“多个相同服务副本处理不同请求”的模型。

### Pipeline Parallelism（PP，流水线并行）

把模型的连续层纵向拆到多张 GPU：

```text
PP rank 0: layers 0..15
PP rank 1: layers 16..31
```

一个 token 先经过 rank 0，再把 hidden states 发给 rank 1。每个 PP rank 只保存自己负责层的权重和 KV Cache，因此 `num_layers_local` 是当前 stage 的层数，不是全模型层数。PP 可以容纳更深的模型，但有跨 stage 激活通信和 pipeline bubble。

### 三者最短区分

```text
TP：拆一层；同一个 token 同时使用多卡。
PP：拆层段；同一个 token 按顺序经过多卡。
DP：复制模型/模型组；不同请求去不同副本。
```

可以组合成 `DP × PP × TP`。例如 8 张 GPU 使用 `DP=2, PP=2, TP=2`：2 个服务副本；每个副本有 2 个 pipeline stages；每个 stage 内再用 2-way TP。

## 3. SGLang-Omni 的一步前瞻 Async Decode

### 同步循环

```text
launch step N GPU
wait GPU
CPU resolve/collect N
更新请求状态
schedule and launch N+1
```

GPU 完成后，CPU 的 Python collect 期间 GPU 可能空闲。

### 一步前瞻循环

当前 Omni 的 `_event_loop_async_decode` 顺序是：

```text
launch(N) → resolve(N-1)
```

`execute_launch` 将 forward、GPU sampling、结果发布和 CUDA event 排入 GPU stream，但不等待。采样出的 token 通过 `FutureMap` 按 request slot 保留在 GPU 侧，下一步 forward 可以直接取得，不需要先等待 Python 把 token 搬回并更新对象。

随后 CPU resolve 上一步：等待其 event（如果还没完成）、读取 staging/snapshot、逐请求 collect、发 stream output、更新 finish state。

### 为什么会 overshoot 一步

假设 step S 让 C 生成 EOS，但 CPU 尚未 resolve S，因此 Scheduler 还不知道 C 已完成。在 resolve S 之前，step S+1 已经 launch，旧 batch snapshot 仍包含 C：

```text
step S snapshot:   [A, C]  -- GPU 产生 C 的 EOS
step S+1 snapshot: [A, C]  -- 已提前 launch，多算 C 一步
CPU resolve S:     此时才把 C 标记 finished
CPU resolve S+1:   识别 C 是 prior-step finished，丢弃它的多余 row
```

这就是 one-step lookahead overrun / overshoot。正确性要求：

1. 每个 pending step 保存自己的 `ScheduleBatch.copy()` snapshot。
2. GPU result row 只能用它所属的 snapshot 解释。
3. prior-step 已 finished / retracted 的 row 不得再次 emit、append token 或 free KV。
4. result rows 与 `batch.reqs` 必须一起裁剪，保持 lockstep。

当前 checkout 中 `_resolve_and_process` 会先在 resolve 前记录 `pre_finished`，调用 resolve 时传入 `skip_rids`，再同步裁剪 `next_token_ids` 和 `batch.reqs`。`_drop_stale_overrun` 处理另一种情况：batch 已构造，但 drain pending step 后其中某些请求刚完成或 retract，必须在真正 forward 前删掉这些 stale rows。

### upstream overlap 与 Omni async decode 不是同一开关

当前 `OmniScheduler._event_loop_overlap()` 明确抛出 `NotImplementedError`，因为 chunked prefill 的 `inflight_middle_chunks` 在该 loop 中会落后一轮并破坏 TTS chunk boundary。当前 Omni 支持的是单 stream、一步前瞻的 `enable_async_decode` 路径。它们互斥，不能把 upstream-style `enable_overlap` 与 Omni async decode 当成同一实现。

## 4. Request slot 如何分配

`ReqToTokenPool` 初始化时创建：

```text
req_to_token shape = [size + 1, max_context_len]
free_slots = [1, 2, ..., size]
```

row 0 留给 CUDA Graph padding；真实请求使用 1..size。

新请求并不是刚到网络入口就立即获得 slot。它通常先进入 `waiting_queue`、做 prefix match 和 admission；当它被组成 extend/prefill batch 并调用 `prepare_for_extend → alloc_for_extend` 时，才调用：

```text
alloc_req_slots
→ req_to_token_pool.alloc(reqs)
→ alloc_rows(number_of_new_requests)
```

`alloc_rows` 从 `free_slots` 尾部取行号，并增加这一行的 generation。随后：

```text
req.kv.req_pool_idx = allocated_row
batch.req_pool_indices[b] = req.kv.req_pool_idx
```

Chunked prefill 继续执行时，如果请求仍 `holds_kv`，它复用原来的 request slot，而不是重新分配。

Prefill 分配完成后，新请求的表行被写成：

```text
req_to_token[req_slot, 0:prefix_len] = prefix_indices
req_to_token[req_slot, prefix_len:seq_len] = newly_allocated_slots
```

finish / abort / retract 的安全释放阶段调用 `free(req)`，把 row ID 放回 `free_slots` 并把 `req.kv.req_pool_idx` 设为 `None`。generation 用于区分同一个 row 被释放后又分给另一代请求的情况，避免异步辅助状态把新旧 owner 混淆。

## 5. Radix Cache 如何复用离散 KV slots

核心结论：

> Radix Tree 匹配的是 token prefix；tree node 的 value 保存与这些 token 一一对应的 KV slot IDs。物理 slot 是否连续不影响匹配。

一个简化 node 可以表示为：

```text
key   = [10, 20, 30]
value = [91,  7, 44]
```

含义不是把 KV 放在树里，而是：

```text
token 10 的 KV 在 slot 91
token 20 的 KV 在 slot 7
token 30 的 KV 在 slot 44
```

新请求 token IDs 为 `[10,20,30,40,50]` 时，Radix Tree 按 token IDs 匹配最长前缀 `[10,20,30]`，返回：

```text
prefix_indices = [91, 7, 44]
```

然后新请求得到 request slot，例如 23，并写入：

```text
req_to_token[23, 0:3] = [91, 7, 44]      # 复用
req_to_token[23, 3:5] = [105, 106]       # 为 40、50 新分配
```

最终新请求的逻辑连续序列映射为：

```text
[10, 20, 30, 40, 50]
  ↓   ↓   ↓   ↓    ↓
[91,  7, 44,105, 106]
```

Attention kernel 通过 `req_to_token` gather 这些 locations；不要求物理连续。

### `page_size=1` 与 paged 模式

- `page_size=1`：Radix key 可以在逐 token 边界分叉，value 是逐 token flat slot IDs。
- `page_size>1`：匹配长度向下对齐到整页；child key 使用下一页的 token tuple；value 仍返回 flattened physical indices。`loc // page_size` 得 page ID，`loc % page_size` 得页内 offset。

因此 NHD 是 K/V tensor 的物理 layout，不是 prefix lookup 算法。Radix Cache 不会靠“K/V 地址连续”判断内容相同，而是靠 token key / namespace 匹配，再取出 node.value 保存的 slots。

### 与 vLLM block hash 的联系

vLLM 常把整块 token IDs 和前序 hash 组成 block hash，命中后得到 physical block。SGLang Radix Cache 把 token sequence 组织成压缩前缀树，node 同时保存该段 token 对应的 physical locations。

```text
vLLM:   block hash → physical block ID
SGLang: radix token path → tensor of physical slot IDs
```

两者都把“内容身份”和“物理地址”分开；只是索引结构和复用粒度不同。SGLang 也会为外部 KV events / storage 计算 page hash，但内存 Radix Tree 的核心 prefix traversal 并不要求 NHD slots 连续。

### 为什么多个请求能安全指向同一 slots

KV 对应已确定的相同 causal prefix，不会因后续 token 改变，因此可以只读共享。Radix node / request 会用 lock/reference 状态保护正在使用的 prefix，使 eviction 不会释放仍被活跃请求引用的 slots。新请求只为未命中的 suffix 分配新 slots。

## 6. 四个最容易混淆的结论

1. `request slot` 是地址表的一行；`KV slot` 是 K/V 内存位置，二者不是同一个 pool。
2. PP 决定一个 worker 保存哪些 layers；TP 决定这些 layers 中一个 worker 保存哪些 heads/weight shards。
3. async decode 的 snapshot 保存的是“这一 forward 的 row 解释”，不是复制整份 KV Cache。
4. Radix Cache 用 token prefix 找 slots；slots 可以离散，连续性不是正确性条件。
