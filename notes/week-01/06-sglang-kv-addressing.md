# SGLang KV Cache 地址映射

本文基于 SGLang v0.5.19 的普通 MHA KV pool，先解释默认 NHD/HND 布局。量化 KV、MLA、Mamba、SWA 和特殊硬件布局会改变物理 tensor shape，但“batch row → request row → token slot → layer K/V”的寻址思想相同。

## 术语

### Batch row

当前这一次 GPU forward 中，请求在 batch tensor 里的临时行号。例如 `running_batch=[A,C]` 时，A 是 row 0，C 是 row 1。请求完成、merge 或 compaction 后，这个行号可以变化。

### req_pool_indices

一个 shape 为 `[batch_size]` 的整数 tensor。第 `b` 个元素保存 batch row `b` 对应的 request slot ID。它把临时 batch row 映射到相对稳定的 request table row。

```text
batch row:        0   1
request:          A   C
req_pool_indices: 7  19
```

### Request slot

`ReqToTokenPool.req_to_token` 查找表中的一整行，不是 KV 数据本身。SGLang v0.5.19 中表的 shape 是：

```text
[max_running_requests + 1, max_context_len]
```

row 0 是 CUDA Graph padding 使用的 dummy row；真实请求从其他空闲 row 中分配。request slot 19 表示该请求使用查找表第 19 行。

### Logical token position

一个请求序列内部的位置，例如 prompt/output 拼接后的第 0、1、2 个位置。它不是 vocabulary token ID。它作为 `req_to_token` 的列下标。

### req_to_token mapping

二维 int32 查找表：

```text
req_to_token[request_slot, logical_token_position] = flat_KV_slot_id
```

表中保存的不是 K/V 数值，也不是 token ID，而是该 token 的 KV 存放位置。

### KV token slot

KV 全局内存池中的一个扁平位置编号 `loc`。逻辑上，一个 token slot 代表一个 token position；同一个 `loc` 会在每个 Attention layer 各选中一行 K 和一行 V。

### Page

Allocator 分配和释放的连续 slot 组，类似 vLLM block。一页包含 `page_size` 个 token slots。一个 slot 仍只对应一个 token；一个 page 才能容纳 `page_size` 个 token positions。

## 四步寻址

要找 batch row `b` 中请求的逻辑 token position `p`：

```text
r = req_pool_indices[b]
loc = req_to_token[r, p]

page_id = loc // page_size
offset_in_page = loc % page_size
```

然后对每一个本地 Attention layer `l` 读取相同 `loc` 对应的 K/V。

### 默认 NHD layout

每个 layer 分别拥有：

```text
K_l shape = [num_slots + padding, num_kv_heads_local, head_dim]
V_l shape = [num_slots + padding, num_kv_heads_local, v_head_dim]
```

读取：

```text
K_l[loc, :, :]
V_l[loc, :, :]
```

单 layer、单 token slot 的 K shape 是 `[num_kv_heads_local, head_dim]`，V 类似。

### HND paged layout

每个 layer 分别拥有：

```text
K_l shape = [num_pages, num_kv_heads_local, page_size, head_dim]
V_l shape = [num_pages, num_kv_heads_local, page_size, v_head_dim]
```

读取：

```text
K_l[page_id, :, offset_in_page, :]
V_l[page_id, :, offset_in_page, :]
```

这里 `local` 表示 Tensor Parallel 时当前 rank 只存自己负责的 KV heads；Pipeline Parallel 时当前 worker 只存自己负责的 layers。

## 具体例子

假设：

```text
page_size = 4
C 的 request slot = 19

req_to_token[19, 0:6] = [12, 13, 14, 15, 28, 29]
```

则：

```text
C position 0 → loc 12 → page 3, offset 0
C position 1 → loc 13 → page 3, offset 1
C position 2 → loc 14 → page 3, offset 2
C position 3 → loc 15 → page 3, offset 3
C position 4 → loc 28 → page 7, offset 0
C position 5 → loc 29 → page 7, offset 1
```

要找 C 的 position 4、layer 10 的 K：

```text
NHD: K_layer10[28, :, :]
HND: K_layer10[7, :, 0, :]
```

Page 7 还能容纳 offset 2、3 两个 token positions。它们可能在后续 Decode 中使用。

## Prefill 与 Decode 如何写映射

Prefill 为所有未命中的 extend tokens 分配一串 `out_cache_loc`，再写入该请求行对应的逻辑位置区间：

```text
req_to_token[req_slot, prefix_len:seq_len] = out_cache_loc
```

普通 Decode 每个请求新增一个位置。对于 batch row `b`：

```text
r = req_pool_indices[b]
p = current_seq_len
req_to_token[r, p] = out_cache_loc[b]
```

随后每一层使用同一个 `out_cache_loc[b]`，把该输入 token 在当前 layer 产生的 K/V 写入该 layer 的物理 buffer。

## 一个 slot 到底装多少 token

- 一个 flat token slot：一个 token position。
- 一个 page/block：`page_size` 个 token positions。
- 一个 slot ID 在每个 layer 的 K buffer 和 V buffer 中都选择一个位置。

如果把所有本地 layers 概念性地叠起来，一个 token slot 对应：

```text
K [num_layers_local, num_kv_heads_local, head_dim]
V [num_layers_local, num_kv_heads_local, v_head_dim]
```

但物理实现是 K/V 的 Python list，每个 local layer 各有一个 tensor，并不是一定真的 stack 成上述四维 tensor。

当 K/V head dimension 相同时，一个 token slot 在当前 rank 的近似字节数为：

```text
2 × num_layers_local × num_kv_heads_local × head_dim × dtype_bytes
```

## 与 vLLM PagedAttention 的联系

vLLM 常用 block table 表达“sequence logical block → physical block”。SGLang 的 `req_to_token` 直接保存“request logical token position → flattened physical token slot”。当 `page_size>1` 时：

```text
physical page  = loc // page_size
offset in page = loc % page_size
```

所以它仍是 paged allocation，只是 Scheduler 侧持有一个逐 token 的扁平地址表；Radix Cache 还能让不同请求的相同 token prefix 指向可复用的物理 KV locations。

## 为什么 batch compaction 不复制 KV

Compaction 改变的是：

```text
batch row b → req_pool_indices[b]
```

request slot r、`req_to_token[r,p]` 和物理 KV slot loc 都可以保持不变。因此 C 从 row 2 变成 row 1 时，只要新 row 1 的 `req_pool_indices[1]` 仍等于 C 的 request slot，就能继续找到原 KV。

## GPU result row 什么时候算安全消费

GPU forward 的结果 row `i` 属于执行该 forward 时快照中的 `batch.reqs[i]`。安全消费至少包括：

1. 该次 forward 完成，相关 device-to-host copy event 完成；SGLang 会在需要时等待 `result.copy_done`。
2. result processor 使用执行时的 batch snapshot，而不是 compact 后的新 batch。
3. 对每个 `i`，读取 `next_token_ids[i]`，写入正确的 `batch.reqs[i].output_ids`，更新 finish state。
4. 该 row 的 logprob、hidden state、grammar、auxiliary output 等请求级结果也完成映射或明确丢弃。
5. 所有仍引用旧 row 顺序的 in-flight overlap results 都有对应 snapshot，后续处理不会把 row i 误认为另一个请求。

非 overlap 路径中，通常在 `process_batch_result_decode(batch, result)` 完成这一轮逐行处理后，就可以在下一个调度边界过滤 finished rows。

Overlap 路径可能已经提前发射了一个仍含旧请求的 overshoot batch。此时旧请求会在那个 batch snapshot 中再出现一次；结果处理看到它已经 finished/retracted 后跳过，不再追加 token。只有不再需要靠旧 live row order 解释未消费结果时，Scheduler 才能安全地压紧当前 batch。

“正确消费”不要求客户端已经听到或看到输出；它指 Scheduler 已经把 GPU 返回的每一行绑定回正确 Req，并提取完仍需要的逐行 metadata。

## 练习订正

1. Compaction 题回答正确：只需让新 batch row 指向 C 原 request slot，后面的 token mapping 与 KV slots 不变。
2. 普通非 speculative Decode 每个活跃请求新增一个 token position。该 position 在模型的每一个 Attention layer 都新增一份 K 和 V；层数由模型架构决定。若全模型有 32 个 Attention layers，就是 32 份 K 和 32 份 V，PP 时分布在不同 rank。
3. Prefix Cache 题方向正确。更精确地说，新请求的 token prefix 与缓存 key 匹配且模型、权重、adapter 等 cache identity 兼容时，可以复用这些 KV。已完成请求的可复用 prefix 会留在 Radix Cache 中，之后在内存压力下按 eviction policy 释放。
