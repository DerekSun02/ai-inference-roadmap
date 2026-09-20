# KV Head 与 DP / PP / TP 数字例子

用一个固定的示例模型解释：

```text
num_layers = 32
hidden_size = 4096
num_attention_heads = 32       # Q heads
num_key_value_heads = 8        # KV heads
head_dim = 4096 / 32 = 128
dtype = BF16 = 2 bytes
GPU count = 8
DP = 2, PP = 2, TP = 2
```

## 1. KV head 是什么

一个 Attention layer 会把当前 token 的 hidden state 分别投影成 Q、K、V。一个 **head** 是其中一组独立学习的投影和一个长度为 `head_dim` 的子空间。

本例对一个 token、一个 layer：

```text
Q shape = [32 Q heads, 128]
K shape = [ 8 KV heads, 128]
V shape = [ 8 KV heads, 128]
```

这是 GQA：`32 / 8 = 4`，所以每 4 个 Q heads 共用 1 组 K/V head：

```text
Q heads  0..3  → KV head 0
Q heads  4..7  → KV head 1
...
Q heads 28..31 → KV head 7
```

KV Cache 保存历史 token 的 K/V，不保存历史 Q，所以它的 head 维度由 `num_key_value_heads` 决定。

- MHA：`num_q_heads == num_kv_heads`。
- GQA：多个 Q heads 共享一个 KV head。
- MQA：所有 Q heads 共享唯一一个 KV head。

## 2. TP=2 的数字例子

TP 把同一个 layer 横向分片。本例每层 32 个 Q heads、8 个 KV heads：

```text
TP rank 0: Q heads 0..15,  KV heads 0..3
TP rank 1: Q heads 16..31, KV heads 4..7
```

所以：

```text
num_q_heads_local  = 32 / 2 = 16
num_kv_heads_local =  8 / 2 = 4
```

同一个 token 会同时交给两个 TP ranks。每个 rank 完成自己的一半 attention heads；经过 output projection 后，通过 collective communication 合并成该 layer 的完整 hidden output。

单层、单 token 的 KV：

```text
全 TP group: K [8,128], V [8,128]
每个 TP rank: K [4,128], V [4,128]
```

## 3. PP=2 的数字例子

PP 把 32 个连续 layers 分成两个 stages：

```text
PP stage 0: layers 0..15   → 16 layers
PP stage 1: layers 16..31 → 16 layers
```

所以每个 PP worker / stage 中：

```text
num_layers_local = 32 / 2 = 16
```

一个 token 的 hidden state 顺序执行：

```text
[4096] → stage 0 的 layers 0..15
       → 跨 GPU 发送中间 hidden [4096]
       → stage 1 的 layers 16..31
       → 最终输出
```

stage 0 不保存 layers 16..31 的权重或 KV；stage 1 不保存 layers 0..15 的权重或 KV。

## 4. DP=2 的数字例子

DP 创建两个完整的模型副本，但“完整副本”指整个 replica group 合起来有一份完整模型。

- 若只有 DP、没有 TP/PP：每张 replica GPU 都保存完整模型。
- 若 DP 与 TP/PP 组合：每个 DP replica 是一组 GPU；组内每张 GPU 只保存自己的 TP/PP shard，整个组合起来才是一份完整模型。

假设 BF16 模型权重约 14 GB：

```text
仅 DP=2：
GPU 0 保存约 14 GB 完整权重，处理请求 A
GPU 1 保存约 14 GB 完整权重，处理请求 B
总计约 28 GB 权重副本
```

组合 `DP=2, PP=2, TP=2` 时，每个 replica 使用 `PP×TP=4` 张 GPU。两个 replica 的模型内容相同，但各自处理不同请求。

## 5. 8 张 GPU 如何组合

```text
Replica 0（处理 A、C）
GPU 0: PP stage 0, TP rank 0 → layers 0..15,  KV heads 0..3
GPU 1: PP stage 0, TP rank 1 → layers 0..15,  KV heads 4..7
GPU 2: PP stage 1, TP rank 0 → layers 16..31, KV heads 0..3
GPU 3: PP stage 1, TP rank 1 → layers 16..31, KV heads 4..7

Replica 1（处理 B、D）
GPU 4: PP stage 0, TP rank 0 → layers 0..15,  KV heads 0..3
GPU 5: PP stage 0, TP rank 1 → layers 0..15,  KV heads 4..7
GPU 6: PP stage 1, TP rank 0 → layers 16..31, KV heads 0..3
GPU 7: PP stage 1, TP rank 1 → layers 16..31, KV heads 4..7
```

DP 不参与 `num_layers_local` 或 `num_kv_heads_local` 的除法，因为 DP 复制同样的布局：

```text
num_layers_local   = num_layers / PP = 32 / 2 = 16
num_kv_heads_local = num_kv_heads / TP = 8 / 2 = 4
```

因此任意一个 rank 对一个 token 概念上保存：

```text
K [16 local layers, 4 local KV heads, 128]
V [16 local layers, 4 local KV heads, 128]
```

物理实现通常不是一个真正的三维 tensor，而是每个 local layer 各有：

```text
K_layer[token_slot] shape = [4,128]
V_layer[token_slot] shape = [4,128]
```

共 16 个 local layers。

## 6. 用 KV bytes 验证没有丢数据

每个 rank、每个 token 的 KV bytes：

```text
2(K+V) × 16 layers × 4 KV heads × 128 × 2 bytes
= 32768 bytes
= 32 KiB
```

一个 DP replica 有 4 个 ranks：

```text
32 KiB × 4 = 128 KiB / token
```

直接用全模型参数计算：

```text
2 × 32 layers × 8 KV heads × 128 × 2 bytes
= 128 KiB / token
```

两者相等，说明 TP/PP 只是把全模型 KV 分散到 4 张 GPU，没有丢失或重复。`DP=2` 会再复制一份完整模型容量：若两个 replicas 各有一个同长度 token，则物理总量为 `128 KiB × 2 = 256 KiB`。

## 7. 适用前提

上述除法假设 layers 和 KV heads 能均匀分片。实际模型可能有：

- 不均匀 PP layer partition；
- KV heads 少于 TP size 时的 replication；
- embeddings、LM head、MoE experts 的特殊放置；
- pipeline stage 首尾额外模块。

所以公式适合建立主模型；查看具体模型时再以 runtime config 和每个 rank 的实际 shard 为准。
