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

## 8. 用实际 Qwen3-TTS 1.7B 配置手算

实验固定 revision 的 `config.json` 中，主 Talker 为：

```text
num_hidden_layers    = 28
num_attention_heads  = 16
num_key_value_heads  = 8
head_dim             = 128
KV dtype             = BF16 = 2 bytes
TP = 1, PP = 1
```

这是 GQA：每 `16 / 8 = 2` 个 Q heads 共用一个 KV head。注意 `hidden_size=2048`，
这里也满足：

```text
num_attention_heads × head_dim = 16 × 128 = 2048
```

### 单层、单 token

```text
K [8 KV heads, 128] × 2 bytes = 2048 bytes
V [8 KV heads, 128] × 2 bytes = 2048 bytes
K + V                         = 4096 bytes = 4 KiB
```

### 28 层、单 token

```text
KV bytes/token
= 2(K+V) × 28 layers × 8 KV heads × 128 head_dim × 2 bytes
= 114688 bytes
= 112 KiB/token
```

这里第一个 `2` 表示 K 和 V 两份，最后一个 `2 bytes` 表示 BF16 每个元素占两字节。
模型名里的 `1.7B` 不能直接推出 KV Cache 大小；必须使用 layer、KV head、head_dim 和
dtype。

### 按 token 长度换算请求 KV

忽略 allocator 对齐、page 内部碎片和共享 prefix 时：

| 一个请求当前已缓存 token 数 | Talker KV 大小 |
|---:|---:|
| 100 | 10.9375 MiB |
| 1,000 | 109.375 MiB |
| 8,192 | 896 MiB = 0.875 GiB |

请求 KV 随“当前缓存 token 数”线性增长，不是请求一进入就一定占满 8192 tokens。
8192 是该实验配置中的最大上下文需求上界。

### 16 个最大长度 running requests

日志给出的最大配置需求是：

```text
16 running × 8192 tokens = 131072 tokens
```

对应：

```text
131072 × 112 KiB = 14 GiB
```

这表示 16 个请求都达到 8192 cached tokens 时的主 Talker KV 需求，不表示 C1 实际用了
14 GiB。

### 反向验证运行日志的 59.47 GiB

运行日志显示：

```text
KV pool slots = 556787 tokens
K size        = 29.74 GiB
V size        = 29.74 GiB
K + V         = 59.47 GiB
```

用手算结果反推：

```text
556787 slots × 112 KiB/slot
= 59.471268 GiB
```

与日志的 `59.47 GiB` 精确吻合。拆开后 K 和 V 各占一半，约 `29.74 GiB`。

### Capacity、reserved memory 与 active usage

必须区分：

```text
pool capacity   = 总共预分配了多少 physical KV slots
active usage    = 当前请求和 prefix cache 实际占用了多少 slots
free slots      = capacity - active usage
```

`59.47 GiB` 是启动时按显存预算预分配的 KV tensor 容量，不是 C1 单请求的活跃 KV。
这解释了为什么 server ready 已占约 69.8 GiB，而 C1 benchmark 峰值只比 ready 多约
252 MiB。预分配 tensor 的物理显存已经存在，之后请求主要是在 pool 中领取和归还 slot。

配置的 live-request 最大需求是 14 GiB，但 pool capacity 更大；额外 slots 还可承载保留
的 prefix/Radix cache，并提供容量余量。是否能接收请求仍同时受 `max_running_requests`、
context 上限和 Scheduler admission 约束，不能因为 pool 有 556787 slots 就推断可以同时跑
`556787 / 8192` 个最大长度请求。

### Code Predictor 为什么没有加进这条公式

同一个 config 里还有 5 层的 `code_predictor_config`。Code Predictor 确实也计算并缓存
K/V，但它不是主 Talker 那种随整段序列 token 数增长的 paged KV pool。

源码预分配固定 tensor：

```text
K predictor cache shape =
[5 layers, 16 max batch slots, 17 predictor positions, 8 KV heads, 128]

V predictor cache shape = same
```

其中 `17 = num_code_groups + 1`。Predictor 为当前 Talker timestep 逐组预测 codec codes，
完成后这些 batch-slot buffers 会在后续 timestep 复用，而不是为整段 Talker token 历史
持续追加 physical slots。

BF16 下固定 Predictor K+V buffer 约为：

```text
2(K+V) × 5 × 16 × 17 × 8 × 128 × 2 bytes
= 5570560 bytes
= 5.3125 MiB
```

相比之下，主 Talker pool 是 `59.47 GiB`，并且公式
`28 × 8 × 128 × BF16 × K/V` 已精确解释日志。因此：

- `112 KiB/token` 只描述主 Talker 的序列 KV；
- Predictor 固定 cache 属于独立的常驻辅助 buffer；
- Predictor CUDA Graph、Vocoder state 等也分别记入其他显存构成；
- 不能把 Predictor 的 5 层再加进 `112 KiB × Talker token slots`。

### Predictor cache 的 16 batch slots

`16` 来自：

```python
max_batch_size = server_args.max_running_requests
```

本实验 `max_running_requests=16`，所以 Predictor 预留 16 个 batch rows。某次实际 Talker
decode batch 若只有 `B=6` 个请求，只使用 cache 的 `:6`；其余 10 行闲置。下一步 batch
完成 compact/merge 后，这些物理行会被新 batch 重用。

因此这里的 batch slot 是“本次 Predictor forward 的固定 batch-row workspace”，不是：

- 主 Talker 的 `req_pool_idx`；
- paged KV pool 的 token slot；
- 16 个 code groups。

### Predictor cache 的 17 predictor positions

Qwen3-TTS 每个音频 timestep 输出 `num_code_groups=16` 个 codec codes。主 Talker 先给出
第 0 组 code 和当前 Talker hidden；Code Predictor 再按顺序生成剩余 15 组 code。

单个 timestep 内可概念化为：

```text
predictor position 0: 当前 Talker hidden
predictor position 1: 第 0 组 codec code embedding
predictor position 2: 已生成的第 1 组 code embedding
predictor position 3: 已生成的第 2 组 code embedding
...
后续位置: 为预测下一 code group 提供前缀 K/V
```

实现按 `num_code_groups + 1 = 17` 预留 position capacity。当前增量路径在最后一组 code
采样完成后不需要再把它 forward 给 Predictor，因此不保证 17 个位置每次全部写满；`17`
是静态 buffer 上限。

最重要的区别：这些是“同一个音频 timestep 内部的 code-group AR positions”，不是整段
语音沿时间增长的 Talker token positions。处理下一 Talker timestep 时，`cache_len` 又从
0 开始，覆盖并复用同一组 Predictor buffer。这就是它固定为约 5.31 MiB、不会随音频
长度线性增长的原因。

## 9. TP 下不要混淆 per-rank 与 group total

若改成 `TP=2, PP=1` 且 8 个 KV heads 均匀切分：

```text
num_kv_heads_local = 8 / 2 = 4
per-rank KV        = 112 KiB / 2 = 56 KiB/token
whole TP group     = 56 × 2 = 112 KiB/token
```

若保持同样 `556787 slots` 的全局 pool capacity：

```text
每张 GPU 的 pool shard ≈ 59.47 / 2 = 29.735 GiB
整个 TP group pool      ≈ 29.735 × 2 = 59.47 GiB
```

所以 `29.735 GiB` 是每张 TP GPU 的 pool shard，不是整个 TP group 的总量。TP 把数据
分散到多卡，通常不会让同一个 replica 的全局 KV 数据凭空减半。
