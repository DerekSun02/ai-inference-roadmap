# KV Cache 为什么避免重复计算

## Retract 后是否重新 Prefill

核心判断不是“请求是否回到 waiting queue”，而是“此前的 KV 是否还保存在某个可恢复的位置”。

在普通的非 PD-disaggregation 路径中，SGLang v0.5.19 的 Decode retract 会释放该请求的 GPU KV，调用 `reset_for_retract()`，把 `already_computed` 和 committed KV length 重置为 0，再把请求加入等待队列。因此该请求再次 admission 时需要通过 EXTEND/Prefill 重建 KV。普通 token 请求会保留已经生成的 `output_ids`，所以重建的是 prompt 加已生成历史，不是丢掉输出从第一个 token 重新生成。

可能的例外：

- 如果 Prefix/Radix Cache 中存在可复用的匹配前缀，只需计算没有命中的后缀。
- 在 Decode disaggregation 模式中，可以把 retracted KV 备份到 CPU host pool，恢复时再加载回 GPU，从而避免重新计算。
- 如果 host backup 失败，请求可能无法恢复并被 abort。
- 使用 `input_embeds` 的特殊路径难以混合原始 embedding 与生成 token，源码会清空生成结果并重新开始 Prefill 和生成。

因此更准确的回答是：**通常需要重新进入 Prefill/Extend，但是否重算全部历史取决于 KV 是否被 host backup 或 prefix cache 保留。**

## 一个 token 在 Attention 中产生什么

对某层的 token hidden state `h_i`，模型计算：

```text
q_i = h_i W_Q
k_i = h_i W_K
v_i = h_i W_V
```

生成第 `t` 个 token 时，新 query `q_t` 需要和从第一个 token 到当前位置的所有 key 做匹配，再用权重汇总所有 value：

```text
scores_t = q_t × [k_1, k_2, ..., k_t]^T
output_t = softmax(scores_t) × [v_1, v_2, ..., v_t]
```

## 没有 KV Cache 为什么重复计算历史

模型本身接收的是 token IDs，而不是“上一次 forward 的中间结果”。如果系统只保存 token IDs，那么下一次 Decode 看到更长的序列时，需要重新从第一层开始恢复所有历史位置的中间状态，并重新计算它们在每一层的 K、V。

```text
已有 [A, B, C]，生成 D：
重新计算 A/B/C 的每层 K、V → 计算 D

已有 [A, B, C, D]，生成 E：
再次重新计算 A/B/C/D 的每层 K、V → 计算 E
```

其中 A、B、C 的 K、V 在两步之间没有改变，却被重复计算。

## 为什么历史 K、V 不会变化

Decoder 使用 causal mask：旧 token 只能看自己和更早的 token，不能看到未来 token。追加 D 后，A、B、C 在各层的表示不会因为未来出现 D 而改变，因此它们的 K、V 可以安全复用。

## 使用 KV Cache 后

Prefill 一次性计算 prompt 中所有 token 的 K、V，并按 layer 和 token position 保存。每个 Decode step 只需要：

1. 对最新 token 计算本层的 Q、K、V。
2. 用新 Q 读取历史 KV Cache。
3. 把最新 K、V 追加到 Cache。
4. 进入下一层并重复。

```text
Prefill [A, B, C] → cache K/V(A,B,C)
Decode D → 读取旧 cache，只计算并追加 K/V(D)
Decode E → 读取旧 cache，只计算并追加 K/V(E)
```

每层旧 token 的 projection、attention 和 MLP 不再执行。代价是 KV Cache 随活跃请求数和序列长度持续占用 GPU 显存，并在 Decode 时不断从显存读取，因此长上下文 Decode 常受到显存容量与带宽限制。

## 单步计算形状直觉

长度为 `n` 的上下文：

- 没有 KV Cache：重新处理 `n` 个位置，Attention 近似形成 `n × n` 的关系。
- 使用 KV Cache：只处理 1 个新位置，新 Q 与 `n` 个历史 K 形成 `1 × n` 的关系。

KV Cache 并没有让新 token 不看历史，而是让它直接读取历史计算结果。

## 下一步

1. 为什么保存 K、V 而不是保存 Q。
2. Prefill、Decode、finish、abort、retract 中 KV 的完整生命周期。
3. 使用模型配置手算每个 token 和每个请求的 KV Cache bytes。
