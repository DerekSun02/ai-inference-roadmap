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
10. 掌握单个 Decode step 的 scaled dot-product Attention：`QKᵀ → scale/mask/softmax → weights·V`。
11. 理解 `sqrt(d_k)` 来自 dot-product 标准差，并能解释 Query/Key 的匹配维度必须相同。
12. 理解 scores/weights 的通用 shape 是 `[query_length, key_length]`，Decode 只是 `query_length=1` 的特例。
13. 建立 KV Cache 从 Prefill 分配、Decode 追加到 finish/abort/retract 处理的生命周期。
14. 理解 SGLang 的 `batch row → req_pool_indices → req_to_token → flat KV slot → per-layer K/V` 地址映射。
15. 区分 token slot 与 page：一个 slot 对应一个 token position，一个 page 包含 `page_size` 个 slots。
16. 理解 GPU result row 的安全消费边界，以及 overlap overshoot row 为什么必须用旧 batch snapshot 处理。
17. 区分 TP、DP、PP，并理解 `num_layers_local` 与 `num_kv_heads_local` 的来源。
18. 理解 SGLang-Omni `enable_async_decode` 的 launch-current / resolve-previous 一步前瞻，以及 overrun row 的处理。
19. 掌握 request slot 从 `free_slots` 分配、绑定到 `req.kv.req_pool_idx`、复用和释放的生命周期。
20. 理解 Radix Cache 以 token prefix 为 key、以离散 KV slot IDs 为 value，不依赖 NHD slots 连续。
21. 理解 KV head、MHA/GQA/MQA 的区别，以及 KV Cache 的 head 维由 `num_key_value_heads` 决定。
22. 能用具体数字推导组合并行下的 `num_layers_local`、`num_kv_heads_local` 和 per-rank KV shape。
23. 完成 DP/PP/TP 布局检查：掌握除法公式，并订正 GPU 2 应负责 `layers 16–31, KV heads 0–3`。
24. 选定 Week 1 主实验模型 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`，明确 Base 和低资源模型为后续/回退路径。
25. 完成 Modal Week 1 baseline 基础设施：固定源码和模型 revision、H100 默认资源、模型缓存、服务生命周期、GPU 采样、原始产物与三 cell 运行入口。

## 当前掌握程度

- 能说明 `concurrency=16` 表示系统中最多或目标存在多个并发请求，不代表每一轮恰好出现 16 个新 Prefill 请求。
- 能说明 coalescing 通过短暂等待增加 Prefill batch 的机会，因此可能提高吞吐但增加 TTFC。
- 能说明新请求先形成独立 Prefill batch，完成后才在安全边界 merge 到 `running_batch`。
- 能说明完成请求的 batch row 必须和 `seq_lens`、`req_pool_indices` 等逐行字段一起 compact。
- 能推演请求完成、KV 压力 retract 和新 Prefill 请求 merge 后的 `running_batch` 与 `waiting_queue`。
- 能解释 causal mask 为什么保证旧 token 的 K、V 在追加未来 token 后保持不变。
- 能区分“保留 token 历史”和“保留历史中间 KV 结果”。
- 能解释历史 K/V 会被未来 Query 读取，而历史 Q 在自己的 Attention 行计算完成后不再使用。
- 能解释 batch row compaction 与物理 KV slot 不需要同步搬迁。
- 能从 request slot 和 logical token position 推导 flat KV slot，并计算 page ID 与 page offset。
- 能说明 TP 拆同一层、PP 拆连续层、DP 分不同请求，以及三者可以组合。
- 能说明 async decode 为什么可能多算一个 finished row，以及 batch snapshot 如何保证结果不串行。
- 能说明 Radix Tree 如何把相同 token prefix 映射为可共享的离散 physical KV slots。
- 能画出 `DP=2, PP=2, TP=2` 的 8-GPU rank 布局，并解释 DP replica group 与单 GPU 完整副本的区别。

## 下一步

1. 在 Modal 执行 prepare 和 smoke，确认实际 GPU、VRAM、driver、服务健康和单请求成功。
2. 从模型 config 手算权重、per-token、per-request 和 batch KV Cache bytes。
3. 亲自运行 C1、C8、1 RPS 三组 baseline，各三次，并保存/取回原始结果。
4. 对比手算显存与 `gpu-samples.csv`，解释差额中权重、CUDA Graph、activation、allocator 和 KV Cache 的来源。
5. 完成两个既有 PR 复盘。
6. 更新 Week 1 周报和记分卡，通过后进入 Week 2。
