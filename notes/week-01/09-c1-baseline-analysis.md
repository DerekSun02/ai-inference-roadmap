# C1 单请求基线分析

## 实验定义

- 模型：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- GPU：Modal `NVIDIA H100 80GB HBM3`
- 工作负载：closed-loop，`concurrency=1`
- warmup：1 个请求，不计入统计
- measured requests：32
- streaming response：raw PCM

`concurrency=1` 的含义是：客户端在前一个请求完整结束后，才发送下一个请求。因此系统
没有跨请求的 continuous batching 机会。这一格不是为了测最大吞吐，而是建立“没有客户
端排队、没有跨请求 batching”时的服务时间基线。

## 最关键的恒等关系

本轮：

```text
mean latency = 0.495 s/request
1 / mean latency = 2.020 requests/s
measured QPS = 2.019 requests/s
```

二者几乎相等不是巧合。在 `concurrency=1` 的 closed-loop 中，一个请求完成才发下一个，
因此吞吐主要由单请求平均完成时间决定。C1 的 QPS 不能用来证明 continuous batching
有效；必须与 C8 对比才能讨论 batching throughput。

## 这是 steady-state，不是 cold-start 数据

本轮从函数开始到服务 ready，模型加载和多组 CUDA Graph 捕获大约用了 5 分钟。服务
ready 后还执行了 1 个约 `9.9 s` 的 warmup request。报告中的 32 个请求是在 warmup
之后才开始，所以上面的 TTFC、latency 和 QPS 都是 warm-server steady-state 指标。

需要严格区分：

```text
deployment cold start ≠ first-request warmup ≠ steady-state request
```

当前 C1 回答第三项。它不能说明新 Modal container 启动需要多久，也不能代表容器 ready
之后第一位用户一定能获得 33.5 ms TTFC。cold start 和首请求 lazy initialization 应在
独立实验中测，不能混入 steady-state baseline。

## TTFC 测到的不是纯 Prefill

本轮 TTFC：

```text
p50 = 33.6 ms
p95 = 37.9 ms
p99 = 38.3 ms
```

这里工具记录的是客户端第一次收到非空 PCM chunk 的时间：

```text
request
→ Preprocessing
→ admission
→ Prefill
→ Decode 出足够 codec tokens
→ Vocoder 生成第一段 PCM
→ HTTP 第一段 payload 到达 client
```

所以这个字段在输出里叫 `audio_ttfp_s`，报告展示为 TTFC。它是用户侧 streaming
首包延迟，不可以直接命名为 Prefill latency。

首包为 `3840 bytes`。per-request 数据显示第一段包含 `0.08 s` 音频；这与 24 kHz、
mono、PCM16 的 `24000 × 1 × 2 × 0.08 = 3840 bytes` 一致。

## 为什么 E2E p95 比 p50 高很多

```text
E2E p50/p95/p99 = 0.484 / 0.797 / 0.929 s
RTF p50/p95/p99 = 0.1154 / 0.1240 / 0.1290
corr(E2E latency, audio duration) = 0.9970
```

绝对 E2E latency 会随输出音频长度增加。本轮 latency 与音频时长的相关系数达到
`0.9970`，而除以音频时长后的 RTF 分布紧得多。因此当前证据更支持：E2E 长尾主要
来自样本生成的音频更长，而不是 C1 中存在明显排队。

RTF 的定义：

```text
RTF = generation wall time / generated audio duration
```

本轮平均 RTF `0.1154`，意思是生成 1 秒音频平均需要约 0.115 秒 wall time，也就是
大约 `1 / 0.1154 ≈ 8.67×` real-time。报告的 audio throughput 是 `8.726 audio-s/s`，
两者方向一致；小差异来自聚合方式和四舍五入。

## ITL 与播放连续性必须一起读

本轮：

```text
ITL mean/p95/p99 = 55.1 / 71.4 / 73.0 ms
audio chunks mean = 9.34
max playback underrun = 0
C50/C100/C200 = 100% / 100% / 100%
```

ITL 是相邻 PCM chunk 到达客户端的间隔。它不能脱离 chunk 携带的音频时长独立判断。
例如第一段携带 80 ms 音频；若下一段在播放 deadline 前到达，就不会断粮。

benchmark 使用一个简单 playback buffer 模型：第一段到达后开始累计可播放时长；以后
每段若晚于已有音频的播放 deadline 才产生 underrun。本轮所有 32 个请求的最大 underrun
都是 0，所以 C50/C100/C200 都为 100%。C50 的严格含义是“最大 underrun 不超过
50 ms 的多 chunk 请求比例”，不是“ITL 小于 50 ms 的 chunk 比例”。

## 显存应该怎样读

```text
GPU total                  81559 MiB
server ready               69825 MiB
sampled peak               70077 MiB
peak - ready                 252 MiB
```

日志同时给出：

- Talker weights 加载约 `3.63 GB`；
- KV pool 预分配 `556787 token slots`，K+V 共 `59.47 GiB`；
- Vocoder incremental state 共 `159.3 MiB`；
- 还有 Prefill/Decode/Vocoder CUDA Graph、activation、allocator reserve 和运行时 buffer。

因此 `70077 MiB` 不能全部叫作“当前请求的 KV Cache”。其中最大部分是启动时预留的
KV pool 容量，即使 C1 同时只跑一个请求，这块 pool 仍然已被保留。C1 benchmark 期间
相对 ready 状态只上升约 `252 MiB`。

## 当前能够下的结论

1. C1 路径健康：`32/32` 成功，0 失败。
2. 无跨请求 batching 时，QPS 与平均 latency 的倒数吻合。
3. 客户端首个可播放 PCM p95 约 38 ms，但这不是纯 Prefill latency。
4. 本轮 E2E 长尾主要由输出音频长度解释，RTF 更适合跨样本比较生成速度。
5. 流式 chunk 足以持续播放，本轮没有 playback underrun。
6. H100 大部分显存是启动时预分配的 KV pool 和常驻状态，不能按 C1 活跃 KV 使用量解释。
7. 当前数值是 warmup 后的 steady-state，不包含 deployment cold start 和首请求 warmup。

## 三次重复后的稳定性结论

| Metric | Repeat 1 | Repeat 2 | Repeat 3 | 三次均值 | 跨 run 极差/均值 |
|---|---:|---:|---:|---:|---:|
| QPS | 2.019 | 2.036 | 2.038 | 2.031 | 0.94% |
| Mean E2E | 495 ms | 491 ms | 491 ms | 492.3 ms | 0.81% |
| Mean RTF | 0.1154 | 0.1143 | 0.1141 | 0.1146 | 1.13% |
| Mean TTFC | 33.5 ms | 33.1 ms | 32.6 ms | 33.1 ms | 2.72% |
| Peak memory | 70077 MiB | 70077 MiB | 70077 MiB | 70077 MiB | 0% |

三次分别使用不同 UUID 的 H100，但每次都是相同 H100 80GB 型号、driver、模型 revision、
源码 commit、seed 和 32 个数据样本。所有 `96/96` measured requests 成功，平均音频时长
和 chunk 数也完全一致。

因此 C1 的中心指标具有良好 repeatability，可以作为 C8/1-RPS 对照。TTFC 看起来相对
波动最大，是因为它本身只有约 33 ms：`0.9 ms` 的绝对变化换算后就是 `2.72%`。判断
回归时必须同时看 absolute delta 和 relative delta。

尾部分位数仍要谨慎：每轮只有 32 个请求，p95/p99 实际由少数请求决定。三次 E2E p95
为 `794–813 ms`、TTFC p95 为 `35.1–38.2 ms`，这种量级的差异目前属于重复实验中的
正常波动，而不是性能回归证据。

## 还不能下的结论

1. 不能从 C1 说明 continuous batching 提升了多少吞吐。
2. 三次 repeat 支持当前固定环境下的 C1 稳定性，但不能外推到其他 GPU 型号、软件版本
   或 workload。
3. 不能从 500 ms GPU utilization 抽样直接估算仍可获得多少吞吐。
4. 不能把 TTFC 等同于 Prefill latency，也不能把峰值显存全部等同于 KV Cache。
5. 不能用这组 steady-state 数字描述 cold start 或首请求延迟。

## 自测题

1. 为什么本轮 `QPS ≈ 1 / mean latency`，到了 C8 后通常不再成立？
2. 为什么 E2E p95 明显高于 p50，但 RTF p95 与 p50 很接近？
3. TTFC 的 33.5 ms 在进入客户端前至少跨过了哪些阶段？
4. 为什么 ITL p95 为 71.4 ms，仍然可以得到 0 playback underrun？
5. 为什么 `70077 MiB` 不能叫作“一个请求占用的 KV Cache”？

## Prefill latency 的服务端测量边界

客户端 TTFC 不能代替 Prefill latency。SGLang-Omni 的 request profiler 已提供：

```text
preprocess_start → preprocess_end
scheduler_request_build_start → scheduler_request_build_end
scheduler_queue_enter → scheduler_prefill_start
scheduler_prefill_start → scheduler_prefill_end
```

其中：

```text
queue wait
= scheduler_prefill_start - scheduler_queue_enter

host-observed first Prefill forward
= scheduler_prefill_end - scheduler_prefill_start
```

第二个区间在 `OmniScheduler._run_batch()` 中包围第一次 `_model_runner.execute()`，可能
包含输入整理、CUDA launch、必要同步和框架开销。若要回答纯 GPU Prefill kernels 花了
多久，需要在独立 profiling pass 中使用 Torch Profiler CUDA trace 或同 stream 的 CUDA
Events；不能直接在异步 CUDA launch 前后用 Python `perf_counter()` 相减。

低开销 request profiler 通过 `/start_request_profile` 和 `/stop_request_profile` 控制。
重型 Torch Profiler 应单独运行，避免污染正式 C1/C8 baseline。

## `nvidia-smi` 在本实验中测量什么

`nvidia-smi` 是 NVIDIA System Management Interface 的命令行工具。它通过 NVIDIA
driver 查询或管理 GPU 状态，常见字段包括：

- GPU 型号、UUID 和 driver version；
- device-wide memory used / total；
- GPU utilization；
- 温度、功耗；
- 占用 GPU 的进程。

本实验每 500 ms 执行连续采样，记录：

```text
timestamp,index,name,uuid,
memory.used,memory.total,utilization.gpu,power.draw
```

正确解释边界：

- `memory.used` 是整张设备已占显存，包括权重、预分配 KV pool、CUDA Graph、allocator
  reserve、context 和其他 buffer，不是某一个请求的显存；
- `utilization.gpu` 表示采样窗口内 GPU 有 kernel 执行的时间比例，不等于 tensor core
  达到了多少理论 FLOPS，也不能直接推出还剩多少吞吐空间；
- 500 ms polling 会漏掉更短的瞬时峰值，因此采样 peak 不是 CUDA allocator 的精确峰值；
- `nvidia-smi` 适合做硬件身份、device-wide 显存和粗粒度利用率证据；kernel 级归因应使用
  Torch Profiler、Nsight Systems/Compute 或 CUDA Events。
