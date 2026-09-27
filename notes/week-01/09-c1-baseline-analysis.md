# C1/C8 基线与并发分析

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

## C8 repeat 1：吞吐收益与长尾代价

C1 使用三次均值作为 baseline，C8 repeat 1 的对比为：

| Metric | C1 baseline | C8 repeat 1 | 变化 |
|---|---:|---:|---:|
| QPS | 2.031 | 4.772 | 2.35× |
| Audio throughput | 8.779 | 20.207 | 2.30× |
| E2E p50 | 481.3 ms | 880 ms | 1.83× |
| E2E p95 | 801.3 ms | 3826 ms | 4.78× |
| TTFC p50 | 33.0 ms | 64.7 ms | 1.96× |
| TTFC p95 | 37.1 ms | 2080.7 ms | 约 56× |
| RTF p50 | 0.1140 | 0.1949 | 1.71× |
| ITL p95 | 70.5 ms | 146.7 ms | 2.08× |
| Peak memory | 70077 MiB | 70803 MiB | +726 MiB |

结果验证了总体预测：continuous batching 提高了 throughput，但没有获得理想 8 倍，且
per-request latency 和流式连续性变差。显存只增加约 1%，验证主 KV pool 是预分配共享
容量。

### TTFC 不是均匀变慢

32 个请求的 TTFC 呈 cohort pattern：

```text
requests 0..7:  1.9235–2.0812 s
request 8:      0.8277 s
requests 9..31: 0.0428–0.0724 s
```

所以 `TTFC p95=2.0807 s` 不能概括成“每个请求都需要约 2 秒”。它主要描述第一 measured
cohort。benchmark 从 measured phase 开始到 server log 首次出现该 cohort Prefill 约有
1.9 秒；现有普通日志不足以区分 Preprocessing、request build、admission 或其他阶段交接。
下一步应先用 repeat 2/3 验证，再用 request event profiler 定位。

### 流式播放出现一个可感知缺口

一个请求出现 `84.8 ms` 最大 playback underrun，因此：

```text
C50  = 31/32 = 96.88%
C100 = 32/32 = 100%
```

这说明 throughput 提升不仅影响 TTFC，也可能影响已经开始播放后的 chunk continuity。

### 固定 seed 不等于 batch-invariant output

同一 32 个 request IDs 在 C1/C8 中生成时长全部改变，平均绝对差 `0.852 s`。整体平均
音频时长仍接近（`4.322 s` vs `4.235 s`），因此 aggregate 对比仍有参考价值，但逐请求
绝对 latency 不再是完全 paired 的 apples-to-apples 比较。RTF 更适合归一化输出长度，
但仍包含 queue/stage wait。

## C8 三次重复：典型稳态与一次 transient

| Metric | C8 repeat 1 | C8 repeat 2 | C8 repeat 3 | repeat 2/3 均值 |
|---|---:|---:|---:|---:|
| QPS | 4.772 | 9.068 | 8.602 | 8.835 |
| Audio throughput | 20.207 | 38.654 | 36.775 | 37.715 |
| E2E p50 | 880 ms | 743 ms | 766 ms | 754.5 ms |
| E2E p95 | 3826 ms | 1315 ms | 1370 ms | 1342.5 ms |
| TTFC p50 | 64.7 ms | 67.0 ms | 71.6 ms | 69.3 ms |
| TTFC p95 | 2080.7 ms | 112.5 ms | 130.3 ms | 121.4 ms |
| RTF p50 | 0.1949 | 0.1875 | 0.1901 | 0.1888 |
| ITL p95 | 146.7 ms | 145.2 ms | 146.8 ms | 146.0 ms |
| Peak memory | 70803 MiB | 70755 MiB | 70756 MiB | 70755.5 MiB |

repeat 2/3 很接近，可视为当前实验的典型 warm steady-state。以两次均值和 C1 三次均值
比较：QPS 提高约 `4.35×`，audio throughput 提高约 `4.30×`；TTFC p50/p95 则提高到
约 `2.10×/3.27×`，RTF p50 约为 `1.66×`。所以你的四个预测中，QPS、TTFC、RTF 和
显存的方向都对；RTF 不是只“稍微”升高，而是典型值约增加 66%。

repeat 1 不能代表典型 C8：第一批 8 个请求 TTFC 为 `1.9235–2.0812 s`，但 repeat 2/3
首批最大值只有 `113.3/135.8 ms`。后两次均无请求超过 `1.8 s`，各有 `25/32` 个请求
低于 `100 ms`。因此首批约 2 秒不是 continuous batching 的必然代价，而是一次未复现的
transient。

### 日志把异常定位到了哪里

三次请求都先由 Coordinator 提交给 preprocessing，然后才出现在 Scheduler 的 Prefill
日志中：

```text
repeat 1: submission 22:38:37.732 → first Prefill 22:38:39.637 ≈ 1905 ms
repeat 2: submission 00:42:39.385 → first Prefill 00:42:39.442 ≈   57 ms
repeat 3: submission 00:49:09.681 → first Prefill 00:49:09.742 ≈   61 ms
```

所以 repeat 1 的主要额外时间发生在：

```text
Coordinator submission
→ Preprocessing
→ stage transport / request build / admission
→ Scheduler starts Prefill
```

这排除了“主要慢在 Prefill 后的 Decode 或 Vocoder”，但不能进一步断言就是
Preprocessing、request build 或 scheduler queue 中的哪一项。Prefill 日志显示的
`queue-req: 0` 只是该日志时刻的快照；请求可能尚未进入 scheduler queue，也可能已经
出队，不能用它证明整个区间不存在等待。精确归因需要 request event profiler。

### 为什么不直接对三次取平均

如果把三次直接求均值，会得到一个既不像 repeat 1、也不像 repeat 2/3 的混合状态。
性能实验应保留两个结论：

1. typical steady-state：用相互接近的 repeat 2/3 给出范围或均值；
2. tail reliability：repeat 1 证明 warmup 后仍可能出现一次较大的 pre-Prefill transient。

repeat 2/3 的 playback underrun 全为 0，repeat 1 唯一的 `84.8 ms` underrun 也未复现。
峰值显存三次为 `70803/70755/70756 MiB`，仍只比 C1 高约 0.7 GiB，支持 KV pool 已预分配
且共享的解释。

## 1-RPS repeat 1：低 offered load 不等于最大并发 1

实验使用 open-loop Poisson arrivals：客户端按到达计划发送请求，不等前一个请求结束，且
没有 client-side concurrency cap。60 个请求在约 `52.24 s` 的 measured window 中完成：

```text
throughput = 60 / 52.24 ≈ 1.149 req/s
```

这并不违反配置的 `1 RPS`。Poisson 的 1 RPS 是长期平均到达率，有限样本中的实际总间隔
会随机波动；本轮到达序列恰好比期望的约 60 秒更紧凑。系统处理能力又高于 offered load，
所以 completion throughput 能跟上 realized arrival throughput。

### 中心指标符合预测

| Metric | C1 三次典型值 | 1-RPS repeat 1 | 解释 |
|---|---:|---:|---|
| QPS | 2.031 | 1.149 | C1 是 service-limited；1-RPS 是 arrival-limited |
| TTFC p50 | 33.0 ms | 33.6 ms | 大多数请求几乎无排队 |
| TTFC p95 | 37.1 ms | 60.1 ms | 偶发重叠增加尾部 |
| RTF p50 | 0.1140 | 0.1171 | 典型单请求效率接近 C1 |
| ITL p95 | 70.5 ms | 83.2 ms | 流式间隔略升 |
| Peak memory | 70077 MiB | 70127 MiB | 预分配 pool 不变，仅高 50 MiB |

measured 窗口内，500 ms GPU utilization 样本平均为 `28.76%`，明显低于 C1 的约
`61.9%`；104 个样本中 56 个为 0%。这订正了“open-loop 会比 C1 更忙”的直觉：C1
完成一个就立即补一个，始终尽量保持一个 active request；1-RPS 则经常没有请求，只在
Poisson 短间隔时出现重叠。

server 的 Decode 日志快照观察到的最大 `running-req` 是 4。它再次说明：

```text
arrival rate = 1 request/second
不等于
maximum concurrency = 1 request
```

### 两类尾部异常发生在同一个早期暂停窗口

60 个请求的 TTFC 分布：

```text
57 requests: < 100 ms
1 request:      168.3 ms
2 requests:    1587.6 ms / 1734.0 ms
```

两个慢 TTFC 请求分别在 `15:58:26.914` 和 `15:58:27.061` 提交给 preprocessing，直到
`15:58:28.627` 才作为同一个 Prefill batch 出现在 scheduler 日志中。与此同时，前两个
已经开始生成的请求分别出现 `1.724 s` 和 `1.827 s` 最大 playback underrun。因此这是
一次影响整条服务进度的约 2 秒暂停，不只是两个新请求在普通 scheduler queue 中排队。

结果为：

```text
TTFC p50/p95/p99 = 33.6 ms / 60.1 ms / 1647.6 ms
C50/C100/C200    = 96.67% / 96.67% / 96.67%
```

p50/p95 描述了绝大多数低负载请求的良好体验；p99 与 continuity 则揭示了少量严重暂停。
普通日志还不能判断根因是某个 stage 的 lazy initialization、阶段阻塞还是其他运行时事件。
repeat 2/3 的任务是判断它是否像 C8 repeat 1 一样不再复现。

## 1-RPS 三次重复后的结论

| Metric | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---:|---:|---:|
| QPS | 1.149 | 0.869 | 0.858 |
| Measured horizon | 52.24 s | 69.09 s | 69.90 s |
| TTFC p50 | 33.6 ms | 40.7 ms | 29.4 ms |
| TTFC p95 | 60.1 ms | 74.6 ms | 1289.0 ms |
| TTFC p99 | 1647.6 ms | 677.5 ms | 2334.5 ms |
| E2E p50 | 518 ms | 531 ms | 492 ms |
| RTF p50 | 0.1171 | 0.1242 | 0.1093 |
| C200 | 96.67% | 96.67% | 96.67% |
| Mean GPU utilization | 28.76% | 22.91% | 24.37% |
| Peak memory | 70127 MiB | 70131 MiB | 70223 MiB |

QPS 的差异主要来自有限 Poisson arrival trace：60 个请求被安排在不同长度的随机时间窗，
系统只是跟随到达速度完成请求。它不说明 repeat 1 的服务 capacity 比 repeat 2/3 高。
三次 `180/180` 全部成功，TTFC p50、E2E p50 和 RTF p50 都接近 C1，说明低 offered
load 下的 typical path 健康。

但是 early transient 在三次中都复现：每次恰好是 measured request 0 和 1 已经开始生成
后，服务出现约 2–2.7 秒无 Prefill/Decode 进展的窗口。这两个请求每轮都发生大于 1 秒的
playback underrun，所以三轮 C200 都是：

```text
58 / 60 = 96.67%
```

暂停期间继续到达的新请求会等待，并在服务恢复后 Prefill。因此不同 Poisson trace 决定了
有多少请求的 TTFC 被暂停覆盖：三轮 TTFC 大于 1 秒的数量分别是 `2/1/4`。这解释了为何
repeat 1/2 的 p95 仍很低，而 repeat 3 的 p95 突然成为 `1.289 s`：

```text
60 requests × 5% = 3 requests
```

尾部异常恰好在 p95 排名边界附近，一个或两个额外受影响请求就能让 p95 跳变。这里报告
`count above threshold + p99 + continuity` 比只报 p95 更可靠。

三轮都只做了一个单请求 warmup，而异常在首次出现 measured 并发之后发生。当前最强但
尚未证实的假设是：single-request warmup 没覆盖首次多请求路径的一次性初始化或阻塞。
最小因果实验不是继续重复同一配置，而是保持 measured `1 RPS` 不变，只把 warmup 改成
一次并发 warmup。如果异常消失，才有证据支持 warmup coverage；具体代码位置仍需要
request-level profiling。
