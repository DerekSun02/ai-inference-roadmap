# Week 1 Qwen3-TTS 最小实验矩阵结果

## 实验前假设

- C1：`concurrency=1` 的 closed-loop 不形成跨请求 batching；QPS 应近似为
  `1 / mean latency`。TTFC 应主要反映无排队时的请求进入、Preprocessing、
  Prefill/首批 Decode、Vocoder 和首个 PCM chunk 传输。
- C8：closed-loop 稳定阶段尽量保持 8 个 client outstanding requests。预计 QPS 高于 C1
  但达不到理想 8 倍；TTFC 尤其 p95 可能因 batching、排队和阶段竞争升高。RTF 可能因
  per-request 等待增加而上升，但 batching efficiency 能否抵消一部分增长必须由实验判断。
  KV pool 已物理预分配，峰值显存应接近 C1；可能因 activation、较大 batch graph 和并发
  Vocoder/辅助状态小幅增加，而不会增加 `7 × 59.47 GiB`。
- 1 RPS：

## 固定环境

- Modal GPU（以 manifest 实际值为准）：`NVIDIA H100 80GB HBM3`
- SGLang-Omni commit：`442e559b40b5040965ec876650b32da05d31769f`
- Model revision：`0c0e3051f131929182e2c023b9537f8b1c68adfe`
- Voice：Ryan
- Corpus：SeedTTS English
- Request language：auto（English-only corpus）
- Stream / format：true / PCM
- Seed / max new tokens：1234 / 2048

## 原始结果表

先运行 `summarize_modal_results.py`，再将表格粘贴到这里。不要手工修改 `benchmarks/raw/` 中的数据。

| Cell | Repeat | GPU | Success/Total | QPS | Audio s/s | TTFC p50/p95 | ITL p95 | E2E p50/p95 | RTF p50 | C50 | Peak MiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1 | 1 | NVIDIA H100 80GB HBM3 | 32/32 | 2.019 | 8.726 | 0.0336/0.0379 | 0.0714 | 0.484/0.797 | 0.1154 | 100.0 | 70077.0 |
| c1 | 2 | NVIDIA H100 80GB HBM3 | 32/32 | 2.036 | 8.801 | 0.0332/0.0351 | 0.0695 | 0.480/0.794 | 0.1140 | 100.0 | 70077.0 |
| c1 | 3 | NVIDIA H100 80GB HBM3 | 32/32 | 2.038 | 8.810 | 0.0322/0.0382 | 0.0706 | 0.480/0.813 | 0.1127 | 100.0 | 70077.0 |
| c8 | 1 | NVIDIA H100 80GB HBM3 | 32/32 | 4.772 | 20.207 | 0.0647/2.0807 | 0.1467 | 0.880/3.826 | 0.1949 | 96.88 | 70803.0 |
| c8 | 2 | | | | | | | | | | |
| c8 | 3 | | | | | | | | | | |
| rps1 | 1 | | | | | | | | | | |
| rps1 | 2 | | | | | | | | | | |
| rps1 | 3 | | | | | | | | | | |

## 事实与解释分离

### 直接观察到的事实

- C1 repeat 1 完成 `32/32` 个 measured requests，失败数为 0；另有 1 个 warmup，
  不计入表中。
- 服务启动和 CUDA Graph 捕获约需 5 分钟；服务 ready 后的首个 warmup 请求约需
  `9.9 s`，二者都不计入下面的 steady-state latency/TTFC/QPS。
- mean latency `0.495 s`，throughput `2.019 req/s`；二者近似互为倒数。
- TTFC p50/p95 为 `33.6/37.9 ms`，首个 PCM payload 固定为 `3840 bytes`。
- E2E latency p50/p95/p99 为 `0.484/0.797/0.929 s`；RTF p50/p95/p99 为
  `0.1154/0.1240/0.1290`。
- 平均音频时长 `4.322 s`，audio throughput `8.726 audio-s/wall-s`。
- ITL mean/p95/p99 为 `55.1/71.4/73.0 ms`；32 个多 chunk 请求的最大播放
  underrun 都是 0，因此 C50/C100/C200 都是 100%。
- ready 后显存为 `69825 MiB`，采样峰值 `70077 MiB`；benchmark 窗口的
  500 ms 采样平均/峰值 GPU utilization 为约 `61.9%/75.0%`。
- per-request 的 E2E latency 与音频时长 Pearson correlation 为 `0.9970`。
- 服务日志显示权重加载占用约 `3.63 GB`；预分配 KV pool 为 `556787 tokens`、
  `59.47 GiB`；Vocoder incremental state 为 `159.3 MiB`，此外还有 CUDA Graph、
  activation、allocator reserve 和其他运行时内存。
- C1 三次共完成 `96/96` 个 measured requests，0 失败，且分别运行在三张不同 UUID 的
  H100 上。三次 QPS 为 `2.019/2.036/2.038`，均值 `2.031`，跨 run 极差约为均值的
  `0.94%`。
- 三次 mean E2E 为 `495/491/491 ms`，极差约 `0.81%`；mean RTF 为
  `0.1154/0.1143/0.1141`，极差约 `1.13%`；mean TTFC 为
  `33.5/33.1/32.6 ms`，绝对极差 `0.9 ms`、相对约 `2.72%`。
- 三次音频平均时长都为 `4.322 s`，平均 chunk 数都为 `9.34`，峰值显存都为
  `70077 MiB`，所有请求的 playback underrun 都是 0。
- 三次首个 warmup request 分别约为 `9.91/9.52/9.39 s`，依然远慢于 warmup 后的
  steady-state request，因此 cold/first-request 与 steady-state 的边界具有重复性。
- C8 repeat 1 完成 `32/32`，QPS `4.772`，是 C1 三次均值 `2.031` 的约 `2.35×`，
  未达到理想 8 倍；audio throughput `20.207 audio-s/s`，约为 C1 的 `2.30×`。
- C8 TTFC p50/p95 为 `64.7 ms/2.0807 s`。前 8 个 measured requests 的 TTFC 都在
  `1.9235–2.0812 s`，第 9 个为 `827.7 ms`，其余 23 个均低于 `100 ms`，所以 p95
  由第一 measured cohort 主导，而非全部请求均匀变慢。
- C8 E2E p50/p95 为 `0.880/3.826 s`，RTF p50/mean/p95 为
  `0.1949/0.3684/1.0162`；相对 C1，中心与尾部都变差，但第一 cohort 对 mean/tail
  影响很大。
- C8 ITL p95/p99 为 `146.7/945.9 ms`。31/32 个请求的最大 playback underrun 不超过
  50 ms；1 个请求为 `84.8 ms`，所以 C50=`96.88%`、C100/C200=`100%`。
- C8 peak memory `70803 MiB`，仅比 C1 的 `70077 MiB` 高 `726 MiB`（约 `1.04%`），
  支持“共享预分配 KV pool，而非每请求复制 pool”的预测。
- 同一 32 个 request IDs 的 C1 与 C8 音频时长全部不同，逐请求平均绝对差 `0.852 s`、
  最大差 `3.84 s`，但整体平均为 `4.322 s` 与 `4.235 s`。固定 seed 没有在不同 batch
  shape 下提供逐请求 bitwise/batch-invariant 输出。

### 当前解释

- C1 是单请求服务时间基线，不是 batching throughput。closed-loop client 只有前一个
  请求结束才发下一个，所以 `QPS ≈ 1 / mean latency` 是工作负载定义直接造成的。
- 本表是 warm-server steady-state 基线，不代表 cold start 或新容器的首请求体验；
  deployment startup、首请求 lazy initialization 必须作为不同实验单独测量。
- 绝对 E2E latency 的尾部主要随输出音频长度变化；归一化后的 RTF 分布明显更紧，
  当前证据不支持把 C1 latency p95 直接解释成排队抖动。
- `TTFC=33.5 ms` 是客户端观测的第一段 PCM 到达时间，不是纯 Prefill 时间。它包含
  请求传输、Preprocessing、admission、Prefill、足够的 Decode codec tokens、Vocoder
  和首个 HTTP chunk 到达。
- 第一段 per-request PCM duration 为 `80 ms`。之后的 chunk 到达速度足以覆盖播放
  deadline，因此本轮没有播放断粮；ITL 不能脱离每个 chunk 携带的音频时长单独判断。
- 峰值显存不能全部称为“实际请求使用的 KV”。启动时已经按
  `mem_fraction_static=0.850` 预分配了巨大的 KV pool；C1 运行期间峰值相对 ready 状态
  只增加约 `252 MiB`。
- 三次核心均值的跨 run 极差约为 `0.8%–2.7%`，且控制变量、输出长度和显存一致，
  因此 C1 足够稳定，可以作为同一实验设计下 C8 与 1-RPS 的对照。这里比较的是同型号
  H100 间的 run-to-run repeatability，不代表跨 GPU 型号或跨软件版本可复现。
- C8 的主要收益是吞吐提高约 `2.35×`，代价是 per-request TTFC、E2E、RTF 和 ITL
  上升。它展示的是 throughput/latency trade-off，而不是“并发 8 所以免费得到 8 倍 QPS”。
- 第一 measured cohort 的约 2 秒 TTFC 与后续多数请求低于 100 ms 形成明显双峰。server
  日志显示 measured phase 开始约 1.9 秒后才出现该 cohort 的 Prefill，且粗粒度日志中的
  scheduler queue 为 0；这与 Preprocessing/request-build/admission 之前或阶段交接延迟
  相符，但现有证据不能定位。需要 request-level profiler 拆分，而不能直接称为 Prefill
  慢或 scheduler queueing。
- C8 使用了更多 logical KV slots，但 pool tensor 启动时已经分配。额外 `726 MiB` 更可能
  来自较大 batch 的 activation/workspace、并发辅助状态或 allocator reserve，而不是新增
  KV pool。

### 支持解释的源码、日志或指标

- `speed_results.json` 的 summary 和 32 条 `per_request` 数据。
- `manifest.json` 的实际 GPU、ready/peak memory 和固定控制变量。
- `gpu-samples.csv` 的 500 ms GPU 利用率、功耗和显存采样。
- `server.log` 的 model weight、KV pool、CUDA Graph 和 Vocoder state 日志。
- `benchmarks/metrics/playback_continuity.py`：C50 表示多 chunk 请求中最大 underrun
  不超过 50 ms 的比例；本轮所有请求实际 underrun 都是 0。

### 替代解释

- 每轮仍只有 32 个请求，p95/p99 由很少的尾部样本决定。三次 E2E p95 为
  `794–813 ms`，TTFC p95 为 `35.1–38.2 ms`；尾部分位数的小幅波动不能当作回归。
- 文本长度、生成 codec token 数和音频时长可能共同影响 latency；当前 benchmark 没有
  输出 completion token 数，因此只能确认与最终音频时长高度相关，不能拆出每一项因果。
- GPU utilization 来自 500 ms 抽样，短 kernel 和瞬时峰值会被平滑，不能据此断言 GPU
  有 38.1% 的可直接利用算力。
- C8 只有一次 repeat，且第一 measured cohort 存在独特的约 2 秒 TTFC。必须用 repeat
  2/3 判断它是系统性行为还是单次异常。
- C8 与 C1 的逐请求生成时长不一致；可能涉及非 batch-invariant sampling/RNG consumption。
  RTF 能归一化长度，但不能消除第一 cohort 等待或完全消除生成路径差异。

## Week 1 必答题

1. 为什么 C8 的客户端并发不等于每次 Prefill batch 都是 8？
2. C8 相对 C1 的 QPS 收益是否伴随 TTFC/ITL 尾延迟变化？
3. 1 RPS 是否基本无队列？用哪些数据支持，而不是凭感觉判断？
4. 第一段可播放 PCM 与可听见语音为什么必须区分？本次工具实际测到哪一个？
5. 峰值显存由哪些部分构成？为什么不能把它全部叫作 KV Cache？

## 局限与下一步最小验证

- 局限：C8 目前只有 repeat 1，第一 measured cohort 的约 2 秒 TTFC 尚未证明可重复。
- 下一步只改变的一个变量：保持 C8 配置完全不变，运行 repeat 2/3；若 cohort 长尾重复，
  再单独做 request-level profiling pass。
- 预期结果：若是稳定的阶段交接/构建行为，repeat 2/3 也会看到首 cohort TTFC 长尾；若
  不重复，则把它视为单次运行异常，C8 summary 应用三次统计而不是只报 repeat 1。
