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
| c8 | 1 | | | | | | | | | | |
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

## Week 1 必答题

1. 为什么 C8 的客户端并发不等于每次 Prefill batch 都是 8？
2. C8 相对 C1 的 QPS 收益是否伴随 TTFC/ITL 尾延迟变化？
3. 1 RPS 是否基本无队列？用哪些数据支持，而不是凭感觉判断？
4. 第一段可播放 PCM 与可听见语音为什么必须区分？本次工具实际测到哪一个？
5. 峰值显存由哪些部分构成？为什么不能把它全部叫作 KV Cache？

## 局限与下一步最小验证

- 局限：C1 已完成三次，但仍不能观察 batching；每轮 32 请求也不足以精确估计极端尾延迟。
- 下一步只改变的一个变量：将 client concurrency 从 1 改为 8，进入 C8；模型、数据、
  seed、GPU 型号和其余 serving 配置不变。
- 预期结果：C8 可能提高 QPS 和 GPU utilization，
  但单请求 TTFC、ITL 或 E2E tail 可能上升。`concurrency=8` 不保证每次 Prefill batch=8。
