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
- 1 RPS：open-loop 的 offered load 固定为平均 `1 req/s`，所以系统稳定时 measured QPS
  应接近 1，而不是接近 C1 的 service capacity `2.03 req/s`。用 C1 mean latency 粗估，
  平均 in-flight 约为 `1 × 0.492 = 0.49`；TTFC p50 预计接近 C1，p95 可能因 Poisson
  短间隔造成偶发请求重叠而略高。由于 C1 几乎持续有 1 个请求、1 RPS 则经常处于空闲，
  时间平均 GPU utilization 和 active KV usage 预计低于 C1；但偶发重叠可能让峰值 active
  KV 高于 C1。启动时预分配的 KV pool 不变，因此 `nvidia-smi` 峰值显存仍应接近 C1/C8，
  不会随平均 active KV 成比例下降。

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
| c8 | 2 | NVIDIA H100 80GB HBM3 | 32/32 | 9.068 | 38.654 | 0.0670/0.1125 | 0.1452 | 0.743/1.315 | 0.1875 | 100.0 | 70755.0 |
| c8 | 3 | NVIDIA H100 80GB HBM3 | 32/32 | 8.602 | 36.775 | 0.0716/0.1303 | 0.1468 | 0.766/1.370 | 0.1901 | 100.0 | 70756.0 |
| rps1 | 1 | NVIDIA H100 80GB HBM3 | 60/60 | 1.149 | 5.280 | 0.0336/0.0601 | 0.0832 | 0.518/1.898 | 0.1171 | 96.67 | 70127.0 |
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
- C8 repeat 2/3 都完成 `32/32`，QPS 为 `9.068/8.602`，TTFC p50 为
  `67.0/71.6 ms`，p95 为 `112.5/130.3 ms`。两个 repeat 的结果接近，但与 repeat 1
  的 `4.772 QPS`、`2.0807 s TTFC p95` 明显不属于同一性能状态。
- repeat 1 从第一批 measured requests 提交到 preprocessing，到 server 首次记录其 Prefill
  约相隔 `1.905 s`；repeat 2/3 对应间隔只有约 `57/61 ms`。因此 repeat 1 的约 2 秒
  TTFC 主要发生在 `preprocessing submission → scheduler Prefill` 之间，而不是 Prefill
  之后的 Decode/Vocoder。
- repeat 2/3 中没有请求 TTFC 超过 `1.8 s`，各有 `25/32` 个请求低于 `100 ms`；首批
  8 个请求最大 TTFC 分别为 `113.3/135.8 ms`。repeat 1 的首 cohort 长尾未复现。
- repeat 2/3 的 playback underrun 全为 0，C50/C100/C200 全为 100%；repeat 1 的单个
  `84.8 ms` underrun 也未复现。
- 1-RPS repeat 1 完成 `60/60`、0 失败，测得 `1.149 QPS`。Poisson arrivals 在有限的
  60 请求窗口中不必精确等于 1；本轮 measured phase 约 `52.24 s`，所以完成吞吐为
  `60 / 52.24 ≈ 1.149 req/s`。
- 1-RPS TTFC p50/p95 为 `33.6/60.1 ms`：中心接近 C1，但 p95 已显示偶发重叠的影响。
  57/60 个请求低于 `100 ms`，另有 1 个为 `168.3 ms`、2 个为 `1.5876/1.7340 s`，
  因而 p99 跳到 `1.6476 s`。
- 两个最严重 TTFC 请求在提交 preprocessing 后约 `1.57–1.71 s` 才进入同一个 Prefill。
  同一暂停窗口还使前两个在途请求分别出现 `1.724/1.827 s` 最大 playback underrun；
  所以 C50/C100/C200 都是 `58/60 = 96.67%`。
- measured 窗口的 500 ms GPU 样本平均/峰值 utilization 为 `28.76%/76%`，其中
  `56/104` 个样本为 0%；显著低于 C1 的约 `61.9%` 平均值，验证低 offered load 下
  GPU 经常空闲。server Decode 日志快照仍观察到最多 4 个 running requests，证明
  `1 RPS` 不等于“最大并发 1”。
- 峰值显存 `70127 MiB`，只比 C1 `70077 MiB` 高 `50 MiB`；预分配 KV pool 使显存
  不会随平均 active requests 大幅下降。

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
- C8 repeat 2/3 的稳态均值约为 `8.835 QPS`，是 C1 均值的 `4.35×`；audio throughput
  均值 `37.715 audio-s/s`，是 C1 的 `4.30×`。代价是 TTFC p50/p95 均值约
  `69.3/121.4 ms`，约为 C1 的 `2.10×/3.27×`，RTF p50 约为 `1.66×`。这展示了
  throughput/latency trade-off，而不是“并发 8 所以免费得到 8 倍 QPS”。
- repeat 1 是一次明确的 transient/outlier，不能用它单独代表 C8 稳态。日志把异常区间
  缩小到请求已经提交给 preprocessing 之后、scheduler 首次 Prefill 之前；这个大区间仍
  包含 Preprocessing、跨 stage 传输、request build 和 admission，普通日志不能继续拆分。
  `queue-req: 0` 只是 Prefill 日志采样时刻的队列状态，也不能反推之前从未等待。
- 1-RPS 的中心指标支持“低负载基本无排队”：TTFC p50 与 C1 相同，RTF p50
  `0.1171` 也接近 C1 的 `0.1140`。但两个 early outliers 说明“平均负载低”不等于
  “每个请求都无等待”，尤其不能只凭 p50 得出可靠性结论。
- 1-RPS repeat 1 的异常同样出现在 measured phase 很早，并伴随一个约 2 秒的服务日志
  空窗。它可能与首次出现多请求重叠时的阶段状态有关，但目前普通日志仍不足以把它命名为
  scheduler queueing、lazy initialization 或具体某个 stage 的阻塞。
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
- 三次 C8 并非完全同分布：repeat 1 存在未复现的 pre-Prefill transient。只报三次算术
  均值会把两次稳定运行和一次异常状态混在一起；当前应同时报告全部原始 runs，并把
  repeat 2/3 作为典型 warm steady-state 范围。
- 1-RPS 当前只有一次 repeat，且 2/60 个请求受到 early transient 影响；p95 尚较稳健，
  p99、mean、E2E p95 和 continuity 会被少量异常显著影响，必须由 repeat 2/3 判断。
- C8 与 C1 的逐请求生成时长不一致；可能涉及非 batch-invariant sampling/RNG consumption。
  RTF 能归一化长度，但不能消除第一 cohort 等待或完全消除生成路径差异。

## Week 1 必答题

1. 为什么 C8 的客户端并发不等于每次 Prefill batch 都是 8？
2. C8 相对 C1 的 QPS 收益是否伴随 TTFC/ITL 尾延迟变化？
3. 1 RPS 是否基本无队列？用哪些数据支持，而不是凭感觉判断？
4. 第一段可播放 PCM 与可听见语音为什么必须区分？本次工具实际测到哪一个？
5. 峰值显存由哪些部分构成？为什么不能把它全部叫作 KV Cache？

## 局限与下一步最小验证

- 局限：现有普通日志只能把 repeat 1 异常定位到较宽的
  `preprocessing submission → scheduler Prefill` 区间，不能确定具体子阶段。
- 下一步只改变的一个变量：保持 1-RPS 配置不变运行 repeat 2/3，验证 early transient
  和两个 playback underrun 是否复现。
- 另开一次诊断性 request-profile pass 复现 C8；它回答 transient 的阶段归因，不与正式
  baseline 数字混算。若异常不再出现，也应保留 repeat 1 作为偶发尾延迟证据。
