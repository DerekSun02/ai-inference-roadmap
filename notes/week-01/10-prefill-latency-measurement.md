# 如何正确测量 Prefill Latency

## 先订正 C1 自测

### C1 与 C8 的 QPS 关系

在没有 client think time、并发始终打满的 closed-loop 中，可用 Little's Law 的交互式
形式近似理解：

```text
throughput ≈ concurrency / mean response time
```

所以：

```text
C1: QPS ≈ 1 / mean latency
C8: QPS ≈ 8 / mean latency
```

但 C8 的公式成立不代表每次 GPU Prefill/Decode batch 都有 8 个请求。`8` 是客户端保持的
outstanding requests；GPU batch size 会随着请求到达、Prefill/Decode 状态、完成时间、
retract 和调度预算动态变化。

### TTS 应使用 RTF

正确缩写是 `RTF`（Real-Time Factor），不是 RTL：

```text
RTF = request latency / generated audio duration
```

它降低了输出音频长度对绝对 E2E latency 的影响，但不会自动消除排队、请求内容差异和
其他系统噪声。

## Prefill Latency 不是客户端指标

客户端只能直接观察 request start、首个音频 chunk 和请求结束。它看不到 Scheduler 何时
真正选中请求，也看不到第一次模型 forward 的边界。因此不能用 TTFC 代替 Prefill
latency，也不能从 E2E 指标中可靠地反推 Prefill latency。

SGLang-Omni 已提供 request-level profiler event：

```text
request_admission
preprocess_start
preprocess_end
scheduler_request_build_start
scheduler_request_build_end
scheduler_queue_enter
scheduler_prefill_start
scheduler_prefill_end
scheduler_first_emit
stage_first_stream_chunk_sent
terminal_response
```

主要区间：

```text
Preprocessing latency
  = preprocess_end - preprocess_start

Request-build latency
  = scheduler_request_build_end - scheduler_request_build_start

Scheduler queue wait
  = scheduler_prefill_start - scheduler_queue_enter

Host-observed first Prefill forward interval
  = scheduler_prefill_end - scheduler_prefill_start
```

源码边界位于 `OmniScheduler._run_batch()`：在 `_model_runner.execute()` 前发出
`scheduler_prefill_start`，第一次执行返回后发出 `scheduler_prefill_end`。事件只对每个
请求记录一次，metadata 还保存 realized batch size。

## 三种不同的“Prefill 时间”

### 1. 用户等待到 Prefill 完成

```text
request admission → scheduler_prefill_end
```

它包含 Preprocessing、request build、queue wait 和第一次模型 forward，适合分析用户为何
等待，但不能命名为纯 Prefill compute。

### 2. Scheduler 观察到的 Prefill forward

```text
scheduler_prefill_start → scheduler_prefill_end
```

这是当前 request event profiler 直接生成的区间。它是很实用的 server-side Prefill
latency，但属于 host-observed wall time，可能包含输入整理、CUDA launch、必要同步和
框架开销。

### 3. 纯 GPU Prefill kernel time

若问题是“GPU 真正执行 Prefill kernels 花了多少时间”，应在 Prefill 窗口内使用：

- Torch Profiler 的 CPU/CUDA Chrome trace；或
- 在正确 CUDA stream 上记录 start/end CUDA Events，并在区间外同步后计算 elapsed time。

不能在 kernel 前后直接用普通 Python `perf_counter()` 后相减，因为 CUDA launch 通常是
异步的；这样常常只测到 host enqueue 时间。

## 推荐实验方法

第一次做 Prefill breakdown 时优先启用低开销 request profiler：

```bash
curl -X POST http://localhost:8000/start_request_profile \
  -d '{"run_id":"prefill-c1","event_dir":"/tmp/profiles/prefill-c1/events"}'

# 运行固定请求集

curl -X POST http://localhost:8000/stop_request_profile -d '{}'

python -m sglang_omni.profiler \
  /tmp/profiles/prefill-c1/events --format table
```

重点读取：

```text
scheduler_queue_enter -> scheduler_prefill_start
scheduler_prefill_start -> scheduler_prefill_end
```

只有发现 Prefill forward 本身异常慢，才开启更重的 Torch Profiler 深挖 kernel。不要在正式
baseline measured run 中临时打开重型 profiler；单独做 profiling pass，避免 profiler
overhead 污染 C1/C8 对比。

## 一句话记忆

```text
TTFC 是用户看到第一段音频的时间；
queue wait 是排队时间；
Prefill event interval 是服务端第一次 forward 的 wall time；
CUDA trace/event 才回答纯 GPU kernel time。
```
