# Qwen3 TTS 请求到音频的链路

## 高层路径

```text
Client
  → Speech Service
  → Preprocessing
  → Admission and Scheduling
  → Prefill
  → Repeated Decode
  → Codec token chunk
  → Vocoder
  → Playable waveform
```

## 职责边界

### Pipeline

Pipeline 组织端到端业务 Stage，例如预处理、TTS generation engine、Vocoder 和输出传递。它回答的是：一个请求要经过哪些处理环节，以及各环节如何连接。

### Scheduler

Scheduler 主要管理某个 generation engine 内部的 GPU 工作，而不是调度整个 Pipeline 的所有 Stage。它负责 waiting queue、admission、Prefill/Decode 选择、batch 构造、KV 资源约束、完成和取消等。

## 第一段可播放音频出现前

至少需要完成：

1. 文本、speaker/reference audio 等预处理。
2. 请求进入生成引擎并等待 admission。
3. Prefill 建立上下文与 KV cache。
4. Decode 产生足够的 codec tokens。
5. Vocoder 收集到可解码 chunk。
6. Codec tokens 转成 waveform 并发回客户端。

因此 TTFC 或 TTFA 不只是模型第一次 forward 的时间，还可能包含排队、预处理、Prefill、若干 Decode step、chunk accumulation 和 Vocoder。

## 待补源码定位

| 环节 | 文件 | 类或函数 | CPU/GPU | 队列或同步点 |
|---|---|---|---|---|
| Speech API | 待填写 | 待填写 | CPU | HTTP/request queue |
| Preprocessing | 待填写 | 待填写 | CPU/GPU | prepared request registry |
| Admission | `omni_scheduler.py` | `get_new_batch_prefill` | CPU | `waiting_queue` |
| Prefill/Decode | 待填写 | 待填写 | GPU | scheduler step |
| Streaming Vocoder | 待填写 | 待填写 | GPU/CPU | codec chunk buffer |
| Output | 待填写 | 待填写 | CPU | output queue/socket |
