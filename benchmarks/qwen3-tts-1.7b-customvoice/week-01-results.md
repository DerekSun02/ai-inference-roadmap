# Week 1 Qwen3-TTS 最小实验矩阵结果

## 实验前假设

- C1：
- C8：
- 1 RPS：

## 固定环境

- Modal GPU（以 manifest 实际值为准）：
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
| c1 | 1 | | | | | | | | | | |
| c1 | 2 | | | | | | | | | | |
| c1 | 3 | | | | | | | | | | |
| c8 | 1 | | | | | | | | | | |
| c8 | 2 | | | | | | | | | | |
| c8 | 3 | | | | | | | | | | |
| rps1 | 1 | | | | | | | | | | |
| rps1 | 2 | | | | | | | | | | |
| rps1 | 3 | | | | | | | | | | |

## 事实与解释分离

### 直接观察到的事实

-

### 当前解释

-

### 支持解释的源码、日志或指标

-

### 替代解释

-

## Week 1 必答题

1. 为什么 C8 的客户端并发不等于每次 Prefill batch 都是 8？
2. C8 相对 C1 的 QPS 收益是否伴随 TTFC/ITL 尾延迟变化？
3. 1 RPS 是否基本无队列？用哪些数据支持，而不是凭感觉判断？
4. 第一段可播放 PCM 与可听见语音为什么必须区分？本次工具实际测到哪一个？
5. 峰值显存由哪些部分构成？为什么不能把它全部叫作 KV Cache？

## 局限与下一步最小验证

- 局限：
- 下一步只改变的一个变量：
- 预期结果：
