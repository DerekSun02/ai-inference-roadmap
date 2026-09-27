# Modal Week 1 Lab

本目录只自动化基础设施：构建环境、固定版本、缓存模型、启动服务、采集 GPU 状态并保存原始结果。实验选择、逐次执行、指标阅读和结论由学习者完成。

## 固定对象

- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Model revision: `0c0e3051f131929182e2c023b9537f8b1c68adfe`
- SGLang-Omni commit: `442e559b40b5040965ec876650b32da05d31769f`
- Default GPU: `H100!`，要求实际 H100，避免自动升级到 H200 后混用硬件
- Voice: Ryan
- Corpus: SeedTTS English
- Streaming: raw PCM
- Sampling seed: 1234
- Output cap: 2048 codec tokens

当前 pinned benchmark 没有向请求传递 `language` 的 CLI 参数，因此请求字段保持 `auto`，但 corpus 全部固定为英文。不要在同一矩阵中换语言、voice、模型 revision 或 GPU。

## 1. 本地只做一次

```bash
cd /Users/sunxin/Desktop/ai-inference-roadmap
python3 -m venv .venv-modal
source .venv-modal/bin/activate
python -m pip install --upgrade "modal>=1.1,<2"
modal setup
modal token info
```

模型是公开的，不需要 Hugging Face token。如果遇到下载限流，可在 Modal Dashboard 创建一个 secret，例如 `huggingface`，其中包含 `HF_TOKEN`，然后在每条命令前增加：

```bash
WEEK1_HF_SECRET=huggingface
```

## 2. 准备模型缓存

这一步不占用 GPU：

```bash
modal run modal/week1_qwen3_tts.py --action prepare
```

缓存保存在 Modal Volume `ai-inference-week1-cache`，之后的 run 不应重新下载完整模型。

## 3. 先跑 smoke，不计入实验

```bash
modal run modal/week1_qwen3_tts.py --action smoke --repeat 0
```

通过标准：服务成功 ready、一个请求成功、结果中 `failed_requests=0`。若失败，先看 `server.log` 和 `benchmark.log`，不要直接开始正式矩阵。

## 4. 你亲自执行最小矩阵

每条正式命令各跑三遍。不要一次性写循环：每跑完一遍，先看输出、记录异常，再决定是否继续。

```bash
# Cell A: closed loop, concurrency=1, 32 measured requests
modal run modal/week1_qwen3_tts.py --action c1 --repeat 1

# Cell B: closed loop, concurrency=8, 32 measured requests
modal run modal/week1_qwen3_tts.py --action c8 --repeat 1

# Cell C: open loop, 1 RPS, 60 Poisson arrivals
modal run modal/week1_qwen3_tts.py --action rps1 --repeat 1
```

随后把 `--repeat` 改为 `2` 和 `3`。`rps1` 的到达间隔是指数分布，因此 60 个请求的发送时间期望约为 60 秒，不保证恰好 60.000 秒；实际 wall clock 保存在结果中。

完成正式矩阵后，若 `rps1` 的 measured phase 开头稳定出现暂停，运行单变量诊断 cell：

```bash
# Measured workload 仍是 open-loop 1 RPS；唯一变化是 1 → 8 个并发 warmup requests
modal run modal/week1_qwen3_tts.py --action rps1_warm8 --repeat 1
```

`BenchmarkRunner` 在 `concurrency=0` 时不施加 warmup semaphore，并通过
`asyncio.gather` 同时发出这 8 个 warmup requests。该 cell 只用于验证
concurrency-shaped warmup coverage，不与正式 `rps1` 三次 baseline 求平均。

每次运行前先写下预测：

1. 这个 cell 的 client outstanding、`waiting_queue` 和 `running_batch` 大概会是什么关系？
2. 相比 C1，C8 的 throughput、TTFC 和 ITL 可能怎样变化？
3. 1 RPS 下是否应该持续排队？如果出现明显 queueing，意味着什么？

## 5. 取回原始结果

```bash
modal volume ls ai-inference-week1-results week-01
modal volume get ai-inference-week1-results week-01 benchmarks/raw/modal-week-01
```

每次运行包含：

- `manifest.json`：所有控制变量、实际 GPU、命令、revision、峰值显存和 summary
- `benchmark/speed_results.json`：正式汇总和 per-request 数据
- `server.log`、`benchmark.log`：排障证据
- `gpu-samples.csv`：500 ms GPU 显存、利用率和功耗采样
- `pip-freeze.txt`、`nvidia-smi-q.txt`：软件与硬件环境
- `benchmark/generated_audio/`：生成音频，用于抽查退化、静音和 runaway

本地汇总命令：

```bash
python3 benchmarks/qwen3-tts-1.7b-customvoice/summarize_modal_results.py \
  benchmarks/raw/modal-week-01
```

脚本只汇总事实，不替你解释因果。把输出和自己的解释填写到 `week-01-results.md`。

## 6. GPU 选择规则

默认使用 `H100!`，与上游 Qwen3-TTS H100 数据更可比。只有在 H100 无法取得或预算明确要求时才换卡：

```bash
WEEK1_MODAL_GPU=L40S modal run modal/week1_qwen3_tts.py --action smoke --repeat 0
```

一旦换卡，整套 C1/C8/RPS1 三次重复必须全部使用同一 GPU 型号，并在结果表说明这是独立矩阵，不能与 H100 数字直接合并。Modal 官方推荐 L40S 作为推理的性价比起点，但本实验优先保证与上游 H100 baseline 的可比性。

## 7. 已知限制

- Docker base tag `hongccc/sglang-omni:dev` 是移动标签；核心源码和 Python 依赖由 commit/版本再次固定，但 manifest 仍需如实记录 base tag。正式发表结果前应进一步固定镜像 digest。
- Week 1 不开启 Prefill coalescing、不改 graph bucket、不做 FP8；这些都会引入额外变量。
- `audio_ttfp_*` 是第一段可播放 PCM 到达时间，不等同于第一段可听见语音。
- GPU sampler 是 500 ms 间隔，峰值显存是采样峰值，不是 CUDA allocator 的瞬时精确峰值。
