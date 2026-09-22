# Week 1 Qwen3 TTS 1.7B CustomVoice Baseline Plan

## 1. Model decision

Primary model:

```text
Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
config: examples/configs/qwen3_tts_1_7b_customvoice.yaml
candidate model revision: 0c0e3051f131929182e2c023b9537f8b1c68adfe
```

The candidate revision is the checkpoint revision recorded in SGLang-Omni issue #1754 for its H100 open-loop baseline. Verify availability before download, then record the resolved Hugging Face commit rather than relying on a moving branch name.

### Why this is the primary model

1. SGLang-Omni issue #1081 lists both Qwen3-TTS 1.7B Base and CustomVoice in the highest-priority voice-model performance group.
2. It reuses the exact `preprocessing → tts_engine → streaming vocoder` path already studied.
3. CustomVoice needs no per-request reference audio, removing reference-clip duration, transcript, encoding and speaker-similarity variation from the first serving baseline.
4. The optimization tracker #1754 supplies a directly comparable methodology: warm server, concurrency sweep, open-loop request rates, TTFA/TTFC, latency, RTF, continuity and failures.
5. The 1.7B checkpoint is the supported CustomVoice choice; the repository cookbook marks 0.6B CustomVoice as backward-compatible but not recommended.

### Why not use Base first

Base is the second experiment, not the first. It adds reference audio and reference encoding, which are valuable for voice-cloning studies but introduce extra variables before the Scheduler/Talker/Vocoder baseline is stable.

### Fallbacks

- If available VRAM or startup stability cannot support the 1.7B profile, use `Qwen/Qwen3-TTS-12Hz-0.6B-Base` as a documented resource-constrained fallback. Keep reference audio fixed across all requests.
- Do not switch to Qwen3-Omni 30B for Week 1: it expands model size and multimodal stages without helping the immediate baseline goal.
- MOSS-TTS-Nano-100M is useful for low-resource smoke tests but is less representative of the Scheduler/KV/streaming bottlenecks targeted by the current study.
- Whisper/Qwen3-ASR are strong later comparison tracks, but switching modality now would discard the TTS code-path context already built.

## 2. Fixed software identity

```text
sglang-omni commit: 442e559b40b5040965ec876650b32da05d31769f
sglang: 0.5.19
transformers: 5.12.1
torch: 2.13.0
qwen-tts: 0.1.1 installed with --no-deps
```

Hardware, CUDA, driver, container digest and resolved model commit remain to be filled on the experiment host.

## 3. Week 1 minimum experiment matrix

Use one fixed English voice and language for every run:

```text
voice: Ryan
language: English
instructions: omitted
stream: true
response format: PCM
```

Start with three cells, three measured repeats each after warmup:

| Cell | Load model | Initial setting | Purpose |
|---|---|---|---|
| A | Closed loop | concurrency 1 | Single-request latency and memory baseline |
| B | Closed loop | concurrency 8 | Continuous-batching baseline |
| C | Open loop | 1 RPS for 60 s | Queue-free arrival-rate baseline |

If the GPU cannot run C8 without failures or memory pressure, reduce the fixed concurrency to C4 and record the reason. Do not silently change it.

After Week 1, expand toward the issue #1754 matrix: C1/C8/C16/C32 and 1/6/10 RPS.

## 4. Metrics

Record at minimum:

- success, failure, capped and degenerate request counts;
- request throughput and audio-seconds throughput;
- first playable PCM time and audible TTFA when available;
- TTFC, end-to-end latency and inter-chunk latency p50/p95/p99;
- RTF and playback underrun/continuity;
- GPU memory before load, after warmup and at peak load;
- active requests, waiting requests and observed batch sizes when metrics exist.

First playable payload is not automatically audible speech; report the two separately instead of calling both TTFA.

## 5. Controls

- Same model and resolved revision.
- Same SGLang-Omni commit and container.
- Same voice, language, request corpus and output-token cap.
- Same streaming flags and codec chunk ramp.
- Same GPU power/clock policy and no competing GPU process during timed runs.
- Record startup flags such as memory fractions, CUDA Graph buckets, max running requests and max queued requests.
- Warm the server before measurement; save every raw JSON artifact.

## 6. Blocking input before launch

Record the actual experiment GPU model, count and VRAM. The model decision is final for a datacenter GPU with adequate headroom; the serving profile and concurrency ceiling cannot be selected safely without this information.
