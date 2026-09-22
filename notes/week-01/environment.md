# Week 1 实验环境

- 日期：
- 代码仓库：
- Commit：`442e559b40b5040965ec876650b32da05d31769f`
- 模型与 revision：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`；候选 revision `0c0e3051f131929182e2c023b9537f8b1c68adfe`，下载后记录 resolved commit
- 操作系统：
- Python：
- PyTorch：`2.13.0`（项目 pin，待实验环境验证）
- CUDA：
- GPU 与数量：
- Driver：
- 容器镜像：
- dtype：BF16（计划值，待实验环境验证）
- 关键启动参数：

## 启动命令

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000
```
## 成功请求

```bash
# 待填写
```

## 错误请求

```bash
# 待填写
```
