# Multi-Model Configuration Guide

PageIndex can run with any provider that exposes an OpenAI-compatible API.

## Supported providers

| Provider | Strength | Typical use case |
| --- | --- | --- |
| Qwen (DashScope) | Strong Chinese + long docs | Chinese document analysis |
| DeepSeek | Strong value for price | General processing, coding docs |
| Zhipu GLM | Chinese understanding + long context | Academic and long-form docs |
| Kimi (Moonshot) | Long-context models | Very long documents |
| SiliconFlow | Many open models + good speed | Batch workloads |
| OpenAI | Strong overall quality | High-accuracy English tasks |
| Yi (01.AI) | Strong Chinese models | Chinese enterprise docs |
| MiniMax | Multimodal lineup | Mixed content workflows |

## Standard `.env` pattern

```bash
CHATGPT_API_KEY=your_api_key
OPENAI_API_BASE=https://provider-url/v1
OPENAI_MODEL=provider-model
```

## Ready-to-use presets

### 1) DeepSeek (recommended cost/performance)

```bash
CHATGPT_API_KEY=sk-your-deepseek-api-key
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

Optional model: `deepseek-coder`

### 2) Zhipu GLM

```bash
CHATGPT_API_KEY=your-glm-api-key
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
OPENAI_MODEL=glm-4-plus
```

Alternative model IDs: `glm-4-air`, `glm-4-flash`, `glm-4-long`

### 3) Kimi (Moonshot)

```bash
CHATGPT_API_KEY=sk-your-kimi-api-key
OPENAI_API_BASE=https://api.moonshot.cn/v1
OPENAI_MODEL=moonshot-v1-32k
```

Alternative model IDs: `moonshot-v1-8k`, `moonshot-v1-128k`

### 4) SiliconFlow

```bash
CHATGPT_API_KEY=sk-your-siliconflow-api-key
OPENAI_API_BASE=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen2.5-72B-Instruct
```

### 5) Qwen (DashScope)

```bash
CHATGPT_API_KEY=sk-your-qwen-api-key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-max
```

Alternative model IDs: `qwen-plus`, `qwen-turbo`, `qwen-long`

### 6) OpenAI

```bash
CHATGPT_API_KEY=sk-your-openai-api-key
# keep OPENAI_API_BASE unset for official OpenAI
OPENAI_MODEL=gpt-4o
```

### 7) Yi (01.AI)

```bash
CHATGPT_API_KEY=your-yi-api-key
OPENAI_API_BASE=https://api.lingyiwanwu.com/v1
OPENAI_MODEL=yi-large
```

### 8) MiniMax

```bash
CHATGPT_API_KEY=your-minimax-api-key
OPENAI_API_BASE=https://api.minimax.chat/v1
OPENAI_MODEL=abab6.5-chat
```

## How to switch models

### Method A: use the helper script (recommended)

```bash
./switch_model.sh
```

### Method B: edit `.env`

```bash
nano .env
```

Update all three values:

```bash
CHATGPT_API_KEY=...
OPENAI_API_BASE=...
OPENAI_MODEL=...
```

### Method C: override model in CLI

```bash
python run_pageindex.py --pdf_path file.pdf --model deepseek-chat
```

## Comparison snapshot

| Model family | Chinese | English | Reasoning | Speed | Cost |
| --- | --- | --- | --- | --- | --- |
| DeepSeek | High | High | High | High | Low |
| Qwen Max | Very high | High | High | Medium | Medium |
| GLM-4 | Very high | High | High | Medium | Medium |
| Kimi | High | High | High | Medium | Medium |
| GPT-4o | High | Very high | Very high | High | High |

## Scenario-based recommendations

- Chinese-heavy workflows: `qwen-max`, `glm-4-plus`, `yi-large`
- English analysis: `gpt-4o`, `deepseek-chat`
- Code/technical docs: `deepseek-coder`, `qwen-max`
- Very long context: `moonshot-v1-128k`, `glm-4-long`
- Cost-sensitive batch jobs: `deepseek-chat`, SiliconFlow model IDs

## Test different models quickly

```bash
python test_models.py
```

Or manual compare:

```bash
python run_pageindex.py --pdf_path file.pdf --model qwen-max
python run_pageindex.py --pdf_path file.pdf --model deepseek-chat
python run_pageindex.py --pdf_path file.pdf --model gpt-4o
```

## Troubleshooting checklist

1. Confirm `CHATGPT_API_KEY` is valid.
2. Confirm `OPENAI_API_BASE` points to an OpenAI-compatible endpoint.
3. Confirm model ID exists for that provider.
4. Confirm account quota and rate limits.
5. Review stack traces in terminal output and logs.

## References

- DeepSeek docs: <https://platform.deepseek.com/docs>
- Zhipu GLM docs: <https://open.bigmodel.cn/dev/api>
- Moonshot docs: <https://platform.moonshot.cn/docs>
- SiliconFlow docs: <https://docs.siliconflow.cn/>
- DashScope docs: <https://help.aliyun.com/zh/dashscope/>
- OpenAI docs: <https://platform.openai.com/docs>
