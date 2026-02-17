# PageIndex Multi-Model Quick Reference

## One-command model switching

```bash
./switch_model.sh
```

## Minimal `.env` template

```bash
CHATGPT_API_KEY=your_api_key
OPENAI_API_BASE=https://provider-endpoint/v1
OPENAI_MODEL=provider-model-name
```

## Recommended provider presets

### Qwen (DashScope)

```bash
CHATGPT_API_KEY=sk-your-qwen-key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-max
```

### DeepSeek

```bash
CHATGPT_API_KEY=sk-your-deepseek-key
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

### Zhipu GLM

```bash
CHATGPT_API_KEY=your-glm-key
OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4
OPENAI_MODEL=glm-4-plus
```

### Kimi (Moonshot)

```bash
CHATGPT_API_KEY=sk-your-kimi-key
OPENAI_API_BASE=https://api.moonshot.cn/v1
OPENAI_MODEL=moonshot-v1-32k
```

### SiliconFlow

```bash
CHATGPT_API_KEY=sk-your-siliconflow-key
OPENAI_API_BASE=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen2.5-72B-Instruct
```

### OpenAI

```bash
CHATGPT_API_KEY=sk-your-openai-key
# leave OPENAI_API_BASE unset for official OpenAI
OPENAI_MODEL=gpt-4o
```

### Anthropic Claude (OpenAI-compatible endpoint)

```bash
CHATGPT_API_KEY=sk-ant-your-key
OPENAI_API_BASE=https://api.anthropic.com/v1
OPENAI_MODEL=claude-3-5-sonnet-20241022
```

### Groq

```bash
CHATGPT_API_KEY=gsk_your_groq_key
OPENAI_API_BASE=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.3-70b-versatile
```

### OpenRouter

```bash
CHATGPT_API_KEY=sk-or-v1-your-key
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_MODEL=anthropic/claude-3.5-sonnet
```

### Ollama (local)

```bash
CHATGPT_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_MODEL=llama3.1:70b
```

## Run PageIndex

```bash
python run_pageindex.py --pdf_path /path/to/file.pdf
python run_pageindex.py --md_path /path/to/file.md
```

## Quick validation

```bash
python test_qwen_api.py
python test_models.py
```

## Documentation index

- [Getting Started](getting-started.md)
- [Complete Startup Guide](pageindex-complete-startup-guide.md)
- [Multi-Model Configuration](multi-model-configuration.md)
- [Global and Custom Models](global-and-custom-models.md)
- [Qwen Configuration](qwen-configuration.md)
- [UV Quick Reference](uv-quick-reference.md)
- [Rollout Summary](multi-model-rollout-summary.md)
