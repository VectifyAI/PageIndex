# Global and Custom Model Configuration

This guide covers non-mainland providers, aggregator platforms, local deployment, and fully custom OpenAI-compatible endpoints.

## Major global providers

### Anthropic Claude

```bash
CHATGPT_API_KEY=sk-ant-your-key
OPENAI_API_BASE=https://api.anthropic.com/v1
OPENAI_MODEL=claude-3-5-sonnet-20241022
```

Common model IDs:
- `claude-4-5-sonnet-20251022`
- `claude-4-opus-20250229`
- `claude-4-sonnet-20250229`
- `claude-4-haiku-20250307`

API key: <https://console.anthropic.com/settings/keys>

### Google Gemini

```bash
CHATGPT_API_KEY=your-gemini-api-key
OPENAI_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai
OPENAI_MODEL=gemini-2.0-flash-exp
```

API key: <https://aistudio.google.com/apikey>

### Mistral

```bash
CHATGPT_API_KEY=your-mistral-api-key
OPENAI_API_BASE=https://api.mistral.ai/v1
OPENAI_MODEL=mistral-large-latest
```

API key: <https://console.mistral.ai/api-keys>

### Cohere

```bash
CHATGPT_API_KEY=your-cohere-api-key
OPENAI_API_BASE=https://api.cohere.ai/v1
OPENAI_MODEL=command-r-plus
```

API key: <https://dashboard.cohere.com/api-keys>

### Perplexity

```bash
CHATGPT_API_KEY=pplx-your-key
OPENAI_API_BASE=https://api.perplexity.ai
OPENAI_MODEL=llama-3.1-sonar-large-128k-online
```

API key: <https://www.perplexity.ai/settings/api>

### Groq (fast inference)

```bash
CHATGPT_API_KEY=gsk_your_groq_key
OPENAI_API_BASE=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.3-70b-versatile
```

API key: <https://console.groq.com/keys>

## Aggregation platforms

### OpenRouter (recommended)

```bash
CHATGPT_API_KEY=sk-or-v1-your-key
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_MODEL=anthropic/claude-3.5-sonnet
```

Why teams use it:
- One API key for many model families
- Easy model A/B comparisons
- Centralized usage and cost metrics

Docs: <https://openrouter.ai/docs>

### Together AI

```bash
CHATGPT_API_KEY=your-together-key
OPENAI_API_BASE=https://api.together.xyz/v1
OPENAI_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
```

### Fireworks AI

```bash
CHATGPT_API_KEY=fw-your-fireworks-key
OPENAI_API_BASE=https://api.fireworks.ai/inference/v1
OPENAI_MODEL=accounts/fireworks/models/llama-v3p1-70b-instruct
```

## Local deployment options

### Ollama (easiest local path)

```bash
# install and start
brew install ollama
ollama serve

# pull model
ollama pull llama3.1:70b

# configure PageIndex
CHATGPT_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_MODEL=llama3.1:70b
```

### LM Studio

```bash
CHATGPT_API_KEY=lm-studio
OPENAI_API_BASE=http://localhost:1234/v1
OPENAI_MODEL=local-model-name
```

### vLLM (production-style serving)

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --port 8000

# PageIndex config
CHATGPT_API_KEY=vllm
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
```

### Text Generation WebUI

```bash
CHATGPT_API_KEY=textgen
OPENAI_API_BASE=http://localhost:5000/v1
OPENAI_MODEL=your-loaded-model
```

## Custom endpoint template

Use this for private hosting, internal gateways, or proxy services.

```bash
CHATGPT_API_KEY=your-custom-api-key
OPENAI_API_BASE=https://your-endpoint.example.com/v1
OPENAI_MODEL=your-model-id
```

## Validation

```bash
python test_qwen_api.py
python run_pageindex.py --pdf_path tests/pdfs/q1-fy25-earnings.pdf
```

## Recommendations

- Highest English quality: Claude or GPT-4o
- Best speed/cost: Groq or DeepSeek
- Highest privacy: local Ollama or vLLM
- Most flexible routing: OpenRouter

## Cautions

1. OpenAI-compatibility is sometimes partial; provider-specific differences may appear.
2. Rate limits vary by provider and plan.
3. Cloud APIs send document content to provider infrastructure.
4. Add usage caps to prevent cost surprises.
