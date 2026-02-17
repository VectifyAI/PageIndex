# Qwen Configuration Guide

## Goal

Configure PageIndex to use Qwen via DashScope's OpenAI-compatible API.

## `.env` configuration

```bash
CHATGPT_API_KEY=sk-your-qwen-key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-max
```

## Default model in config

```yaml
# pageindex/config.yaml
model: "qwen-max"
```

## Supported Qwen model IDs

- `qwen-max`: strongest quality
- `qwen-plus`: balanced quality/cost
- `qwen-turbo`: fastest and cheapest
- `qwen-long`: long-context workloads

## Usage

```bash
source .venv/bin/activate

# use default model from env/config
python run_pageindex.py --pdf_path /path/to/file.pdf

# override model at runtime
python run_pageindex.py --pdf_path /path/to/file.pdf --model qwen-plus
```

## Switch back to OpenAI

```bash
# .env
CHATGPT_API_KEY=sk-your-openai-key
# remove or comment this line when using official OpenAI
# OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=gpt-4o
```

## Connectivity test

```bash
python test_qwen_api.py
```

## Troubleshooting

### Error: invalid API key

- Verify your key in DashScope console
- Confirm key is active and has quota
- Re-run after reloading environment

### Error: model not found

- Use one of: `qwen-max`, `qwen-plus`, `qwen-turbo`, `qwen-long`
- Check for typos and case sensitivity

### Error: endpoint mismatch

- Confirm `OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1`

### Error: env not loaded

```bash
source .venv/bin/activate
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('CHATGPT_API_KEY') is not None)"
```

## Best-practice recommendations

- Use `qwen-max` for difficult document understanding tasks.
- Use `qwen-plus` for routine workloads.
- Use `qwen-turbo` for bulk processing and previews.
- Lower `--max-tokens-per-node` for cost-sensitive jobs.

## Useful links

- Qwen homepage: <https://tongyi.aliyun.com/>
- DashScope docs: <https://help.aliyun.com/zh/dashscope/>
- DashScope API keys: <https://dashscope.console.aliyun.com/apiKey>
