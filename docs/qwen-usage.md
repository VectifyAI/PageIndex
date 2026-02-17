# Qwen Setup Complete

Your PageIndex project is ready to run with **Qwen (DashScope OpenAI-compatible mode)**.

## Current setup (example)

- `CHATGPT_API_KEY`: configured
- `OPENAI_API_BASE`: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `OPENAI_MODEL`: `qwen-max`

## Quick start

```bash
# 1) activate environment
source .venv/bin/activate

# 2) test API connection
python test_qwen_api.py

# 3) process a PDF
python run_pageindex.py --pdf_path /path/to/your/file.pdf
```

## Use another Qwen model

```bash
python run_pageindex.py --pdf_path file.pdf --model qwen-plus
python run_pageindex.py --pdf_path file.pdf --model qwen-turbo
python run_pageindex.py --pdf_path file.pdf --model qwen-long
```

## Troubleshooting quick checks

```bash
# confirm env values are loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_BASE')); print(os.getenv('OPENAI_MODEL'))"

# confirm you are inside the virtual env
which python
```

## Related docs

- [Qwen Configuration](qwen-configuration.md)
- [Getting Started](getting-started.md)
- [UV Quick Reference](uv-quick-reference.md)
- [Multi-Model Configuration](multi-model-configuration.md)
