# PageIndex Complete Startup Guide

This is an end-to-end guide for environment setup, model configuration, execution patterns, and operations.

## Contents

1. Environment setup
2. Provider setup
3. First successful run
4. Operational workflows
5. Performance tuning
6. Troubleshooting and FAQ

## 1) Environment setup

### System requirements

- Python 3.9+ recommended
- macOS/Linux/Windows
- A model provider key (or local model server)

### Preferred path: UV

```bash
cd /Users/huidongdezhizhen/Desktop/PageIndex
./setup_uv.sh
source .venv/bin/activate
```

Manual UV flow:

```bash
uv venv .venv
uv pip install -e .
uv lock
source .venv/bin/activate
```

Fallback pip flow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2) Provider setup

PageIndex supports any OpenAI-compatible endpoint.

### Standard `.env` format

```bash
CHATGPT_API_KEY=your_api_key
OPENAI_API_BASE=https://provider-url/v1
OPENAI_MODEL=provider-model-id
```

### Quick switch

```bash
./switch_model.sh
```

### Popular presets

Qwen:

```bash
CHATGPT_API_KEY=sk-your-qwen-key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-max
```

DeepSeek:

```bash
CHATGPT_API_KEY=sk-your-deepseek-key
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
```

OpenAI:

```bash
CHATGPT_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o
# keep OPENAI_API_BASE unset
```

Ollama local:

```bash
CHATGPT_API_KEY=ollama
OPENAI_API_BASE=http://localhost:11434/v1
OPENAI_MODEL=llama3.1:70b
```

## 3) First successful run

### Validate connection

```bash
python test_qwen_api.py
```

### Process a PDF

```bash
python run_pageindex.py --pdf_path tests/pdfs/q1-fy25-earnings.pdf
```

### Check output

```bash
ls -lh results/
cat results/q1-fy25-earnings_structure.json
```

### Process Markdown

```bash
python run_pageindex.py --md_path tutorials/doc-search/README.md
```

## 4) Operational workflows

### Batch processing

```bash
for file in ./tests/pdfs/*.pdf; do
  echo "Processing: $file"
  python run_pageindex.py --pdf_path "$file"
done
```

### Multi-model comparison

```bash
python run_pageindex.py --pdf_path file.pdf --model qwen-max
python run_pageindex.py --pdf_path file.pdf --model deepseek-chat
python run_pageindex.py --pdf_path file.pdf --model gpt-4o
```

### Team reproducibility with `uv.lock`

```bash
# setup by one team member
uv lock

# setup by others
uv venv .venv
uv sync
```

## 5) Performance tuning

### Useful runtime parameters

```bash
python run_pageindex.py \
  --pdf_path file.pdf \
  --max-pages-per-node 8 \
  --max-tokens-per-node 15000 \
  --if-add-node-summary yes \
  --if-add-doc-description no
```

### Tuning strategy

- Lower `--max-tokens-per-node` to reduce cost.
- Lower `--max-pages-per-node` if extraction is unstable on huge pages.
- Use faster models for bulk indexing; use stronger models for critical documents.

## 6) Troubleshooting

### `command not found: uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
brew install uv
```

### `ModuleNotFoundError: No module named 'pageindex'`

```bash
source .venv/bin/activate
uv pip install -e .
```

### API failures

Check:

1. API key validity
2. endpoint correctness
3. model ID correctness
4. account quota
5. network access to provider

### PDF processing failures

- Validate PDF opens correctly.
- Try smaller sample first.
- For scanned PDFs, use OCR-friendly workflows.

## FAQ

### Does it support Chinese documents?

Yes. Qwen, GLM, and Yi are common choices for Chinese-heavy workloads.

### Can it run fully offline?

Yes, with local serving stacks such as Ollama or vLLM.

### How long does one document take?

Depends on model speed and document size. Small docs are often minutes, large docs can be significantly longer.

## Related docs

- [Getting Started](getting-started.md)
- [Quick Reference](quick-reference.md)
- [Multi-Model Configuration](multi-model-configuration.md)
- [Global and Custom Models](global-and-custom-models.md)
- [UV Quick Reference](uv-quick-reference.md)
- [Qwen Configuration](qwen-configuration.md)
