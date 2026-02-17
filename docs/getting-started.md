# PageIndex Getting Started

This guide helps you set up PageIndex from scratch and run your first document.

## What PageIndex does

PageIndex builds a hierarchical tree structure from long documents and uses LLM reasoning for retrieval. The workflow is vectorless (no vector DB and no chunking-based retrieval).

## Prerequisites

- Python 3.8+
- macOS/Linux/Windows terminal
- One API key for an OpenAI-compatible provider

## 1) Install dependencies

### Option A: UV (recommended)

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv

cd /Users/huidongdezhizhen/Desktop/PageIndex
./setup_uv.sh
source .venv/bin/activate
```

### Option B: pip

```bash
cd /Users/huidongdezhizhen/Desktop/PageIndex
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

## 2) Configure model provider

Edit `.env`:

```bash
CHATGPT_API_KEY=your_api_key
OPENAI_API_BASE=https://provider-endpoint/v1
OPENAI_MODEL=provider-model-id
```

Example (Qwen):

```bash
CHATGPT_API_KEY=sk-your-qwen-key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen-max
```

To switch providers quickly:

```bash
./switch_model.sh
```

## 3) Run PageIndex

### Process PDF

```bash
python run_pageindex.py --pdf_path /path/to/document.pdf
```

### Process Markdown

```bash
python run_pageindex.py --md_path /path/to/document.md
```

### Check outputs

```bash
ls -la ./results/
```

Generated file pattern:

- `{input_name}_structure.json`

## Common options

```bash
python run_pageindex.py \
  --pdf_path /path/to/document.pdf \
  --model gpt-4o \
  --toc-check-pages 20 \
  --max-pages-per-node 10 \
  --max-tokens-per-node 20000 \
  --if-add-node-id yes \
  --if-add-node-summary yes \
  --if-add-doc-description no \
  --if-add-node-text no
```

Markdown-specific options:

```bash
python run_pageindex.py \
  --md_path /path/to/document.md \
  --if-thinning yes \
  --thinning-threshold 5000 \
  --summary-token-threshold 200
```

## Example output structure

```json
{
  "title": "Financial Stability",
  "node_id": "0006",
  "start_index": 21,
  "end_index": 22,
  "summary": "...",
  "nodes": [
    {
      "title": "Monitoring Financial Vulnerabilities",
      "node_id": "0007",
      "start_index": 22,
      "end_index": 28,
      "summary": "..."
    }
  ]
}
```

## Troubleshooting

### API key not found

- Ensure `.env` exists in repo root.
- Ensure `CHATGPT_API_KEY` is set.
- Re-activate environment and retry.

### Dependency install failure

```bash
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

### PDF parse issues

- Test with a smaller clean PDF first.
- Verify file path and file integrity.
- For scanned PDFs, use OCR-first workflows.

### Token limit exceeded

Lower node size:

```bash
python run_pageindex.py \
  --pdf_path file.pdf \
  --max-tokens-per-node 15000 \
  --max-pages-per-node 5
```

## Next docs

- [Quick Reference](quick-reference.md)
- [Multi-Model Configuration](multi-model-configuration.md)
- [Global and Custom Models](global-and-custom-models.md)
- [Qwen Configuration](qwen-configuration.md)
- [UV Quick Reference](uv-quick-reference.md)
