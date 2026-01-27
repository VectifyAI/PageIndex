# Self Hosting (Open-Source)

This repository focuses on **generating PageIndex tree structures** locally (PDF → tree JSON). It does **not** ship the same hosted SaaS/API backend as `api.pageindex.ai`.

If you want to “self host” for internal use, the typical approach is:

1) run the CLI (`run_pageindex.py`) in your own environment, or  
2) wrap the CLI/library with a small HTTP server.

## Minimal FastAPI Server

An example server is provided at `examples/self_host_fastapi.py`. It exposes a single endpoint that accepts a PDF upload and returns the generated tree JSON.

### Install

```bash
pip3 install -r requirements.txt fastapi uvicorn
```

### Configure

Create a `.env` in the repo root with your LLM settings (example):

```bash
CHATGPT_API_KEY=
CHATGPT_MODEL=gpt-4o-2024-11-20
CHATGPT_BASE_URL=https://api.openai.com/v1
```

### Run

```bash
uvicorn examples.self_host_fastapi:app --reload --port 8000
```

### Call

```bash
curl -F "file=@/path/to/document.pdf" "http://localhost:8000/index"
```

Addresses #82
