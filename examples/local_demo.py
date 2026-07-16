"""
Agentic Vectorless RAG with PageIndex SDK - Local Demo

A simple example of using LocalClient for self-hosted document indexing
and agent-based QA. The agent uses OpenAI Agents SDK to reason over
the document's tree structure index.

Steps:
  1 — Download and index a PDF
  2 — Stream a question with tool call visibility

Requirements:
    pip install pageindex
    export OPENAI_API_KEY=your-api-key   # or any LiteLLM-supported provider
"""
import asyncio
from pathlib import Path
import requests
from pageindex import LocalClient

_EXAMPLES_DIR = Path(__file__).parent
PDF_URL = "https://arxiv.org/pdf/1706.03762.pdf"
PDF_PATH = _EXAMPLES_DIR / "documents" / "attention.pdf"
WORKSPACE = _EXAMPLES_DIR / "workspace"
MODEL = "gpt-4o-2024-11-20"  # any LiteLLM-supported model

# Download PDF if needed
if not PDF_PATH.exists():
    print(f"Downloading {PDF_URL} ...")
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(PDF_URL, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(PDF_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("Download complete.\n")

client = LocalClient(model=MODEL, storage_path=str(WORKSPACE))
col = client.collection()

doc_id = col.add(str(PDF_PATH))
print(f"Indexed: {doc_id}\n")

# Streaming query
stream = col.query(
    "What is the main architecture proposed in this paper and how does self-attention work?",
    stream=True,
)

async def main():
    streamed_text = False
    async for event in stream:
        if event.type == "text_delta":
            print(event.data, end="", flush=True)
            streamed_text = True
        elif event.type == "tool_call":
            if streamed_text:
                print()
                streamed_text = False
            print(f"[tool call] {event.data['name']}")
        elif event.type == "tool_result":
            preview = str(event.data)[:200] + "..." if len(str(event.data)) > 200 else event.data
            print(f"[tool output] {preview}")
        elif event.type == "text_done":
            print()
            streamed_text = False

asyncio.run(main())
