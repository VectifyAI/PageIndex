"""
Agentic Vectorless RAG with PageIndex SDK - Cloud Demo

Uses CloudClient for fully-managed document indexing and QA.
No LLM API key needed — the cloud service handles everything.

Steps:
  1 — Upload and index a PDF via PageIndex cloud
  2 — Stream a question with tool call visibility

Requirements:
    pip install pageindex
    export PAGEINDEX_API_KEY=your-api-key
"""
import asyncio
import os
from pathlib import Path
import requests
from pageindex import CloudClient

_EXAMPLES_DIR = Path(__file__).parent
PDF_URL = "https://arxiv.org/pdf/2603.15031"
PDF_PATH = _EXAMPLES_DIR / "documents" / "attention-residuals.pdf"

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

client = CloudClient(api_key=os.environ["PAGEINDEX_API_KEY"])
collection = client.collection()

doc_id = collection.add(str(PDF_PATH))
print(f"Indexed: {doc_id}\n")

# Streaming query
stream = collection.query("What is the main contribution of this paper?", stream=True)

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
            args = event.data.get("args", "")
            print(f"[tool call] {event.data['name']}({args})")
        elif event.type == "text_done":
            print()
            streamed_text = False

asyncio.run(main())
