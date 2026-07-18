"""
Agentic Vectorless RAG with PageIndex — Demo

Build a document-QA agent with self-hosted PageIndex and the OpenAI Agents SDK.
Instead of vector similarity search and chunking, PageIndex builds a
hierarchical tree index and lets an agent reason over it for human-like,
context-aware retrieval.

This demo wires up your OWN agent + tools against the PageIndex Collection API.
For the batteries-included version, just use ``col.query(..., stream=True)`` —
see local_demo.py.

Agent tools:
  - get_document()           — document metadata (name, type, description)
  - get_document_structure() — the document's tree-structure index
  - get_page_content()       — text of specific pages / line ranges

Steps:
  1 — Index a PDF and view its tree structure
  2 — View document metadata
  3 — Ask a question (agent reasons over the index and auto-calls tools)

Requirements:
    pip install pageindex openai-agents
    export OPENAI_API_KEY=your-api-key   # or any LiteLLM-supported provider
"""
import sys
import json
import asyncio
import concurrent.futures
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.model_settings import ModelSettings
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from openai.types.responses import ResponseTextDeltaEvent, ResponseReasoningSummaryTextDeltaEvent

from pageindex import LocalClient

PDF_URL = "https://arxiv.org/pdf/2603.15031"

_EXAMPLES_DIR = Path(__file__).parent
PDF_PATH = _EXAMPLES_DIR / "documents" / "attention-residuals.pdf"
WORKSPACE = _EXAMPLES_DIR / "workspace"

AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.
TOOL USE:
- Call get_document() first to confirm the document's name and type.
- Call get_document_structure() to identify relevant page ranges.
- Call get_page_content(pages="5-7") with tight ranges; never fetch the whole document.
- Before each tool call, output one short sentence explaining the reason.
Answer based only on tool output. Be concise.
"""


def _normalize_model_for_agents_sdk(model: str) -> str:
    """The OpenAI Agents SDK only recognizes 'openai/' and 'litellm/' model
    prefixes; route any other LiteLLM-style provider path (e.g. 'anthropic/...')
    through litellm explicitly, mirroring what PageIndex itself does internally
    for its built-in agent."""
    if model and "/" in model and not model.startswith(("litellm/", "openai/")):
        return f"litellm/{model}"
    return model


def query_agent(col, doc_id: str, prompt: str, model: str, verbose: bool = False) -> str:
    """Run a document QA agent using the OpenAI Agents SDK.

    Streams text output token-by-token and returns the full answer string.
    Tool calls are always printed; verbose=True also prints arguments and output previews.
    """

    @function_tool
    def get_document() -> str:
        """Get document metadata: name, type, and description."""
        doc = col.get_document(doc_id)
        doc.pop("structure", None)  # keep tool output small for the LLM context
        return json.dumps(doc, ensure_ascii=False)

    @function_tool
    def get_document_structure() -> str:
        """Get the document's full tree structure (without text) to find relevant sections."""
        return json.dumps(col.get_document_structure(doc_id), ensure_ascii=False)

    @function_tool
    def get_page_content(pages: str) -> str:
        """
        Get the text content of specific pages or line numbers.
        Use tight ranges: e.g. '5-7' for pages 5 to 7, '3,8' for pages 3 and 8, '12' for page 12.
        For Markdown documents, use line numbers from the structure's line_num field.
        """
        return json.dumps(col.get_page_content(doc_id, pages), ensure_ascii=False)

    agent = Agent(
        name="PageIndex",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[get_document, get_document_structure, get_page_content],
        model=_normalize_model_for_agents_sdk(model),
        # model_settings=ModelSettings(reasoning={"effort": "low", "summary": "auto"}),  # Uncomment to enable reasoning
    )

    async def _run():
        streamed_run = Runner.run_streamed(agent, prompt)
        current_stream_kind = None
        async for event in streamed_run.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                if isinstance(event.data, ResponseReasoningSummaryTextDeltaEvent):
                    if current_stream_kind != "reasoning":
                        if current_stream_kind is not None:
                            print()
                        print("\n[reasoning]: ", end="", flush=True)
                    delta = event.data.delta
                    print(delta, end="", flush=True)
                    current_stream_kind = "reasoning"
                elif isinstance(event.data, ResponseTextDeltaEvent):
                    if current_stream_kind != "text":
                        if current_stream_kind is not None:
                            print()
                        print("\n[text]: ", end="", flush=True)
                    delta = event.data.delta
                    print(delta, end="", flush=True)
                    current_stream_kind = "text"
            elif isinstance(event, RunItemStreamEvent):
                item = event.item
                if item.type == "tool_call_item":
                    if current_stream_kind is not None:
                        print()
                    raw = item.raw_item
                    args = getattr(raw, "arguments", "{}")
                    args_str = f"({args})" if verbose else ""
                    print(f"\n[tool call]: {raw.name}{args_str}", flush=True)
                    current_stream_kind = None
                elif item.type == "tool_call_output_item" and verbose:
                    if current_stream_kind is not None:
                        print()
                    output = str(item.output)
                    preview = output[:200] + "..." if len(output) > 200 else output
                    print(f"\n[tool call output]: {preview}", flush=True)
                    current_stream_kind = None
        if current_stream_kind is not None:
            print()
        return "" if not streamed_run.final_output else str(streamed_run.final_output)

    # Only the detection is guarded, not the run, so a real error inside _run
    # isn't misread as "no running loop".
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run()).result()


if __name__ == "__main__":

    set_tracing_disabled(True)

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

    # Setup: self-hosted local client + a collection
    client = LocalClient(storage_path=str(WORKSPACE))
    col = client.collection("agentic-demo")

    # Step 1: Index PDF and view tree structure
    print("=" * 60)
    print("Step 1: Index PDF and view tree structure")
    print("=" * 60)
    # Content-hash dedup: re-running reuses the existing doc_id, no re-index.
    doc_id = col.add(str(PDF_PATH))
    print(f"\ndoc_id: {doc_id}")
    print("\nTree Structure (top-level sections):")
    for node in col.get_document_structure(doc_id):
        print(f"  - {node.get('title', '(untitled)')}")

    # Step 2: View document metadata
    print("\n" + "=" * 60)
    print("Step 2: View document metadata")
    print("=" * 60)
    meta = col.get_document(doc_id)
    meta.pop("structure", None)
    print("\n" + json.dumps(meta, ensure_ascii=False, indent=2))

    # Step 3: Agent Query
    print("\n" + "=" * 60)
    print("Step 3: Agent Query (auto tool-use)")
    print("=" * 60)
    question = "Explain Attention Residuals in simple language."
    print(f"\nQuestion: '{question}'")
    query_agent(col, doc_id, question, client.retrieve_model, verbose=True)
