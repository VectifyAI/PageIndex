"""
Agentic Vectorless RAG with PageIndex - Demo

A simple example of building a document QA agent with the PageIndex SDK in
local mode and the OpenAI Agents SDK. Instead of vector similarity search and
chunking, PageIndex builds a hierarchical tree index and uses agentic LLM
reasoning for human-like, context-aware retrieval.

The agent tools come straight from the SDK — ``client.as_openai_tools()``
exposes the PageIndex tool contract (browse_documents, get_document,
get_document_structure, get_page_content) and ``client.agent_instructions()``
provides the retrieval playbook, so the whole agent is a few lines. Swap
``PageIndexLocalClient()`` for ``PageIndexCloudClient(api_key=...)`` and the
same code runs against the cloud.

Steps:
  1 — Index a PDF locally and view its tree structure index
  2 — View document metadata
  3 — Ask a question (agent reasons over the index and auto-calls tools)

Requirements: pip install "pageindex[openai]"; OPENAI_API_KEY in the environment.
"""
import sys
import asyncio
import concurrent.futures
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Runner, set_tracing_disabled
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from openai.types.responses import ResponseTextDeltaEvent, ResponseReasoningSummaryTextDeltaEvent

from pageindex import PageIndexAPIError, PageIndexLocalClient
import pageindex.utils as utils

PDF_URL = "https://arxiv.org/pdf/2603.15031"

_EXAMPLES_DIR = Path(__file__).parent
PDF_PATH = _EXAMPLES_DIR / "documents" / "attention-residuals.pdf"
DOC_ID_PATH = _EXAMPLES_DIR / "documents" / "attention-residuals.doc_id"
STORAGE_PATH = _EXAMPLES_DIR / ".pageindex"


def query_agent(client: PageIndexLocalClient, doc_id: str, prompt: str, verbose: bool = False) -> str:
    """Run a document QA agent using the OpenAI Agents SDK.

    Streams text output token-by-token and returns the full answer string.
    Tool calls are always printed; verbose=True also prints arguments and output previews.
    """
    agent = Agent(
        name="PageIndex",
        instructions=client.agent_instructions(doc_id=doc_id),
        tools=client.as_openai_tools(),
        model=client.retrieve_model,
        # model_settings=ModelSettings(reasoning={"effort": "low", "summary": "auto"}),  # from agents.model_settings import ModelSettings
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

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        return asyncio.run(_run())


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

    # Setup: local mode — no PageIndex API key needed, your LLM key does the work
    client = PageIndexLocalClient(storage_path=str(STORAGE_PATH))

    # Step 1: Index PDF and view tree structure
    print("=" * 60)
    print("Step 1: Index PDF and view tree structure")
    print("=" * 60)
    doc_id = None
    if DOC_ID_PATH.exists():
        cached = DOC_ID_PATH.read_text().strip()
        try:
            client.get_document(cached)
            doc_id = cached
        except PageIndexAPIError:
            DOC_ID_PATH.unlink()
    if doc_id:
        print(f"\nLoaded cached doc_id: {doc_id}")
    else:
        doc_id = client.submit_document(str(PDF_PATH), wait=True)["doc_id"]
        DOC_ID_PATH.write_text(doc_id)
        print(f"\nIndexed. doc_id: {doc_id}")
    print("\nTree Structure (top-level sections):")
    structure = client.get_tree(doc_id, node_summary=True)["result"]
    utils.print_tree(structure)

    # Step 2: View document metadata
    print("\n" + "=" * 60)
    print("Step 2: View document metadata")
    print("=" * 60)
    doc_metadata = client.get_document(doc_id)
    print(f"\n{doc_metadata}")

    # Step 3: Agent Query
    print("\n" + "=" * 60)
    print("Step 3: Agent Query (auto tool-use)")
    print("=" * 60)
    question = "Explain Attention Residuals in simple language."
    print(f"\nQuestion: '{question}'")
    query_agent(client, doc_id, question, verbose=True)
