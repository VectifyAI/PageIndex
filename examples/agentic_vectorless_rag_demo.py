"""
Agentic Vectorless RAG with PageIndex - Demo

A simple example of building a document QA agent with self-hosted PageIndex
and the OpenAI Agents SDK. Instead of vector similarity search and chunking,
PageIndex builds a hierarchical tree index and uses agentic LLM reasoning for
human-like, context-aware retrieval.

Agent tools:
  - get_document()           — document metadata (status, page count, etc.)
    - get_document_structure() — tree structure with depth and paginated top-level parts
    - get_children()           — subtree expansion for a selected node
  - get_page_content()       — retrieve text content of specific pages

Steps:
  1 — Index a PDF and view its tree structure index
  2 — View document metadata
  3 — Ask a question (agent reasons over the index and auto-calls tools)

Requirements: pip install openai-agents
"""

import sys
import json
import asyncio
import concurrent.futures
import os
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.model_settings import ModelSettings
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseReasoningSummaryTextDeltaEvent,
)

from pageindex import PageIndexClient
import pageindex.utils as utils

# PDF_URL = "https://library.e.abb.com/public/c82ebba1dc5e4d8eadadb3b199e1953d/41_23-820-EN_A.pdf"

_EXAMPLES_DIR = Path(__file__).parent
PDF_PATH = _EXAMPLES_DIR / "documents" / "hpe_manual.pdf"
WORKSPACE = _EXAMPLES_DIR / "workspace"
STRUCTURE_MODE_TOP_LEVEL = "top_level"
STRUCTURE_MODE_CHILDREN = "children"
STRUCTURE_MODE = STRUCTURE_MODE_CHILDREN
STRUCTURE_TOKEN_LIMIT = 5000

TOP_LEVEL_AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.
TRAVERSAL MODE:
- Use get_document() first to confirm status and page/line count.
- Use get_document_structure(part=...) to inspect the top-level outline one part at a time.
- Each node includes has_children, which tells you whether there is more structure beneath it.
- next_steps shows available actions: request the next part, jump to last part, or fetch page content.
- If no section in the current part matches your query, request the next part to search further.
- Use get_page_content(pages="5-7") with tight ranges; never fetch the whole document.
- Before each tool call, output one short sentence explaining the reason.
Answer based only on tool output. Be concise.
"""

CHILDREN_AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.
TRAVERSAL MODE:
- Use get_document() first to confirm status and page/line count.
- Use get_document_structure() first to inspect the top-level outline.
- Each node includes has_children, which tells you whether there is more structure beneath it.
- Expand only relevant branches with get_children(node_id=...) or get_children(node_path=...).
- Use get_page_content(pages="5-7") with tight ranges; never fetch the whole document.
- Before each tool call, output one short sentence explaining the reason.
Answer based only on tool output. Be concise.
"""


def _normalize_openai_compatible_model(model: str | None) -> str | None:
    """Ensure custom hosted models are routed through an OpenAI-compatible provider."""
    if not model:
        return None
    if model.startswith(("openai/", "litellm/")):
        return model
    if model.startswith(("gemini", "gemini-pro", "gemini-2.5pro")):
        return model
    return f"openai/{model}"


def query_agent(
    client: PageIndexClient, doc_id: str, prompt: str, verbose: bool = False
) -> str:
    """Run a document QA agent using the OpenAI Agents SDK.

    Streams text output token-by-token and returns the full answer string.
    Tool calls are always printed; verbose=True also prints arguments and output previews.
    """

    @function_tool
    def get_document() -> str:
        """Get document metadata: status, page count, name, and description."""
        return client.get_document(doc_id)

    @function_tool
    def get_page_content(pages: str) -> str:
        """
        Get the text content of specific pages or line numbers.
        Use tight ranges: e.g. '5-7' for pages 5 to 7, '3,8' for pages 3 and 8, '12' for page 12.
        For Markdown documents, use line numbers from the structure's line_num field.
        """
        return client.get_page_content(doc_id, pages)

    if STRUCTURE_MODE == STRUCTURE_MODE_TOP_LEVEL:

        @function_tool
        def get_document_structure(part: int = 1) -> str:
            """
            Get the top-level outline page selected by part.
            - part is 1-based (e.g., part=1 is the first page of results).
            - has_children shows whether a node can be expanded.
            - next_steps provides guidance on what to do next.
            """
            return client.get_document_structure(
                doc_id,
                parts=part,
                token_limit=STRUCTURE_TOKEN_LIMIT,
            )

        agent_instructions = TOP_LEVEL_AGENT_SYSTEM_PROMPT
        tools = [get_document, get_document_structure, get_page_content]
    else:

        @function_tool
        def get_document_structure() -> str:
            """
            Get the top-level outline.
            Each node includes has_children for recursive expansion decisions.
            """
            return client.get_document_structure(doc_id, depth=1)

        @function_tool
        def get_children(node_id: str = "", node_path: str = "", depth: int = 1) -> str:
            """
            Expand one subtree for the selected node.
            - Use node_id when available.
            - node_path is a 1-based fallback like '2.3'.
            - depth=1 returns immediate children only.
            """
            return client.get_children(
                doc_id, node_id=node_id, node_path=node_path, depth=depth
            )

        agent_instructions = CHILDREN_AGENT_SYSTEM_PROMPT
        tools = [get_document, get_document_structure, get_children, get_page_content]

    agent = Agent(
        name="PageIndex",
        instructions=agent_instructions,
        tools=tools,
        model=client.retrieve_model,
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

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()
    except RuntimeError:
        return asyncio.run(_run())


if __name__ == "__main__":

    set_tracing_disabled(True)

    # Optional: route requests to an OpenAI-compatible endpoint.
    # Example:
    #   PAGEINDEX_OPENAI_BASE_URL=https://qwen3vl30b.gc.example.com/v1
    #   PAGEINDEX_MODEL=QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ
    openai_base_url = os.getenv("PAGEINDEX_OPENAI_BASE_URL")
    if openai_base_url:
        os.environ["OPENAI_BASE_URL"] = openai_base_url.rstrip("/")

    configured_model = _normalize_openai_compatible_model(os.getenv("PAGEINDEX_MODEL"))

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

    # Setup
    client = PageIndexClient(
        workspace=WORKSPACE,
        model=configured_model,
        retrieve_model=configured_model,
    )

    # Step 1: Index PDF and view tree structure
    print("=" * 60)
    print("Step 1: Index PDF and view tree structure")
    print("=" * 60)
    doc_id = next(
        (
            did
            for did, doc in client.documents.items()
            if doc.get("doc_name") == PDF_PATH.name
        ),
        None,
    )
    if doc_id:
        print(f"\nLoaded cached doc_id: {doc_id}")
    else:
        doc_id = client.index(PDF_PATH)
        print(f"\nIndexed. doc_id: {doc_id}")
    print("\nTree Structure (top-level sections):")
    structure = json.loads(client.get_document_structure(doc_id))
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
    question = "What do i do after i unscrew the filter housing for the compressed air regulator and remove it? What needs to be done before that?"
    # question = "What do i do to properly pack the system?"
    print(f"\nQuestion: '{question}'")
    query_agent(client, doc_id, question, verbose=True)
