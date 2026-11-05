# Local document research with web search

Combine PageIndex's local document tools with current public web sources in an
[OpenAI Agents SDK](https://docs.pageindex.ai/sdk/agents) agent. The example uses
[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp) over
Streamable HTTP for free web search and page fetching, with no Parallel account or
API key. Anonymous searches use Fast mode and have lower rate limits.

## Install

From the repository root, install the checkout in a fresh environment. PageIndex
declares the `openai-agents` and `mcp` dependencies used by this example.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
export OPENAI_API_KEY="your-openai-key"
```

The OpenAI model and local indexing use your model-provider key and may incur
charges. No PageIndex Cloud key is needed.

## Index a document, then ask a question

Create a local library once, or use a library you already indexed with PageIndex:

```python
from pageindex import PageIndexLocalClient

client = PageIndexLocalClient(storage_path="./research-library")
doc_id = client.submit_document("examples/documents/2023-annual-report.pdf")["doc_id"]
print(doc_id)
```

Run from the repository root:

```bash
python examples/parallel_search/agent.py \
  "Summarize the report's main business risks, then find current public context. Cite document pages and web URLs separately." \
  --storage-path ./research-library --doc-id YOUR_DOCUMENT_ID
```

Omit `--doc-id` to let the agent discover documents in the library. Use `--model`
to select an OpenAI model other than the example's `gpt-4.1` default. Document
context guides retrieval; it does not restrict access to other documents in the
chosen library. Document citations retain PageIndex's citation tags; web sources
use URLs.

The agent uses PageIndex's existing read-only tools and discovers `web_search` and
`web_fetch` from the MCP server. The HTTP connection carries a project User-Agent
and no authorization header. It closes when the run finishes or fails. Tool calls
have a 60-second session timeout and the agent is limited to 12 turns.

Only use documents and questions whose public research topics you are comfortable
sending to the web service. The prompt asks the model to keep private document
text out of web calls, but this is not an enforced data boundary. The agent also
sends document evidence to your OpenAI model. SDK tracing is disabled in this
example.
