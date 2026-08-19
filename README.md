<div align="center">
  
<a href="https://vectify.ai/pageindex" target="_blank">
  <img src="https://github.com/user-attachments/assets/46201e72-675b-43bc-bfbd-081cc6b65a1d" alt="PageIndex Banner" />
</a>

<br/>
<br/>

<p align="center">
  <a href="https://trendshift.io/repositories/14736" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14736" alt="VectifyAI%2FPageIndex | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# PageIndex: Vectorless, Reasoning-based RAG

<p align="center"><b>Reasoning-based RAG&nbsp; ◦ &nbsp;No Vector DB, No Chunking&nbsp; ◦ &nbsp;Context-Aware Retrieval&nbsp; ◦ &nbsp;Reads Like a Human</b></p>

<h4 align="center">
  <a href="https://vectify.ai">🌐 Website</a>&nbsp; • &nbsp;
  <a href="https://chat.pageindex.ai">🖥️ Chat Platform</a>&nbsp; • &nbsp;
  <a href="https://pageindex.ai/developer">🔌 MCP & API</a>&nbsp; • &nbsp;
  <a href="https://docs.pageindex.ai">📖 Docs</a>&nbsp; • &nbsp;
  <a href="https://discord.com/invite/VuXuf29EUj">💬 Discord</a>&nbsp; • &nbsp;
  <a href="https://ii2abc2jejf.typeform.com/to/tK3AXl8T">✉️ Contact</a>&nbsp;
</h4>
  
</div>

```bash
pip install pageindex
export OPENAI_API_KEY=...        # your own LLM key — PageIndex runs on it
```

```python
from pageindex import PageIndexClient

client = PageIndexClient()                                  # local mode
doc_id = client.submit_document("report.pdf")["doc_id"]     # build the tree index
print(client.chat("What was the 2023 operating margin?", doc_id=doc_id))
```

No vector DB, no chunking, no embedding model — the whole thing runs on your machine, calling your own LLM provider. [Jump to the quickstart ↓](#-quickstart)

<details open>
<summary><h2>📢 Updates</h2></summary>

- 🔥 [**PageIndex SDK**](#-quickstart) — `pip install pageindex` now ships **local mode**: index, retrieve, and chat entirely on your machine with your own LLM key, or point the same client at PageIndex Cloud with an API key.
- ⚡ [**PageIndex Flash**](#-step-1-build-the-tree-index) — tree structure generation from PDFs in seconds, with structure extracted heuristically instead of by an LLM.
- [**Agentic Vectorless RAG**](examples/agentic_vectorless_rag_demo.py) — a simple agentic, vectorless RAG example with *self-hosted PageIndex*, using OpenAI Agents SDK.
- [**Scale PageIndex to Millions of Documents**](https://pageindex.ai/blog/pageindex-filesystem) — *PageIndex File System* is a file-level tree indexing layer that lets PageIndex reason over an entire corpus, not just a single document.
- [PageIndex Chat](https://chat.pageindex.ai) — Human-like document analysis agent [platform](https://chat.pageindex.ai) for professional long documents. Also available via [MCP](https://pageindex.ai/developer) or [API](https://pageindex.ai/developer).
- [PageIndex Framework](https://pageindex.ai/blog/pageindex-intro) — Deep dive into PageIndex: an *agentic, in-context tree index* that enables LLMs to perform *reasoning-based, context-aware retrieval* over long documents.

</details>



# 📑 What is PageIndex?

Are you frustrated with vector database retrieval accuracy for long professional documents? Vector-based RAG retrieves by semantic **similarity**. But **similarity ≠ relevance** — what retrieval actually needs is relevance, and relevance requires **reasoning**. On professional documents that demand contextual understanding, domain expertise, and multi-step reasoning, similarity search misses what is relevant but not similar, and returns what is similar but not relevant.

Inspired by AlphaGo, **[PageIndex](https://vectify.ai/pageindex)** replaces the vector index with a **hierarchical tree index** and lets an LLM **reason** its way through it — the way a human expert flips to the right section of a long report. Retrieval happens in two steps:

1. **Index** — turn the document into a “table-of-contents” **tree structure**
2. **Retrieve** — **search that tree** with LLM reasoning, agentically

<div align="center">
  <a href="https://pageindex.ai/blog/pageindex-intro" target="_blank" title="The PageIndex Framework">
    <img src="https://docs.pageindex.ai/images/cookbook/vectorless-rag.png" width="70%">
  </a>
</div>


### 🎯 Why it works

> PageIndex is a vectorless, reasoning-based RAG engine that mirrors how humans read, delivering traceable, explainable, and context-aware retrieval, without vector databases or chunking.

| | Vector RAG | **PageIndex** |
|---|---|---|
| **Index** | embeddings in a vector DB | tree structure of the document itself |
| **Unit** | fixed-size chunks | natural sections |
| **Retrieval** | approximate similarity search | LLM reasoning over the tree |
| **Result** | opaque, “vibe retrieval” | traceable to explicit sections and page numbers |
| **Context** | query embedding only | full context: conversation history, domain knowledge |

It is ideal for financial reports, legal documents, regulatory filings, technical manuals, medical literature, academic textbooks — any long, complex professional document.

PageIndex achieved **state-of-the-art** [98.7% accuracy](https://github.com/VectifyAI/Mafin2.5-FinanceBench) on FinanceBench (financial document QA benchmark), vastly outperforming vector-based RAG — see [Benchmarks](#-benchmarks).



# ⚡ Quickstart

### 1. Install

```bash
pip install pageindex
```

Local mode and the agent surfaces below require `pageindex >= 0.2.10`. Python 3.10+.

### 2. Set your LLM key

Local mode needs **your own** LLM provider key — PageIndex calls it both to index (node summaries) and to answer. Export it, or put it in a `.env` file next to your script:

```bash
export OPENAI_API_KEY=your_openai_key_here
```

```bash
# .env
OPENAI_API_KEY=your_openai_key_here
```

Any provider [LiteLLM](https://docs.litellm.ai/docs/providers) supports works — set that provider's key (`ANTHROPIC_API_KEY`, …) and pass a provider-prefixed model name (`anthropic/claude-sonnet-4-6`, `bedrock/…`). For a self-hosted OpenAI-compatible server (vLLM, TGI, Ollama), point `OPENAI_BASE_URL` at it and set `OPENAI_API_KEY` to whatever token it expects.

> This is *not* a PageIndex key — local mode never talks to our servers. A PageIndex [API key](https://dash.pageindex.ai/api-keys) is only for [cloud mode](#or-use-pageindex-cloud), where the managed LLM is included.

### 3. Index a document, then ask

```python
from pageindex import PageIndexClient

client = PageIndexClient(                     # local mode — uses OPENAI_API_KEY
    index_model="gpt-5.6-luna",               # builds the tree: a cheap model is enough
    chat_model="gpt-5.6-sol",                 # answers questions: this is what drives accuracy
)
doc_id = client.submit_document("report.pdf")["doc_id"]

answer = client.chat("What was the 2023 operating margin, and where is it stated?",
                     doc_id=doc_id)
print(answer)
```

Those two are also the defaults, so `PageIndexClient()` on its own gives you the same thing — but the two roles are worth naming, because the rule of thumb is **spend on the chat model, not the index model**:

- **`index_model` — a weak model is fine.** Flash reads the hierarchy out of the PDF's own layout, so the model is not inventing the structure; it only writes the node summaries. Dropping to a cheaper model barely changes the tree, and it is a one-off cost per document either way.
- **`chat_model` — use the best you can afford.** This is the model that reasons over the tree, decides what to open, and reads the pages. It is where accuracy actually comes from, and it is billed per question (see [Benchmarks](#-benchmarks)).

What just happened:

- `submit_document` parsed the PDF and built its **tree index**, stored under `./.pageindex`. Indexing is synchronous and one-off — later questions reuse the same `doc_id`.
- `chat` ran a **document-QA agent** over that tree: it read the table of contents, reasoned about which sections could hold the answer, opened only those pages, and answered from them.
- The answer is grounded in real sections and page numbers, so you can check it.

### Or use PageIndex Cloud

Same client, same methods — pass an [API key](https://dash.pageindex.ai/api-keys) and the work happens on our servers, with the production OCR, tree-building, and retrieval pipeline behind it:

```python
client = PageIndexClient(api_key="pi-...")
doc_id = client.submit_document("report.pdf", wait=True)["doc_id"]
print(client.chat("What was the 2023 operating margin?", doc_id=doc_id))
```


# 🛠️ Using PageIndex

## 🌲 Step 1: Build the tree index

`submit_document` defaults to **Flash** indexing: the structure is extracted from the PDF's own layout (no LLM), and a model is called only for node summaries and the tree-optimization expansion pass. It takes seconds.

```python
doc_id = client.submit_document("report.pdf")["doc_id"]
```

Inspect what you got:

```python
tree = client.get_document_structure(doc_id)    # titles, page ranges, summaries — no text
client.list_documents()                         # everything you have indexed
```

A PageIndex tree looks like this — a table of contents optimized for LLMs and agents:

```jsonc
{
  "title": "Financial Stability",
  "node_id": "0006",
  "start_index": 21,
  "end_index": 22,
  "summary": "The Federal Reserve ...",
  "nodes": [
    {
      "title": "Monitoring Financial Vulnerabilities",
      "node_id": "0007",
      "start_index": 22,
      "end_index": 28,
      "summary": "The Federal Reserve's monitoring ..."
    },
    {
      "title": "Domestic and International Cooperation and Coordination",
      "node_id": "0008",
      "start_index": 28,
      "end_index": 31,
      "summary": "In 2023, the Federal Reserve collaborated ..."
    }
  ]
}
```

See more example [documents](https://github.com/VectifyAI/PageIndex/tree/main/examples/documents) and generated [tree structures](https://github.com/VectifyAI/PageIndex/tree/main/examples/documents/results).

<details>
<summary><b>Naming models, and where documents are stored</b></summary>
<br>

```python
client = PageIndexClient(
    index_model="gpt-5.6-luna",              # tree: node summaries + expansion pass
    chat_model="anthropic/claude-sonnet-4-6",# answering, on another provider
    storage_path=".pageindex",               # where indexed documents live
)
```

Model names mean what [LiteLLM](https://docs.litellm.ai/docs/providers) says they mean: a **bare name** goes to an OpenAI-compatible backend (selected by `OPENAI_API_KEY` / `OPENAI_BASE_URL`), and a **`provider/model`** name reaches that provider directly with its own key. Write `openai/Qwen/...` for a self-hosted server that itself serves slashed ids.

The two roles can sit on different providers. `model="..."` sets both at once, and a per-call `model=` on the chat surfaces overrides `chat_model` for that question.

</details>

<details>
<summary><b>Just the tree, without the client</b></summary>
<br>

```python
from pageindex import page_index_flash, page_index

result = page_index_flash("report.pdf")        # heuristic structure + LLM summaries
result = page_index("report.pdf")              # full LLM-built tree
```

Or from the command line, in this repo:

```bash
python3 run_pageindex.py --pdf_path /path/to/your/document.pdf
python3 run_pageindex.py --md_path  /path/to/your/document.md   # markdown, by "#" heading level
```

<details>
<summary>Optional CLI parameters</summary>
<br>

The structure-tuning flags from `--toc-check-pages` down require `--mode standard`; `--optimize` requires Flash mode.

```
--mode                  Processing mode: flash (default) or standard
--index-model           LLM model used to index the document (default: gpt-5.6-luna)
--optimize              Tree optimization for retrieval: full (default), merge, or off
--toc-check-pages       Pages to check for table of contents (default: 20)
--max-pages-per-node    Max pages per node (default: 10)
--max-tokens-per-node   Max tokens per node (default: 20000)
--if-add-node-id        Add node ID (yes/no, default: yes)
--if-add-node-summary   Add node summary (yes/no, default: yes)
--if-add-doc-description Add doc description (yes/no, default: yes)
```
</details>

> **Markdown mode note:** headings are read from `#` levels, so the file must be correctly formatted. If your Markdown was converted from PDF or HTML, most converters destroy the hierarchy — use [PageIndex OCR](https://pageindex.ai/blog/ocr), which is built to preserve it.

</details>

## 💬 Step 2: Ask questions

`chat()` is the one-line surface. Underneath it is a document-QA agent, and you can talk to it over whichever protocol your stack already speaks:

```python
client.chat("What changed in the risk factors?", doc_id=doc_id)            # → answer string
client.chat(question, doc_id=doc_id, stream=True)                          # → text chunks
client.chat_completions(messages, doc_id=doc_id)                           # OpenAI Chat Completions envelope
client.responses("...", doc_id=doc_id, reasoning={"effort": "high"})       # OpenAI Responses (agentic)
client.messages("...", model="claude-sonnet-4-6", doc_id=doc_id)           # Anthropic Messages
```

- **`chat`** — sugar over `chat_completions`: pass a string or a role/content history, get the answer back.
- **`chat_completions`** — the same engine with the full envelope: token usage, streaming metadata, `finish_reason`.
- **`responses`** — the agentic surface. The whole process transcript (including tool outputs) rides in `items`; append it to your next call's `input` to keep the agent's memory and the provider's prompt cache warm. Local mode, Responses-capable backends.
- **`messages`** — Claude-native, via the Anthropic SDK's tool runner (`pip install 'pageindex[anthropic]'`). Local mode.

Pass a list of ids to `doc_id` to search several documents at once, and keep it identical across a conversation's calls.

## 🤖 Step 3: Put PageIndex inside your own agent

Instead of calling PageIndex's agent, hand PageIndex's tools to yours. One call fills every slot:

```python
# OpenAI Agents SDK
from agents import Agent, Runner
agent = Agent(**client.openai_agent_config(doc_id=doc_id))
result = Runner.run_sync(agent, "Summarize the auditor's concerns.")

# Anthropic SDK tool runner            (pip install 'pageindex[anthropic]')
runner = anthropic_client.beta.messages.tool_runner(
    **client.anthropic_runner_config(model="claude-sonnet-4-6", doc_id=doc_id),
    messages=[{"role": "user", "content": "Summarize the auditor's concerns."}],
)

# Claude Agent SDK                     (pip install 'pageindex[claude]')
options = ClaudeAgentOptions(**client.claude_agent_config(doc_id=doc_id))

# Anything else — plain Python functions for LangChain, PydanticAI, ...
tools = client.agent_tools()
```

Each `*_config` helper is sugar over the explicit pieces — `client.agent_instructions()` for the system prompt and `client.as_openai_tools()` / `as_anthropic_tools()` / `as_claude_mcp()` for the tools — so you can swap in your own prompt whenever you need to. Locally, `doc_id` is enforced at the tool layer, not just prompted: out-of-scope lookups return `NOT_FOUND`.


```bash
python3 examples/agentic_vectorless_rag_demo.py
```

## ☁️ Local or Cloud

The same `PageIndexClient` runs both. Omit `api_key` for local, pass one for cloud.

| | **Local** (this repo) | **Cloud** ([API key](https://dash.pageindex.ai/api-keys)) |
|---|---|---|
| Parsing | standard PDF text extraction | [PageIndex OCR](https://pageindex.ai/blog/ocr) — built to preserve document hierarchy |
| Scanned / image-only PDFs | not supported | supported |
| LLM | yours — bring a provider key (`OPENAI_API_KEY`, …) and pay that provider | managed, included with your PageIndex key |
| Storage | `./.pageindex` on disk | hosted library, folders, search |
| Where data goes | never leaves your machine | PageIndex Cloud |
| Image retrieval & understanding | not supported — text layer only | supported |
| Citations & references | page-level | line-level |
| Extras | — | folders, hosted search, MCP server |

`PageIndexCloudClient` / `PageIndexLocalClient` pin the mode explicitly if you would rather not infer it from `api_key`.

The cloud service is also available as a ChatGPT-style [chat platform](https://chat.pageindex.ai), or via [MCP](https://pageindex.ai/developer) and [API](https://pageindex.ai/developer). For dedicated or private deployment (VPC, on-prem), [contact us](https://ii2abc2jejf.typeform.com/to/gVv7qkaN) or [book a demo](https://calendly.com/pageindex/meet).



# 📊 Benchmarks

### What indexing costs

Building a tree locally runs **about $0.001 per page** with `index_model="gpt-5.6-luna"` — so a 1,000-page textbook costs a little over a dollar and a few minutes, once, and every later question reuses it. Measured over nine PDFs, from a 9-page whitepaper to a 1,098-page textbook, 2,800 pages in total.

**The index model is not the bottleneck.** PageIndex is designed not to rely heavily on the model used at index time, so in our experiments a cheap model does not hurt quality.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/index-cost-dark.png">
  <img src="assets/index-cost-light.png" alt="Indexing cost against document length, log-log, for nine PDFs from 9 to 1,098 pages. Points track a $0.0011-per-page reference line; the spread around it is text density, not length.">
</picture>



### Open-source PageIndex, running locally

[**PageIndex-OSS-Benchmark**](https://github.com/VectifyAI/PageIndex-OSS-Benchmark) measures exactly the setup in the quickstart above — `PageIndexClient()` in local mode, flash indexing, no OCR — on 62 lookup questions over 34 PDFs (1,945 pages) drawn from [MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2). Every question's answer is a fact stated in running text, so a wrong answer is a **retrieval or reading failure**, not a reasoning one.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-dark.png">
  <img src="assets/results-light.png" alt="Accuracy against average cost per question. Each model forms a near-vertical reasoning-effort ladder; moving between models costs an order of magnitude a step.">
</picture>


Full results, data, and the runner are in the [benchmark repo](https://github.com/VectifyAI/PageIndex-OSS-Benchmark).

### PageIndex leads a finance QA benchmark

Mafin 2.5, a reasoning-based RAG system for financial document analysis powered by PageIndex, reached a state-of-the-art [**98.7% accuracy**](https://vectify.ai/blog/Mafin2.5) on [FinanceBench](https://arxiv.org/abs/2311.11944), far ahead of vector-based RAG systems on SEC filings and earnings disclosures.

<div align="center">
  <a href="https://github.com/VectifyAI/Mafin2.5-FinanceBench">
    <img src="https://github.com/user-attachments/assets/571aa074-d803-43c7-80c4-a04254b782a3" width="70%">
  </a>
</div>

Explore the full [benchmark results](https://github.com/VectifyAI/Mafin2.5-FinanceBench) and the [blog post](https://vectify.ai/blog/Mafin2.5).


# 🧭 Resources

* 📝 [Blog](https://pageindex.ai/blog): technical articles, research insights, and product updates.
* 🔧 [Developer](https://pageindex.ai/developer): MCP setup, API docs, and integration guides.
* 🧪 [Cookbooks](https://docs.pageindex.ai/cookbook): hands-on, runnable examples and advanced use cases — try the [Vectorless RAG](https://colab.research.google.com/github/VectifyAI/PageIndex/blob/main/cookbook/pageindex_RAG_simple.ipynb) and the OCR-free, vision-based [Vision RAG](https://colab.research.google.com/github/VectifyAI/PageIndex/blob/main/cookbook/vision_RAG_pageindex.ipynb) notebooks in Colab.
* 📖 [Tutorials](https://docs.pageindex.ai/tutorials): practical guides and strategies, including *Document Search* and *Tree Search*.

---

# ⭐ Support Us

Leave us a star 🌟 if you like our project. Thank you!  

<p>
  <img src="https://github.com/user-attachments/assets/eae4ff38-48ae-4a7c-b19f-eab81201d794" width="80%">
</p>

Please cite this work as:
```
Mingtian Zhang, Yu Tang and PageIndex Team,
"PageIndex: Next-Generation Vectorless, Reasoning-based RAG",
PageIndex Blog, Sep 2025.
```

<details>
<summary>Or use the BibTeX citation.</summary>

```bibtex
@article{zhang2025pageindex,
  author = {Mingtian Zhang and Yu Tang and PageIndex Team},
  title = {PageIndex: Next-Generation Vectorless, Reasoning-based RAG},
  journal = {PageIndex Blog},
  year = {2025},
  month = {September},
  note = {https://pageindex.ai/blog/pageindex-intro},
}
```
</details>


### Connect with Us

<div align="center">

[![Website](https://img.shields.io/badge/Website-2D72CF?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDEgMSAxMWgyLjV2MTJoNnYtN2g1djdoNlYxMUgyM3oiLz48L3N2Zz4%3D)](https://pageindex.ai)&nbsp;
[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/PageIndexAI)&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTIwLjQ1IDIwLjQ1aC0zLjU1di01LjU3YzAtMS4zMy0uMDMtMy4wNC0xLjg1LTMuMDQtMS44NSAwLTIuMTQgMS40NS0yLjE0IDIuOTR2NS42N0g5LjM1VjloMy40MXYxLjU2aC4wNWMuNDgtLjkgMS42NC0xLjg1IDMuMzctMS44NSAzLjYgMCA0LjI3IDIuMzcgNC4yNyA1LjQ2djYuMjh6TTUuMzQgNy40M2EyLjA2IDIuMDYgMCAxIDEgMC00LjEzIDIuMDYgMi4wNiAwIDAgMSAwIDQuMTN6TTcuMTIgMjAuNDVIMy41NlY5aDMuNTZ2MTEuNDV6TTIyLjIyIDBIMS43N0MuNzkgMCAwIC43NyAwIDEuNzN2MjAuNTRDMCAyMy4yMy43OSAyNCAxLjc3IDI0aDIwLjQ1QzIzLjIgMjQgMjQgMjMuMjMgMjQgMjIuMjdWMS43M0MyNCAuNzcgMjMuMiAwIDIyLjIyIDB6Ii8%2BPC9zdmc%2B)](https://www.linkedin.com/company/vectify-ai/)&nbsp;
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/invite/VuXuf29EUj)&nbsp;
[![Book a Demo](https://img.shields.io/badge/Book_a_Demo-6E7E96?style=for-the-badge&logo=googlecalendar&logoColor=white)](https://calendly.com/pageindex/meet)&nbsp;
[![Contact Us](https://img.shields.io/badge/Contact_Us-3B82F6?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjIgNCAyMCAxNiI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTIwIDRINGMtMS4xIDAtMiAuOS0yIDJ2MTJjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0wIDQtOCA1LTgtNVY2bDggNSA4LTV6Ii8%2BPC9zdmc%2B)](https://ii2abc2jejf.typeform.com/to/tK3AXl8T)

</div>

---

© 2026 [Vectify AI](https://vectify.ai)
