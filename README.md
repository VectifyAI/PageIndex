<div align="center">
  
<a href="https://vectify.ai/pageindex" target="_blank">
<img width="1471" height="491" alt="pi_github_banner_low" src="https://github.com/user-attachments/assets/bae02956-6c4e-4a0b-adea-257b0be4aaa1" />
</a>

<br/>
<br/>

<p align="center">
  <a href="https://trendshift.io/repositories/14736" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14736" alt="VectifyAI%2FPageIndex | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# PageIndex: Vectorless, Reasoning-based RAG

<p align="center"><b>Reasoning-based RAG&nbsp; ◦ &nbsp;No Vector DB, No Chunking&nbsp; ◦ &nbsp;Context-Aware Retrieval&nbsp; ◦ &nbsp;Reads Like a Human</b></p>

<h4 align="center">
  <a href="https://pageindex.ai/developer">🌐 Website</a>&nbsp; • &nbsp;
  <a href="https://developer.pageindex.ai/">🖥️ Platform</a>&nbsp; • &nbsp;
  <a href="https://docs.pageindex.ai">📖 Docs</a>&nbsp; • &nbsp;
  <a href="https://pageindex.ai/blog">📝 Blog</a>&nbsp; • &nbsp;
  <a href="https://ii2abc2jejf.typeform.com/to/tK3AXl8T">✉️ Contact</a>&nbsp;
</h4>
  
</div>



<details open>
<summary><h2>Updates</h2></summary>

- [Aug '26] 🔥 [**PageIndex SDK**](#quickstart): `pip install -U pageindex` now ships **local mode**: index, retrieve, and chat entirely on your machine with your own LLM key, or point the same client at PageIndex Cloud with an API key.
- [Aug '26] ⚡ [**PageIndex Flash**](https://docs.pageindex.ai/sdk/documents): tree structure generation from PDFs in seconds, with structure extracted heuristically from the document's own layout info instead of built by an LLM.
- [Scale PageIndex to Millions of Documents](https://pageindex.ai/blog/pageindex-filesystem): *PageIndex File System* is a file-level tree indexing layer that lets PageIndex reason over an entire corpus, not just a single document.
- [PageIndex App](https://app.pageindex.ai): a human-like document analysis agent for long professional documents.<!-- Also available via [MCP](https://pageindex.ai/developer) or [API](https://pageindex.ai/developer). -->

</details>



# What is PageIndex?

Are you frustrated with vector database retrieval accuracy for long and complex documents? Vector-based RAG retrieves by semantic **similarity**. But **similarity ≠ relevance** — what retrieval actually needs is relevance, and relevance requires **reasoning**. On professional documents that demand contextual understanding, domain expertise, and multi-step reasoning, similarity search misses what is relevant but not similar, and returns what is similar but not relevant.

Inspired by AlphaGo, **[PageIndex](https://vectify.ai/pageindex)** replaces the vector index with a **hierarchical tree index** and lets an LLM **reason** its way through it, the way a human expert turns to and reads the right section of a long report. Retrieval happens in two steps:

1. **Index**: generate a **tree-structure index** for each document
2. **Retrieve**: agentically **search that tree** with LLM reasoning

<div align="center">
  <a href="https://pageindex.ai/blog/pageindex-intro" target="_blank" title="The PageIndex Framework">
    <img src="https://docs.pageindex.ai/images/cookbook/vectorless-rag.png" width="70%">
  </a>
</div>


### TL;DR

<blockquote>PageIndex is a <b>vectorless</b>, <b>reasoning-based RAG</b> engine that <b>mirrors how humans read</b>, delivering <b>traceable</b>, <b>explainable</b>, and <b>context-aware</b> retrieval, with <b>no vector DBs</b> or <b>chunking</b>.</blockquote>

### Compare with Vector RAG

| | Vector RAG | **PageIndex** |
|---|---|---|
| **Index** | vector index | tree index |
| **Unit** | fixed-size chunks | natural sections |
| **Retrieval** | semantic similarity search | LLM reasoning over the tree |
| **Result** | opaque, “vibe retrieval” | traceable to explicit references |
| **Context** | query embedding only | full context: conversation history, domain knowledge, etc. |

It is ideal for financial reports, legal documents, regulatory filings, technical manuals, medical literature, academic textbooks, and any other long, complex professional document.




# Quickstart

```bash
pip install -U pageindex
```


```python
import os
from pageindex import PageIndexClient

os.environ["OPENAI_API_KEY"] = "your-openai-key"

client = PageIndexClient(
    index="gpt-5.6-luna",               # model to build the tree index
    chat="gpt-5.6-sol",                 # model to search the tree
)
doc_id = client.submit_document("report.pdf")["doc_id"]

answer = client.chat("What was the 2023 operating margin?", doc_id=doc_id)
print(answer)
```

### Model Recommendations

- **`index=`: a basic model is sufficient.** The tree structure itself is extracted from the document layout without an LLM; the index model only summarizes and refines it, which a basic model does well.
- **`chat=`: use the best model you can afford.** The chat model searches the tree to retrieve information. See [Query cost and accuracy](#query-cost-and-accuracy).

### [Use PageIndex through the SDK client →](https://docs.pageindex.ai/sdk/chat)

Configure other models, streaming, multi-document search, citations, and more.

### [Integrate PageIndex with your own agent →](https://docs.pageindex.ai/sdk/agents)

Drop PageIndex tools into the OpenAI Agents SDK, the Claude Agent SDK, or any other framework.


# Benchmarks

### Local indexing cost and time

Building a tree locally runs **about $0.001 per page** with `gpt-5.6-luna` as the index model, so a 1,000-page textbook costs a little over a dollar and a few minutes, once, and every later question reuses it. PageIndex is designed not to rely heavily on the model used at index time, so in our experiments a basic model does not hurt quality.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/index-cost-dark.png">
  <img src="assets/index-cost-light.png" width="75%" alt="Indexing cost against document length, log-log, for nine PDFs from 9 to 1,098 pages. Points track a $0.0011-per-page reference line; the spread around it is text density, not length.">
</picture>
</div>

Indexing time also scales predictably with document length. In the same local setup, the benchmark documents (9 to 1,098 pages) finished in roughly **13 seconds to 4.5 minutes**.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/index-time-dark.png">
  <img src="assets/index-time-light.png" width="75%" alt="Indexing time against document length, log-log, for nine PDFs from 9 to 1,098 pages. The measured indexing times range from about 13 seconds to 4.5 minutes and increase predictably with document length.">
</picture>
</div>



### Query cost and accuracy

[**PageIndex-OSS-Benchmark**](https://github.com/VectifyAI/PageIndex-OSS-Benchmark) measures exactly the setup in the quickstart above (`PageIndexClient()` in local mode, flash indexing, no OCR) on 62 lookup questions over 34 PDFs (1,945 pages) drawn from [MMLongBench-Doc-V2](https://github.com/VectifyAI/MMLongBench-Doc-V2). Every question's answer is a fact stated in running text, so a wrong answer is a **retrieval or reading failure**, not a reasoning one.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-dark.png">
  <img src="assets/results-light.png" width="75%" alt="Accuracy against average cost per question. Each model forms a near-vertical reasoning-effort ladder; moving between models costs an order of magnitude a step.">
</picture>
</div>


Full results, data, and the runner are in the [benchmark repo](https://github.com/VectifyAI/PageIndex-OSS-Benchmark).

### Cost per query vs. native PDF input

The alternative to retrieval is handing the model the whole PDF on every question. That cost grows with the document; PageIndex's does not, because it reads only the nodes its reasoning reaches. On documents where both routes return the same answer, native PDF input costs **2.1× more at 52 pages and 16.6× more at 420** (`gpt-5.6-sol`, prompt caching excluded) — and at 805 pages the document no longer fits in the context window at all.

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/query-cost-dark.png">
  <img src="assets/query-cost-light.png" width="75%" alt="Cost per query relative to PageIndex retrieval, for five PDFs from 52 to 805 pages. Passing the PDF natively costs 2.1x, 3.4x, 7.8x, and 16.6x more at 52, 85, 198, and 420 pages; at 805 pages it exceeds the model's context window.">
</picture>
</div>

### Leading accuracy on FinanceBench

PageIndex reached a state-of-the-art [**98.7% accuracy**](https://vectify.ai/blog/Mafin2.5) on [FinanceBench](https://arxiv.org/abs/2311.11944) (financial document QA benchmark), vastly outperforming vector-based RAG.

<div align="center">
<a href="https://github.com/VectifyAI/Mafin2.5-FinanceBench">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/financebench-dark.png">
  <img src="assets/financebench-light.png" width="70%" alt="FinanceBench accuracy: PageIndex 98.7%, vector RAG 50%.">
</picture>
</a>
</div>

Explore the full FinanceBench [evaluation results](https://github.com/VectifyAI/Mafin2.5-FinanceBench) and the [blog post](https://vectify.ai/blog/Mafin2.5).


# PageIndex Cloud

The open-source version is ideal for text-heavy PDFs and local workflows. With **PageIndex Cloud, document indexing and storage run in the cloud**: PageIndex handles parsing, OCR, image understanding, tree-index construction, and managed storage for you. The chat and retrieval layer remains **compatible with your model**, so you can search the cloud-hosted index using the model provider your application already uses.

Moving indexing and storage from Local to Cloud only requires a [PageIndex API key](https://developer.pageindex.ai/):

```python
import os
from pageindex import PageIndexClient

os.environ["PAGEINDEX_API_KEY"] = "your-pageindex-key"
os.environ["OPENAI_API_KEY"] = "your-openai-key"

client = PageIndexClient(
    index="cloud",                       # build and store the index in PageIndex Cloud
    chat="gpt-5.6-sol",                  # use your preferred compatible model for chat
)


doc_id = client.submit_document("report.pdf", wait=True)["doc_id"]
print(client.chat("What was the 2023 operating margin?", doc_id=doc_id))
```

To submit several documents first and wait for them later, use the public polling helper:

```python
submission = client.submit_document("report.pdf")
document = client.wait_until_completed(
    submission["doc_id"], timeout=300, poll_interval=5
)
```

`poll_interval` is the initial delay between status requests. Repeated requests back off automatically to avoid excessive polling.

| Capability | **Local** (this repo) | **Cloud** ([get an API key](https://developer.pageindex.ai/)) |
|---|---|---|
| Best for | text-heavy PDFs and local workflows | scanned, image-heavy, and large document collections |
| Indexing | runs locally | runs in PageIndex Cloud, with production OCR and image understanding |
| Storage | local | managed in PageIndex Cloud |
| Chat model | your model | your model, or the managed chat included with your key |
| Citations | page-level | line-level |
| Image understanding | — | ✅ |
| Multi-document scale | manual | PageIndex File System |
| MCP server | — | ✅ |

### More About PageIndex Cloud

- [Scale PageIndex to Millions of Documents](https://pageindex.ai/blog/pageindex-filesystem): **PageIndex File System** is a Cloud-only, file-level tree indexing layer that lets PageIndex reason over an entire corpus, not just a single document.

### Ready to Try It?

- Get a [PageIndex API key](https://developer.pageindex.ai/)
- Read the [PageIndex Cloud documentation](https://docs.pageindex.ai/)

For dedicated deployment (VPC or on-premises), [contact us](https://ii2abc2jejf.typeform.com/to/gVv7qkaN) or [book a demo](https://calendly.com/pageindex/meet).



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

© 2026 [PageIndex AI](https://pageindex.ai)
