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

<p align="center"><b>Reasoning-based RAG&nbsp; ◦ &nbsp;No Vector DB&nbsp; ◦ &nbsp;No Chunking&nbsp; ◦ &nbsp;Human-like Retrieval</b></p>

<h4 align="center">
  <a href="https://vectify.ai">🏠 Homepage</a>&nbsp; • &nbsp;
  <a href="https://chat.pageindex.ai">🖥️ Chat Platform</a>&nbsp; • &nbsp;
  <a href="https://pageindex.ai/mcp">🔌 MCP</a>&nbsp; • &nbsp;
  <a href="https://docs.pageindex.ai">📚 Docs</a>&nbsp; • &nbsp;
  <a href="https://discord.com/invite/VuXuf29EUj">💬 Discord</a>&nbsp; • &nbsp;
  <a href="https://ii2abc2jejf.typeform.com/to/tK3AXl8T">✉️ Contact</a>&nbsp;
</h4>
  
</div>


<details open>
<summary><h2>🔥 Updates</h2></summary>

- [**An Agentic Vectorless RAG Example**](https://github.com/VectifyAI/PageIndex/blob/main/examples/openai_agents_demo.py): A complete *agentic, vectorless RAG* [example](https://github.com/VectifyAI/PageIndex/blob/main/examples/openai_agents_demo.py) with self-hosted PageIndex, using OpenAI Agents SDK.
- [PageIndex Chat](https://chat.pageindex.ai): Human-like document-analysis agent [platform](https://chat.pageindex.ai) for professional long documents. Also available via [MCP](https://pageindex.ai/mcp) or [API](https://docs.pageindex.ai/quickstart).
- [PageIndex Framework](https://pageindex.ai/blog/pageindex-intro): The PageIndex framework — an *agentic, in-context tree index* that enables LLMs to perform *reasoning-based, human-like retrieval* over long documents.

 <!-- **🧪 Cookbooks:**
- [Vectorless RAG](https://docs.pageindex.ai/cookbook/vectorless-rag-pageindex): A minimal, hands-on example of reasoning-based RAG using PageIndex. No vectors, no chunking, and human-like retrieval.
- [Vision-based Vectorless RAG](https://docs.pageindex.ai/cookbook/vision-rag-pageindex): OCR-free, vision-only RAG with PageIndex's reasoning-native retrieval workflow that works directly over PDF page images. -->

</details>

---

# 📑 Introduction to PageIndex

Are you frustrated with vector database retrieval accuracy for long professional documents? Traditional vector-based RAG relies on semantic *similarity* rather than true *relevance*. But **similarity ≠ relevance** — what we truly need in retrieval is **relevance**, and that requires **reasoning**. When working with professional documents that demand domain expertise and multi-step reasoning, similarity search often falls short.

Inspired by AlphaGo, we propose **[PageIndex](https://vectify.ai/pageindex)** — a **vectorless**, **reasoning-based RAG** system that builds a **hierarchical tree index** from long documents and uses LLMs to **reason** *over that index* for **agentic, context-aware retrieval**.
It simulates how *human experts* navigate and extract knowledge from complex documents through *tree search*, enabling LLMs to *think* and *reason* their way to the most relevant document sections. PageIndex performs retrieval in two steps:

1. Generate a “Table-of-Contents” **tree structure index** of documents
2. Perform reasoning-based retrieval through **tree search**

<div align="center">
  <a href="https://pageindex.ai/blog/pageindex-intro" target="_blank" title="The PageIndex Framework">
    <img src="https://docs.pageindex.ai/images/cookbook/vectorless-rag.png" width="70%">
  </a>
</div>

### 🎯 Core Features

Compared to traditional vector-based RAG, **PageIndex** features:
- **No Vector DB**: Uses document structure and LLM reasoning for retrieval, instead of vector similarity search.
- **No Chunking**: Documents are organized into natural sections, not artificial chunks.
- **Human-like Retrieval**: Simulates how human experts navigate and extract knowledge from complex documents.
- **Better Explainability and Traceability**: Retrieval is based on reasoning — traceable and interpretable, with page and section references. No more opaque, approximate vector search (“vibe retrieval”).

PageIndex powers a reasoning-based RAG system that achieved **state-of-the-art** [98.7% accuracy](https://github.com/VectifyAI/Mafin2.5-FinanceBench) on FinanceBench, demonstrating superior performance over vector-based RAG solutions in professional document analysis (see our [blog post](https://vectify.ai/blog/Mafin2.5) for details).

### 📍 Explore PageIndex

To learn more, please see a detailed introduction of the [PageIndex framework](https://pageindex.ai/blog/pageindex-intro). Check out this GitHub repo for open-source code, and the [cookbooks](https://docs.pageindex.ai/cookbook), [tutorials](https://docs.pageindex.ai/tutorials), and [blog](https://pageindex.ai/blog) for additional usage guides and examples.

The PageIndex service is available as a ChatGPT-style [chat platform](https://chat.pageindex.ai), or can be integrated via [MCP](https://pageindex.ai/mcp) or [API](https://docs.pageindex.ai/quickstart).

### 🛠️ Deployment Options
- Self-host — run locally with this open-source repo.
- Cloud Service — try instantly with our [Chat Platform](https://chat.pageindex.ai/), or integrate with [MCP](https://pageindex.ai/mcp) or [API](https://docs.pageindex.ai/quickstart).
- _Enterprise_ — private or on-prem deployment. [Contact us](https://ii2abc2jejf.typeform.com/to/tK3AXl8T) or [book a demo](https://calendly.com/pageindex/meet) for more details.

### 🧪 Quick Hands-on

- 🔥 [**Agentic Vectorless RAG**](examples/openai_agents_demo.py) (**latest**) — a complete **agentic vectorless RAG** [example](https://github.com/VectifyAI/PageIndex/blob/main/examples/openai_agents_demo.py) with *self-hosted* PageIndex, using OpenAI Agents SDK.
- Try the [Vectorless RAG](https://github.com/VectifyAI/PageIndex/blob/main/cookbook/pageindex_RAG_simple.ipynb) notebook — a *minimal*, hands-on example of reasoning-based RAG using PageIndex.
- Check out [Vision-based Vectorless RAG](https://github.com/VectifyAI/PageIndex/blob/main/cookbook/vision_RAG_pageindex.ipynb) — no OCR; a minimal, vision-based & reasoning-native RAG pipeline that works directly over page images.
  
<div align="center">
  <a href="https://github.com/VectifyAI/PageIndex/blob/main/examples/openai_agents_demo.py" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/View_on_GitHub-Agentic_Vectorless_RAG-blue?style=for-the-badge&logo=github" alt="View on GitHub: Agentic Vectorless RAG" />
  </a>
  <br/>
  <a href="https://colab.research.google.com/github/VectifyAI/PageIndex/blob/main/cookbook/pageindex_RAG_simple.ipynb" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/Open_In_Colab-Vectorless_RAG-orange?style=for-the-badge&logo=googlecolab" alt="Open in Colab: Vectorless RAG" />
  </a>
  &nbsp;&nbsp;
  <a href="https://colab.research.google.com/github/VectifyAI/PageIndex/blob/main/cookbook/vision_RAG_pageindex.ipynb" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/Open_In_Colab-Vision_RAG-orange?style=for-the-badge&logo=googlecolab" alt="Open in Colab: Vision RAG" />
  </a>
</div>

---

# 🌲 PageIndex Tree Structure

PageIndex can transform lengthy PDF documents into a semantic **tree structure**, similar to a _"table of contents"_ but optimized for use with Large Language Models (LLMs). It's ideal for: financial reports, regulatory filings, academic textbooks, legal or technical manuals, and any document that exceeds LLM context limits.

Below is an example PageIndex tree structure. Also see more example [documents](https://github.com/VectifyAI/PageIndex/tree/main/tests/pdfs) and generated [tree structures](https://github.com/VectifyAI/PageIndex/tree/main/tests/results).

```jsonc
...
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
...
```

You can generate the PageIndex tree structure with this open-source repo, or use our [API](https://docs.pageindex.ai/quickstart).

---

# ⚙️ Package Usage

You can follow these steps to generate a PageIndex tree from a PDF document.

### 1. Install dependencies

```bash
pip3 install --upgrade -r requirements.txt
```

### 2. Set your LLM API key

Create a `.env` file in the root directory with your LLM API key, with multi-LLM support via [LiteLLM](https://docs.litellm.ai/docs/providers):

```bash
OPENAI_API_KEY=your_openai_key_here
```

### 3. Generate PageIndex structure for your PDF

```bash
python3 run_pageindex.py --pdf_path /path/to/your/document.pdf
```

<details>
<summary>Optional parameters</summary>
<br>
You can customize the processing with additional optional arguments:

```
--model                 LLM model to use (default: gpt-4o-2024-11-20)
--toc-check-pages       Pages to check for table of contents (default: 20)
--max-pages-per-node    Max pages per node (default: 10)
--max-tokens-per-node   Max tokens per node (default: 20000)
--if-add-node-id        Add node ID (yes/no, default: yes)
--if-add-node-summary   Add node summary (yes/no, default: yes)
--if-add-doc-description Add doc description (yes/no, default: yes)
```
</details>

<details>
<summary>Markdown support</summary>
<br>
We also provide markdown support for PageIndex. You can use the `--md_path` flag to generate a tree structure for a markdown file.

```bash
python3 run_pageindex.py --md_path /path/to/your/document.md
```

> Note: in this mode, we use "#" to determine node headings and their levels. For example, "##" is level 2, "###" is level 3, etc. Make sure your markdown file is formatted correctly. If your Markdown file was converted from a PDF or HTML, we don't recommend using this mode, since most existing conversion tools cannot preserve the original hierarchy. Instead, use our [PageIndex OCR](https://pageindex.ai/blog/ocr), which is designed to preserve the original hierarchy, to convert the PDF to a markdown file and then use this mode.
</details>

### Agentic Vectorless RAG Example

For a complete **agentic vectorless RAG** example with PageIndex, see [`examples/openai_agents_demo.py`](examples/openai_agents_demo.py) (using [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)).

```bash
# Install optional dependency
pip3 install openai-agents

# Run the demo
python3 examples/openai_agents_demo.py
```

<!--
# ☁️ Improved Tree Generation with PageIndex OCR

This repo is designed for generating PageIndex tree structure for simple PDFs, but many real-world use cases involve complex PDFs that are hard to parse by classic Python tools. However, extracting high-quality text from PDF documents remains a non-trivial challenge. Most OCR tools only extract page-level content, losing the broader document context and hierarchy.

To address this, we introduced PageIndex OCR — the first long-context OCR model designed to preserve the global structure of documents. PageIndex OCR significantly outperforms other leading OCR tools, such as those from Mistral and Contextual AI, in recognizing true hierarchy and semantic relationships across document pages.

- Experience next-level OCR quality with PageIndex OCR at our [Dashboard](https://dash.pageindex.ai/).
- Integrate PageIndex OCR seamlessly into your stack via our [API](https://docs.pageindex.ai/quickstart).

<p align="center">
  <img src="https://github.com/user-attachments/assets/eb35d8ae-865c-4e60-a33b-ebbd00c41732" width="80%">
</p>
-->

---

# 📈 Case Study: PageIndex Leads Finance QA Benchmark

[Mafin 2.5](https://vectify.ai/mafin) is a reasoning-based RAG system for financial document analysis, powered by **PageIndex**. It achieved a state-of-the-art [**98.7% accuracy**](https://vectify.ai/blog/Mafin2.5) on the [FinanceBench](https://arxiv.org/abs/2311.11944) benchmark, significantly outperforming traditional vector-based RAG systems.

PageIndex's hierarchical indexing and reasoning-driven retrieval enable precise navigation and extraction of relevant context from complex financial reports, such as SEC filings and earnings disclosures.

Explore the full [benchmark results](https://github.com/VectifyAI/Mafin2.5-FinanceBench) and our [blog post](https://vectify.ai/blog/Mafin2.5) for detailed comparisons and performance metrics.

<div align="center">
  <a href="https://github.com/VectifyAI/Mafin2.5-FinanceBench">
    <img src="https://github.com/user-attachments/assets/571aa074-d803-43c7-80c4-a04254b782a3" width="70%">
  </a>
</div>

---

# 🧭 Resources

* 🧪 [Cookbooks](https://docs.pageindex.ai/cookbook/vectorless-rag-pageindex): hands-on, runnable examples and advanced use cases.
* 📖 [Tutorials](https://docs.pageindex.ai/doc-search): practical guides and strategies, including *Document Search* and *Tree Search*.
* 📝 [Blog](https://pageindex.ai/blog): technical articles, research insights, and product updates.
* 🔌 [MCP setup](https://pageindex.ai/mcp#quick-setup) & [API docs](https://docs.pageindex.ai/quickstart): integration details and configuration options.

---

# ⭐ Support Us
Please cite this work as:
```
Mingtian Zhang, Yu Tang and PageIndex Team,
"PageIndex: Next-Generation Vectorless, Reasoning-based RAG",
PageIndex Blog, Sep 2025.
```

<details>
<summary>Or use the BibTeX citation</summary>

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

Leave us a star 🌟 if you like our project. Thank you!  

<p>
  <img src="https://github.com/user-attachments/assets/eae4ff38-48ae-4a7c-b19f-eab81201d794" width="80%">
</p>

### Connect with Us

[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/PageIndexAI)&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/vectify-ai/)&nbsp;
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/invite/VuXuf29EUj)&nbsp;
[![Contact Us](https://img.shields.io/badge/Contact_Us-3B82F6?style=for-the-badge&logo=envelope&logoColor=white)](https://ii2abc2jejf.typeform.com/to/tK3AXl8T)

---

© 2026 [Vectify AI](https://vectify.ai)
