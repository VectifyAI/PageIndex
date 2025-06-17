<div align="center">
  <a href="https://vectify.ai/pageindex" target="_blank">
    <img src="https://github.com/user-attachments/assets/15c609f9-443d-4d81-a1f3-aa5a6051b676" alt="pg_logo_small" width="300px">
  </a>
</div>

# 📄 PageIndex

Are you frustrated with vector database retrieval accuracy for long professional documents? Traditional vector-based RAG relies on semantic *similarity* rather than true *relevance*. But **similarity ≠ relevance** — what we truly need in retrieval is **relevance**, and that requires **reasoning**. When working with professional documents that demand domain expertise and multi-step reasoning, similarity search often falls short.

🧠 **Reasoning-based RAG** offers a better alternative: enabling LLMs to *think* and *reason* their way to the most relevant document sections. Inspired by AlphaGo, we use *tree search* to perform structured document retrieval. 

**[PageIndex](https://vectify.ai/pageindex)** is a *document indexing system* that builds *search tree structures* from long documents, making them ready for reasoning-based RAG.  It has been used to develop a RAG system that achieved 98.7% accuracy on [FinanceBench](https://vectify.ai/blog/Mafin2.5), demonstrating state-of-the-art performance in document analysis.

<div align="center">
  <a href="https://vectify.ai/pageindex">
    <img src="https://github.com/user-attachments/assets/6604d932-bdf7-435e-8c28-2213e6ea6a5b" alt="PageIndex" width="700px"/>
  </a>
</div>

Self-host it with this open-source repo, or try our ☁️ [Cloud service](https://pageindex.vectify.ai/) — no setup required, with advanced features like OCR for complex and scanned PDFs.

Built by <a href="https://vectify.ai" target="_blank">Vectify AI</a>.

---

# **⭐ What is PageIndex**

PageIndex can transform lengthy PDF documents into a semantic **tree structure**, similar to a *"table of contents"* but optimized for use with Large Language Models (LLMs).
It's ideal for: financial reports, regulatory filings, academic textbooks, legal or technical manuals, and any document that exceeds LLM context limits.

### ✅ Key Features
    
- **Hierarchical Tree Structure**  
  Enables LLMs to traverse documents logically — like an intelligent, LLM-optimized table of contents.

- **Chunk-Free Segmentation**  
  No arbitrary chunking. Nodes follow the natural structure of the document.

- **Precise Page Referencing**  
  Every node contains its summary and start/end page physical index, allowing pinpoint retrieval.

- **Scales to Massive Documents**  
  Designed to handle hundreds or even thousands of pages with ease.

### 📦 PageIndex Format

Here is an example output. See more [example documents](https://github.com/VectifyAI/PageIndex/tree/main/docs) and [generated trees](https://github.com/VectifyAI/PageIndex/tree/main/results).

```
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

---

### ⚠️ Bug Fix Notice

A bug introduced on **April 18** has now been fixed.

If you pulled the repo between **April 18–23**, please update to the latest version:

```bash
git pull origin main
```

Thanks for your understanding 🙏


---

# 🚀 Package Usage

Follow these steps to generate a PageIndex tree from a PDF document.

### 1. Install dependencies

- use pip
```bash
pip3 install -r requirements.txt
```
- use uv (recommended)
```bash
pip install uv
uv sync
```

### 2. Set your OpenAI API key

Copy the `.env.example` file to `.env` and add your API key:

```bash
CHATGPT_API_KEY=your_openai_key_here
BASE_URL=the_model_provider_url_here (default: https://api.openai.com/v1)
```

### 3. Run PageIndex on your PDF

```bash
python3 run_pageindex.py --pdf_path /path/to/your/document.pdf
```
You can customize the processing with additional optional arguments:

```
--model                 OpenAI model to use (default: gpt-4o-2024-11-20)
--toc-check-pages       Pages to check for table of contents (default: 20)
--max-pages-per-node    Max pages per node (default: 10)
--max-tokens-per-node   Max tokens per node (default: 20000)
--if-add-node-id        Add node ID (yes/no, default: yes)
--if-add-node-summary   Add node summary (yes/no, default: no)
--if-add-doc-description Add doc description (yes/no, default: yes)
```

---

# ☁️ Cloud API & Platform (Beta)

Don't want to host it yourself? Try our [hosted API](https://pageindex.vectify.ai/) for PageIndex. The hosted service leverages our custom OCR model for more accurate PDF recognition, delivering better tree structures for complex documents. Ideal for rapid prototyping, production environments, and documents requiring advanced OCR.

You can also upload PDFs from your browser and explore results visually with our [web Dashboard](https://pageindex.ai/files) — no coding needed.

Leave your email in [this form](https://ii2abc2jejf.typeform.com/to/meB40zV0) to receive 1,000 pages for free.

---

# 📈 Case Study: Mafin 2.5 on FinanceBench

[Mafin 2.5](https://vectify.ai/) is a state-of-the-art reasoning-based RAG model designed specifically for financial document analysis. Powered by **PageIndex**, it achieved a market-leading [**98.7% accuracy**](https://vectify.ai/blog/Mafin2.5) on the [FinanceBench](https://arxiv.org/abs/2311.11944) benchmark — significantly outperforming traditional vector-based RAG systems.

PageIndex's hierarchical indexing enabled precise navigation and extraction of relevant content from complex financial reports, such as SEC filings and earnings disclosures.

👉 See the full [benchmark results](https://github.com/VectifyAI/Mafin2.5-FinanceBench) and our [blog post](https://vectify.ai/blog/Mafin2.5) for detailed comparisons and performance metrics.

<div align="center">
  <a href="https://github.com/VectifyAI/Mafin2.5-FinanceBench">
    <img src="https://github.com/user-attachments/assets/571aa074-d803-43c7-80c4-a04254b782a3" width="90%">
  </a>
</div>

---

# 🧠 Reasoning-Based RAG with PageIndex

Use PageIndex to build **reasoning-based retrieval systems** without relying on semantic similarity. Great for domain-specific tasks where nuance matters ([more examples](https://pageindex.vectify.ai/examples/rag)).

### 🔖 Preprocessing Workflow Example
1. Process documents using PageIndex to generate tree structures.
2. Store the tree structures and their corresponding document IDs in a database table.
3. Store the contents of each node in a separate table, indexed by node ID and tree ID.

### 🔖 Reasoning-Based RAG Framework Example
1. Query Preprocessing:
    - Analyze the query to identify the required knowledge
2. Document Selection: 
    - Search for relevant documents and their IDs
    - Fetch the corresponding tree structures from the database
3. Node Selection:
    - Search through tree structures to identify relevant nodes
4. LLM Generation:
    - Fetch the corresponding contents of the selected nodes from the database
    - Format and extract the relevant information
    - Send the assembled context along with the original query to the LLM
    - Generate contextually informed responses


### 🔖 Example Prompt for Node Selection

```python
prompt = f"""
You are given a question and a tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Question: {question}

Document tree structure: {structure}

Reply in the following JSON format:
{{
    "thinking": <reasoning about where to look>,
    "node_list": [node_id1, node_id2, ...]
}}
"""
```
👉 For more examples, see the [PageIndex Dashboard](https://pageindex.vectify.ai/).

---

# 🛤 Roadmap

- [x]  [Detailed examples of document selection, node selection, and RAG pipelines](https://pageindex.vectify.ai/examples/rag)
- [x]  [Integration of reasoning-based retrieval and semantic-based retrieval](https://pageindex.vectify.ai/examples/hybrid-rag)
- [ ]  Efficient tree search methods introduction
- [ ]  Technical report on the design of PageIndex

---

# 🚧 Notice
This project is in its early beta development, and all progress will remain open and transparent. We welcome you to raise issues, reach out with questions, or contribute directly to the project.  

Due to the diverse structures of PDF documents, you may encounter instability during usage. For a more accurate and stable version with a leading OCR integration, please try our [hosted API for PageIndex](https://pageindex.vectify.ai/). Leave your email in [this form](https://ii2abc2jejf.typeform.com/to/meB40zV0) to receive 1,000 pages for free.

Together, let's push forward the revolution of reasoning-based RAG systems.

### 🙋 FAQ
- **Does PageIndex support other LLMs besides OpenAI?**  
  Currently optimized for GPT models, but future versions will support more.

- **Can PageIndex handle scanned PDFs?**  
  Yes! Our [Cloud API](https://pageindex.vectify.ai/) includes advanced OCR specifically for scanned and complex PDFs.

---

# 📬 Contact Us

Need customized support for your documents or reasoning-based RAG system?

:loudspeaker: [Join our Discord](https://discord.com/invite/nnyyEdT2RG)

:envelope: [Leave us a message](https://ii2abc2jejf.typeform.com/to/meB40zV0)

<div align="center">
  <a href="https://vectify.ai" target="_blank">
    <img src="https://github.com/user-attachments/assets/55abe487-9d21-44ad-b686-a008c2d2b7e7" alt="Vectify AI Logo" width="180">
  </a>
</div>
