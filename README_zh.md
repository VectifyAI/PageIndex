<div align="center">
  
<a href="https://vectify.ai/pageindex" target="_blank">
  <img src="https://github.com/user-attachments/assets/46201e72-675b-43bc-bfbd-081cc6b65a1d" alt="PageIndex Banner" />
</a>

<br/>
<br/>

<p align="center">
  <a href="https://trendshift.io/repositories/14736" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14736" alt="VectifyAI%2FPageIndex | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# PageIndex：无向量、推理驱动的 RAG

<p align="center"><b>基于推理的 RAG&nbsp; ◦ &nbsp;无需向量数据库与文档切片&nbsp; ◦ &nbsp;上下文感知&nbsp; ◦ &nbsp;类人检索</b></p>

<h4 align="center">
  <a href="https://vectify.ai">🌐 官网</a>&nbsp; • &nbsp;
  <a href="https://chat.pageindex.ai">🖥️ 对话平台</a>&nbsp; • &nbsp;
  <a href="https://pageindex.ai/developer">🔌 MCP & API</a>&nbsp; • &nbsp;
  <a href="https://docs.pageindex.ai">📖 文档</a>&nbsp; • &nbsp;
  <a href="https://discord.com/invite/VuXuf29EUj">💬 Discord</a>&nbsp; • &nbsp;
  <a href="https://ii2abc2jejf.typeform.com/to/tK3AXl8T">✉️ 联系我们</a>&nbsp;
</h4>
  
</div>


<details open>
<summary><h2>📢 最新动态</h2></summary>

- 🔥 [**Agentic 无向量 RAG**](https://github.com/VectifyAI/PageIndex/blob/main/examples/agentic_vectorless_rag_demo.py) — 基于 OpenAI Agents SDK 的简易 *agentic 无向量 RAG* [示例](#agentic-vectorless-rag-示例)，使用自托管 PageIndex。
- [**PageIndex 扩展到百万级文档**](https://pageindex.ai/blog/pageindex-filesystem) — *PageIndex 文件系统* 是文件级树状索引层，让 PageIndex 能够在整个语料库（而非单篇文档）上进行推理，实现大规模文档搜索。
- [PageIndex Chat](https://chat.pageindex.ai) — 面向专业长文档的类人文档分析 agent [平台](https://chat.pageindex.ai)。同时支持 [MCP](https://pageindex.ai/developer) 与 [API](https://pageindex.ai/developer) 接入。
- [PageIndex 框架](https://pageindex.ai/blog/pageindex-intro) — 深入解读 PageIndex：一种 *agentic 上下文树索引*，使 LLM 能够对长文档进行*基于推理的上下文感知检索*。

</details>

---

# 📑 PageIndex 简介

你是否对向量数据库在专业长文档上的检索精度感到困扰？传统的基于向量的 RAG 依赖语义*相似度*而非真正的*相关性*。但**相似度 ≠ 相关性**——检索真正需要的是**相关性**，而这需要**推理**。当处理需要专业领域知识和多步推理的专业文档时，相似度搜索往往力不从心。

受 AlphaGo 的启发，我们提出了 **[PageIndex](https://vectify.ai/pageindex)**——一种**无向量**、**推理驱动**的 RAG 系统。它从长文档中构建**层次树状索引**，并利用 LLM **在该索引上进行推理**，实现 **agentic、上下文感知的检索**。它模拟了*人类专家*通过*树搜索*在复杂文档中导航和提取知识的方式，使 LLM 能够*思考*和*推理*出最相关的文档章节。PageIndex 的检索分为两步：

1. 生成文档的"目录式"**树结构索引**
2. 通过**树搜索**进行基于推理的检索

<div align="center">
  <a href="https://pageindex.ai/blog/pageindex-intro" target="_blank" title="PageIndex 框架">
    <img src="https://docs.pageindex.ai/images/cookbook/vectorless-rag.png" width="70%">
  </a>
</div>

### 🎯 核心特性

与传统基于向量的 RAG 相比，**PageIndex** 具有以下特点：
- **无需向量数据库**：利用文档结构和 LLM 推理进行检索，而非向量相似度搜索。
- **无需文档切片**：文档按自然章节组织，而非人为切分的片段。
- **更好的可解释性与可追溯性**：检索基于推理，可追溯、可解释，附带页面和章节引用。告别不透明的近似向量搜索（"凭感觉检索"）。
- **上下文感知检索**：检索依赖完整上下文（如对话历史和领域知识），并能轻松融入新的上下文信息。
- **类人检索**：模拟人类专家在复杂文档中导航和提取知识的方式。

PageIndex 驱动的推理型 RAG 系统在 FinanceBench 上取得了**业界领先的 [98.7% 准确率](https://github.com/VectifyAI/Mafin2.5-FinanceBench)**，展现了其在专业文档分析中远超向量 RAG 方案的卓越性能。详见我们的[博客文章](https://vectify.ai/blog/Mafin2.5)。

### 📍 探索 PageIndex

了解更多，请参阅 [PageIndex 框架](https://pageindex.ai/blog/pageindex-intro)的详细介绍。在本 GitHub 仓库中可获取开源代码，同时可查看 [Cookbooks](https://docs.pageindex.ai/cookbook)、[教程](https://docs.pageindex.ai/tutorials) 和[博客](https://pageindex.ai/blog)获取更多使用指南和示例。

PageIndex 服务以 ChatGPT 风格的[对话平台](https://chat.pageindex.ai)提供，也可通过 [MCP](https://pageindex.ai/developer) 或 [API](https://pageindex.ai/developer) 集成。

### 🛠️ 部署方式
- **自托管** — 使用本开源仓库在本地运行（使用标准 PDF 解析）。
- **云服务** — 生产级管线，拥有增强的 OCR、树构建和检索能力，以获得最佳效果。可立即在[对话平台](https://chat.pageindex.ai/)上体验，或通过 [MCP](https://pageindex.ai/developer) 与 [API](https://pageindex.ai/developer) 集成。
- **企业版** — 私有化或本地部署。[联系我们](https://ii2abc2jejf.typeform.com/to/tK3AXl8T) 或[预约演示](https://calendly.com/pageindex/meet)了解更多。

### 🧪 快速上手

- 🔥 [**Agentic 无向量 RAG**](examples/agentic_vectorless_rag_demo.py)（**最新**）— 基于 OpenAI Agents SDK 的简易完整 **agentic 无向量 RAG** [示例](#agentic-vectorless-rag-示例)，使用*自托管* PageIndex。
- 试用 [Vectorless RAG](https://github.com/VectifyAI/PageIndex/blob/main/cookbook/pageindex_RAG_simple.ipynb) Notebook — 使用 PageIndex 进行推理型 RAG 的*最小化*、实战示例。
- 查看 [Vision-based Vectorless RAG](https://github.com/VectifyAI/PageIndex/blob/main/cookbook/vision_RAG_pageindex.ipynb) — 无需 OCR；直接基于页面图像的最小化视觉推理型 RAG 管线。
  
<div align="center">
  <a href="https://github.com/VectifyAI/PageIndex/blob/main/examples/agentic_vectorless_rag_demo.py" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/在_GitHub_上查看-Agentic_Vectorless_RAG-blue?style=for-the-badge&logo=github" alt="在 GitHub 上查看：Agentic Vectorless RAG" />
  </a>
  <br/>
  <a href="https://colab.research.google.com/github/VectifyAI/PageIndex/blob/main/cookbook/pageindex_RAG_simple.ipynb" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/在_Colab_中打开-Vectorless_RAG-orange?style=for-the-badge&logo=googlecolab" alt="在 Colab 中打开：Vectorless RAG" />
  </a>
  &nbsp;&nbsp;
  <a href="https://colab.research.google.com/github/VectifyAI/PageIndex/blob/main/cookbook/vision_RAG_pageindex.ipynb" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/在_Colab_中打开-Vision_RAG-orange?style=for-the-badge&logo=googlecolab" alt="在 Colab 中打开：Vision RAG" />
  </a>
</div>

---

# 🌲 PageIndex 树结构

PageIndex 可以将冗长的 PDF 文档转化为语义**树结构**，类似于*目录*，但针对大语言模型 (LLM) 的使用进行了优化。它适用于：财务报告、监管文件、学术教材、法律或技术手册，以及任何超出 LLM 上下文限制的文档。

以下是 PageIndex 树结构示例。也可查看更多的示例[文档](https://github.com/VectifyAI/PageIndex/tree/main/examples/documents)和生成的[树结构](https://github.com/VectifyAI/PageIndex/tree/main/examples/documents/results)。

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

你可以使用此开源仓库生成 PageIndex 树结构，或使用我们的 [API](https://pageindex.ai/developer) 获取由增强 OCR 和树构建管线驱动的更高质量结果。

---

# ⚙️ 使用指南

> **注意：** 本包使用标准 PDF 解析。对于复杂 PDF 的使用场景，我们的[云服务](https://pageindex.ai/developer)（通过 MCP 和 API）提供增强的 OCR、树构建和检索能力。

按照以下步骤，从 PDF 文档生成 PageIndex 树结构。

### 1. 安装依赖

```bash
pip3 install --upgrade -r requirements.txt
```

### 2. 设置 LLM API 密钥

在项目根目录创建 `.env` 文件，填入你的 LLM API 密钥。通过 [LiteLLM](https://docs.litellm.ai/docs/providers) 支持多种 LLM：

```bash
OPENAI_API_KEY=your_openai_key_here
```

### 3. 为你的 PDF 生成 PageIndex 结构

```bash
python3 run_pageindex.py --pdf_path /path/to/your/document.pdf
```

<details>
<summary>可选参数</summary>
<br>
你可以通过以下可选参数自定义处理流程：

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
<summary>Markdown 支持</summary>
<br>
PageIndex 同样支持 Markdown 文件。你可以使用 `--md_path` 参数为 Markdown 文件生成树结构。

```bash
python3 run_pageindex.py --md_path /path/to/your/document.md
```

> 注意：在此模式下，我们使用 "#" 来确定节点标题及其层级。例如，"##" 为第 2 级，"###" 为第 3 级，以此类推。请确保你的 Markdown 文件格式正确。如果你的 Markdown 文件是从 PDF 或 HTML 转换而来的，我们不建议使用此模式，因为大多数现有转换工具无法保留原始层级结构。此时建议使用我们的 [PageIndex OCR](https://pageindex.ai/blog/ocr)（专为保留层级结构设计）将 PDF 转换为 Markdown 文件后再使用此模式。
</details>

## Agentic 无向量 RAG 示例

关于使用 PageIndex 与 OpenAI Agents SDK 构建端到端 _**agentic 无向量 RAG**_ 的简易示例，请参见 [`examples/agentic_vectorless_rag_demo.py`](examples/agentic_vectorless_rag_demo.py)。

```bash
# Install optional dependency
pip3 install openai-agents

# Run the demo
python3 examples/agentic_vectorless_rag_demo.py
```

---

# 📈 案例研究：PageIndex 领跑金融 QA 基准

[Mafin 2.5](https://vectify.ai/mafin) 是由 **PageIndex** 驱动的金融文档分析推理型 RAG 系统。它在 [FinanceBench](https://arxiv.org/abs/2311.11944) 基准上取得了业界领先的 [**98.7% 准确率**](https://vectify.ai/blog/Mafin2.5)，显著超越传统的向量 RAG 系统。

PageIndex 的层次索引和推理驱动检索，能够精准导航并提取复杂金融报告（如 SEC 文件和财报披露）中的相关内容。

查看完整的[基准测试结果](https://github.com/VectifyAI/Mafin2.5-FinanceBench)和我们的[博客文章](https://vectify.ai/blog/Mafin2.5)，了解详细的对比和性能指标。

<div align="center">
  <a href="https://github.com/VectifyAI/Mafin2.5-FinanceBench">
    <img src="https://github.com/user-attachments/assets/571aa074-d803-43c7-80c4-a04254b782a3" width="70%">
  </a>
</div>

---

# 🧭 资源

* 📝 [博客](https://pageindex.ai/blog)：技术文章、研究洞察和产品更新。
* 🔧 [开发者](https://pageindex.ai/developer)：MCP 配置、API 文档和集成指南。
* 🧪 [Cookbooks](https://docs.pageindex.ai/cookbook)：可运行的实战示例和高级用例。
* 📖 [教程](https://docs.pageindex.ai/tutorials)：实用指南和策略，包括*文档搜索*和*树搜索*。

---

# ⭐ 支持我们

如果你喜欢我们的项目，请给我们一颗星 🌟。感谢！

<p>
  <img src="https://github.com/user-attachments/assets/eae4ff38-48ae-4a7c-b19f-eab81201d794" width="80%">
</p>

请引用本工作：
```
Mingtian Zhang, Yu Tang and PageIndex Team,
"PageIndex: Next-Generation Vectorless, Reasoning-based RAG",
PageIndex Blog, Sep 2025.
```

<details>
<summary>或使用 BibTeX 引用。</summary>

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


### 与我们联系

<div align="center">

[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/PageIndexAI)&ensp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/vectify-ai/)&ensp;
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/invite/VuXuf29EUj)&ensp;
[![联系我们](https://img.shields.io/badge/联系我们-3B82F6?style=for-the-badge&logo=envelope&logoColor=white)](https://ii2abc2jejf.typeform.com/to/tK3AXl8T)

</div>

---

© 2026 [Vectify AI](https://vectify.ai)
