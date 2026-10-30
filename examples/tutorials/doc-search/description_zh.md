
## 按描述搜索文档

对于没有元数据的文档，可以使用 LLM 生成的描述来辅助文档筛选。这是一种轻量级方法，最适用于少量文档的场景。

### 示例流程

#### 生成 PageIndex 树结构
将所有文档上传至 PageIndex，获取其 `doc_id` 和树结构。

#### 生成文档描述

基于每篇文档的 PageIndex 树结构和节点摘要，为每篇文档生成一句话描述。

```python
prompt = f"""
You are given a table of contents structure of a document.
Your task is to generate a one-sentence description for the document that makes it easy to distinguish from other documents.

Document tree structure: {PageIndex_Tree}

Directly return the description, do not include any other text.
"""
```

#### 使用 LLM 搜索

使用 LLM 将用户查询与生成的文档描述进行对比，从而选取相关文档。

以下是根据文档描述进行文档选取的示例提示词：

```python
prompt = f"""
You are given a list of documents with their IDs, file names, and descriptions. Your task is to select documents that may contain information relevant to answering the user query.

Query: {query}

Documents: [
    {
        "doc_id": "xxx",
        "doc_name": "xxx",
        "doc_description": "xxx"
    }
]

Response Format:
{{
    "thinking": "<Your reasoning for document selection>",
    "answer": <Python list of relevant doc_ids>, e.g. ['doc_id1', 'doc_id2']. Return [] if no documents are relevant.
}}

Return only the JSON structure, with no additional output.
"""
```

#### 使用 PageIndex 检索

使用已筛选出的文档的 PageIndex `doc_id`，通过 PageIndex 检索 API 进行进一步的检索。

## 💬 帮助与社区
如果你需要关于在你的场景中如何实施文档搜索的建议，请联系我们。

- 🤝 [加入我们的 Discord](https://discord.gg/VuXuf29EUj)  
- 📨 [给我们留言](https://ii2abc2jejf.typeform.com/to/meB40zV0)
