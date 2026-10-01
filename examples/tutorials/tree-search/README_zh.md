## 树搜索 示例
本教程提供了使用 PageIndex 树结构进行检索的基础示例。

### 基础 LLM 树搜索示例
一种简单的策略是使用 LLM agent 进行树搜索。以下是一个基础的树搜索提示词。

```python
prompt = f"""
You are given a query and the tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure: {PageIndex_Tree}

Reply in the following JSON format:
{{
  "thinking": <your reasoning about which nodes are relevant>,
  "node_list": [node_id1, node_id2, ...]
}}
"""
```
<callout>
在我们的控制台和检索 API 中，我们使用了 LLM 树搜索与基于价值函数的蒙特卡洛树搜索（[MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)）相结合的方式。更多细节即将发布。
</callout>

### 融合用户偏好或专家知识
与基于向量的 RAG 不同——后者需要微调嵌入模型才能融入专家知识或用户偏好——在 PageIndex 中，你只需将用户偏好或专家知识直接添加到 LLM 树搜索提示词中即可。以下是一个示例流程。

#### 1. 偏好检索

当收到查询时，系统会从数据库或一组领域规则中选取最相关的用户偏好或专家知识片段。这可以通过关键词匹配、语义相似度或基于 LLM 的相关性搜索来实现。

#### 2. 结合偏好的树搜索
将偏好融入树搜索提示词中。

**融合专家偏好的增强树搜索示例**

```python
prompt = f"""
You are given a question and a tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure:  {PageIndex_Tree}

Expert Knowledge of relevant sections: {Preference}

Reply in the following JSON format:
{{
  "thinking": <reasoning about which nodes are relevant>,
  "node_list": [node_id1, node_id2, ...]
}}
"""
```

**专家偏好示例**
> If the query mentions EBITDA adjustments, prioritize Item 7 (MD&A) and footnotes in Item 8 (Financial Statements) in 10-K reports.

通过融合用户或专家偏好，节点搜索变得更加精准有效，同时利用了文档结构和领域专长两方面的优势。

## 💬 帮助与社区
如果你需要关于在你的场景中如何实施文档搜索的建议，请联系我们。

- 🤝 [加入我们的 Discord](https://discord.gg/VuXuf29EUj)  
- 📨 [给我们留言](https://ii2abc2jejf.typeform.com/to/tK3AXl8T)
