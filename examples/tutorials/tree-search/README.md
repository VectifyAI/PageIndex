## Tree Search Examples
This tutorial provides a basic example of how to perform retrieval using the PageIndex tree.

### Basic LLM Tree Search Example
A simple strategy is to use an LLM agent to conduct tree search. Here is a basic tree search prompt.

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
In our dashboard and retrieval API, we use a combination of LLM tree search and value function-based Monte Carlo Tree Search ([MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)). More details will be released soon.
</callout>

### Integrating User Preference or Expert Knowledge
Unlike vector-based RAG where integrating expert knowledge or user preference requires fine-tuning the embedding model, in PageIndex, you can incorporate user preferences or expert knowledge by simply adding knowledge to the LLM tree search prompt. Here is an example pipeline.


#### 1. Preference Retrieval

When a query is received, the system selects the most relevant user preference or expert knowledge snippets from a database or a set of domain-specific rules. This can be done using keyword matching, semantic similarity, or LLM-based relevance search.

#### 2. Tree Search with Preference
Integrating preference into the tree search prompt.

**Enhanced Tree Search with Expert Preference Example**

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

**Example Expert Preference**
> If the query mentions EBITDA adjustments, prioritize Item 7 (MD&A) and footnotes in Item 8 (Financial Statements) in 10-K reports.



By integrating user or expert preferences, node search becomes more targeted and effective, leveraging both the document structure and domain-specific insights.

## 💬 Help & Community
Contact us if you need any advice on conducting document searches for your use case.

- 🤝 [Join our Discord](https://discord.gg/VuXuf29EUj)  
- 📨 [Leave us a message](https://ii2abc2jejf.typeform.com/to/tK3AXl8T)
