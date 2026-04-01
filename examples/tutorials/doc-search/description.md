
## Document Search by Description

For documents that don't have metadata, you can use LLM-generated descriptions to help with document selection. This is a lightweight approach that works best with a small number of documents.


### Example Pipeline


#### PageIndex Tree Generation
Upload all documents into PageIndex to get their `doc_id` and tree structure.

#### Description Generation

Generate a description for each document based on its PageIndex tree structure and node summaries.
```python
prompt = f"""
You are given a table of contents structure of a document. 
Your task is to generate a one-sentence description for the document that makes it easy to distinguish from other documents.
    
Document tree structure: {PageIndex_Tree}

Directly return the description, do not include any other text.
"""
```

#### Search with LLM

Use an LLM to select relevant documents by comparing the user query against the generated descriptions.

Below is a sample prompt for document selection based on their descriptions:

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

#### Retrieve with PageIndex

Use the PageIndex `doc_id` of the retrieved documents to perform further retrieval via the PageIndex retrieval API.



## 💬 Help & Community
Contact us if you need any advice on conducting document searches for your use case.

- 🤝 [Join our Discord](https://discord.gg/VuXuf29EUj)  
- 📨 [Leave us a message](https://ii2abc2jejf.typeform.com/to/meB40zV0)