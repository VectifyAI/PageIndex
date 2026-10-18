"""PageIndex can use your Claude subscription to call Claude models through
Claude Code: usage is covered by your plan, and no API key is needed.

    pip install -U "pageindex[claude]"
    python pageindex-claude-subscription.py
"""
from pathlib import Path

import requests
from pageindex import PageIndexClient

url = "https://github.com/VectifyAI/PageIndex/raw/main/examples/documents/q1-fy25-earnings.pdf"
Path("q1-fy25-earnings.pdf").write_bytes(requests.get(url).content)

client = PageIndexClient(
    index="claude-code/haiku",          # model to build the tree index
    chat="claude-code/opus",            # model to search the tree
)
doc_id = client.submit_document("q1-fy25-earnings.pdf")["doc_id"]

answer = client.chat("What was the total revenue this quarter?", doc_id=doc_id)
print(answer)
