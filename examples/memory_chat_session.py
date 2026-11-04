"""Explicit in-memory sessions over an already indexed local document.

Requires an existing own-model chat configuration, as the stateless SDK does.
This example does not provision credentials or index a document.
"""
import asyncio
import sys

from pageindex import PageIndexLocalClient
from pageindex.integrations.memory import MemoryChatSession


async def main(doc_id, model):
    client = PageIndexLocalClient(storage_path=".pageindex", chat_model=model)
    session = MemoryChatSession(client, doc_id=doc_id, model=model)
    first = await session.chat("What is this document about?")
    print(first["choices"][0]["message"]["content"])
    second = await session.chat("Explain the most important point in more detail.")
    print(second["choices"][0]["message"]["content"])
    # session.messages is a copy of all committed questions, answers, tool
    # calls and results. session.context exposes the fixed prompt context.
    await session.clear()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
