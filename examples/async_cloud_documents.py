"""Cloud document I/O only; requires a PageIndex service key, no model key.

Run: python examples/async_cloud_documents.py document.pdf
Set PAGEINDEX_API_KEY in the environment before running against the service.
"""
import asyncio
import sys

import httpx

from pageindex import AsyncPageIndexCloudClient


async def main(file_path):
    async with AsyncPageIndexCloudClient(
        timeout=httpx.Timeout(30, connect=10), max_retries=2,
    ) as client:
        # This deadline covers uploading only. A timeout after the bytes were
        # sent leaves the server outcome unknown: do not blindly upload again.
        submitted = await asyncio.wait_for(client.submit_document(file_path), timeout=120)
        doc_id = submitted["doc_id"]
        print(submitted)
        print(await client.get_document(doc_id))  # includes processing status
        # Processing is a separate phase; these may still return "processing".
        tree, documents = await asyncio.gather(client.get_tree(doc_id), client.list_documents())
        print(tree, documents)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
