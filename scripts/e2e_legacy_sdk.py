"""End-to-end smoke test of the legacy SDK compatibility layer against the real cloud API.

Run: PAGEINDEX_API_KEY=... uv run python scripts/e2e_legacy_sdk.py
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pageindex import PageIndexClient


def log(step: str, detail: str = "") -> None:
    print(f"[e2e] {step}" + (f" — {detail}" if detail else ""), flush=True)


def main() -> int:
    api_key = os.environ.get("PAGEINDEX_API_KEY")
    if not api_key:
        print("PAGEINDEX_API_KEY not set", file=sys.stderr)
        return 1

    pdf = Path("examples/documents/attention-residuals.pdf")
    if not pdf.exists():
        print(f"Test PDF missing: {pdf}", file=sys.stderr)
        return 1

    client = PageIndexClient(api_key=api_key)
    log("init", f"cloud mode (key={api_key[:6]}…)")

    # 1) submit_document (legacy SDK signature — fire-and-forget)
    submit_resp = client.submit_document(file_path=str(pdf))
    doc_id = submit_resp["doc_id"]
    log("submit_document", f"doc_id={doc_id}")

    try:
        # 2) poll is_retrieval_ready (with hard timeout)
        deadline = time.time() + 600  # 10 min
        while time.time() < deadline:
            if client.is_retrieval_ready(doc_id):
                log("is_retrieval_ready", "True")
                break
            time.sleep(8)
        else:
            log("is_retrieval_ready", "TIMEOUT")
            return 2

        # 3) get_tree
        tree = client.get_tree(doc_id)
        node_count = len(tree.get("result") or tree.get("tree") or [])
        log("get_tree", f"top-level nodes={node_count}, status={tree.get('status')}")

        # 4) get_document (metadata)
        meta = client.get_document(doc_id)
        log("get_document", f"name={meta.get('name')!r} pages={meta.get('pageNum')} status={meta.get('status')}")

        # 5) chat_completions (non-stream)
        chat = client.chat_completions(
            messages=[{"role": "user", "content": "What is this paper about? Answer in one sentence."}],
            doc_id=doc_id,
        )
        answer = (chat.get("choices") or [{}])[0].get("message", {}).get("content", "")
        log("chat_completions", f"answer={answer[:120]!r}")

        # 6) chat_completions (stream) — full consumption
        log("chat_completions stream", "starting…")
        print("[stream] ", end="", flush=True)
        chunk_count = 0
        for chunk in client.chat_completions(
            messages=[{"role": "user", "content": "List 3 keywords from this paper."}],
            doc_id=doc_id,
            stream=True,
        ):
            print(chunk, end="", flush=True)
            chunk_count += 1
        print()  # newline after streaming
        log("chat_completions stream", f"chunks received={chunk_count}")

    finally:
        # 7) delete_document
        del_resp = client.delete_document(doc_id)
        log("delete_document", f"resp={del_resp}")

    log("done", "all steps OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
