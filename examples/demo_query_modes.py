"""Demo: exercise Collection.query() in all modes.

Creates a temp workspace with 2 small markdown docs, then runs:
  Case 1  — single-doc collection, no doc_ids       (open mode, no warning)
  Case 2  — multi-doc collection, no doc_ids        (open mode, UserWarning)
  Case 2b — same as Case 2 + PAGEINDEX_EXPERIMENTAL_MULTIDOC=1 (warning silenced)
  Case 3  — scoped: doc_ids=[one_id]                (no list_documents call)
  Case 4  — scoped: doc_ids=[id1, id2]              (no list_documents call)

Requirements:
  - OPENAI_API_KEY (or any LiteLLM-supported provider key) in env or .env
"""
import asyncio
import os
import shutil
import tempfile
import warnings
from pathlib import Path

# Load .env if present
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from pageindex import PageIndexClient


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


WORKSPACE = tempfile.mkdtemp(prefix="pi_demo_")
print(f"Workspace: {WORKSPACE}")

docs_dir = Path(WORKSPACE) / "docs"
docs_dir.mkdir()
alpha_md = docs_dir / "alpha.md"
alpha_md.write_text(
    "# Alpha\n\n"
    "## Introduction\n"
    "Alpha is about apples and their nutritional value.\n\n"
    "## Health benefits\n"
    "Apples contain fiber and vitamin C, support digestion, and may help "
    "regulate blood sugar.\n"
)
beta_md = docs_dir / "beta.md"
beta_md.write_text(
    "# Beta\n\n"
    "## Introduction\n"
    "Beta is about bananas and potassium.\n\n"
    "## Energy\n"
    "Bananas provide quick energy from natural sugars and are rich in "
    "potassium, supporting muscle function.\n"
)

client = PageIndexClient(model="gpt-4o-2024-11-20", storage_path=WORKSPACE)


async def stream_and_collect(coro_or_stream) -> list[str]:
    """Iterate a QueryStream, print tool calls and answer, return tool-call names."""
    calls: list[str] = []
    async for ev in coro_or_stream:
        if ev.type == "tool_call":
            calls.append(ev.data["name"])
            print(f"  [tool] {ev.data['name']}({ev.data.get('args','')})")
        elif ev.type == "text_done":
            text = str(ev.data)
            print(f"  [answer] {text[:160]}{'...' if len(text) > 160 else ''}")
    return calls


try:
    # ── Case 1 ────────────────────────────────────────────────────────────
    banner("Case 1: single-doc collection, no doc_ids (no warning expected)")
    single = client.collection("single_test")
    d_alpha_solo = single.add(str(alpha_md))
    print(f"Indexed: {d_alpha_solo}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        answer = single.query("What is alpha about?")
        uw = [w for w in caught if issubclass(w.category, UserWarning)]
    print(f"UserWarning count: {len(uw)} (expected 0)")
    print(f"Answer: {answer[:160]}{'...' if len(answer) > 160 else ''}")

    # ── Case 2 ────────────────────────────────────────────────────────────
    banner("Case 2: multi-doc collection, no doc_ids (UserWarning expected)")
    multi = client.collection("multi_test")
    d1 = multi.add(str(alpha_md))
    d2 = multi.add(str(beta_md))
    print(f"Indexed: {d1}, {d2}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        answer = multi.query("What are these documents about?")
        uw = [w for w in caught if issubclass(w.category, UserWarning)]
    print(f"UserWarning count: {len(uw)} (expected 1)")
    for w in uw:
        print(f"  ⚠ {str(w.message)[:140]}")
    print(f"Answer: {answer[:160]}{'...' if len(answer) > 160 else ''}")

    # ── Case 2b ───────────────────────────────────────────────────────────
    banner("Case 2b: same as Case 2 + PAGEINDEX_EXPERIMENTAL_MULTIDOC=1 (silenced)")
    prev = os.environ.get("PAGEINDEX_EXPERIMENTAL_MULTIDOC")
    os.environ["PAGEINDEX_EXPERIMENTAL_MULTIDOC"] = "1"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            answer = multi.query("What are these documents about?")
            uw = [w for w in caught if issubclass(w.category, UserWarning)]
        print(f"UserWarning count: {len(uw)} (expected 0)")
        print(f"Answer: {answer[:160]}{'...' if len(answer) > 160 else ''}")
    finally:
        if prev is None:
            del os.environ["PAGEINDEX_EXPERIMENTAL_MULTIDOC"]
        else:
            os.environ["PAGEINDEX_EXPERIMENTAL_MULTIDOC"] = prev

    # ── Case 3 ────────────────────────────────────────────────────────────
    banner(f"Case 3: scoped, doc_ids=[{d1[:8]}…]  (no list_documents)")

    async def case3():
        calls = await stream_and_collect(
            multi.query("What are apples good for?", doc_ids=[d1], stream=True)
        )
        assert "list_documents" not in calls, f"unexpected list_documents call: {calls}"
        print(f"Tools called: {calls}")
    asyncio.run(case3())

    # ── Case 4 ────────────────────────────────────────────────────────────
    banner(f"Case 4: scoped, doc_ids=[{d1[:8]}…, {d2[:8]}…]  (no list_documents)")

    async def case4():
        calls = await stream_and_collect(
            multi.query("Compare alpha and beta briefly.",
                        doc_ids=[d1, d2], stream=True)
        )
        assert "list_documents" not in calls, f"unexpected list_documents call: {calls}"
        print(f"Tools called: {calls}")
    asyncio.run(case4())

    print("\nAll cases passed.")

finally:
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    print(f"\nCleaned up {WORKSPACE}")
