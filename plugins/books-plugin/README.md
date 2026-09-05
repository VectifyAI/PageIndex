# books-plugin

Query and digest a personal PageIndex book library from Claude Code.

## Install (Claude Code, desktop/CLI)

    claude plugin marketplace add utkarsh04agrawal/PageIndex
    claude plugin install books-plugin@book-library-plugins

This gives you the `/book-query` skill fully working out of the box,
already wired to the shared library — no further setup. `/book-digest`
works for reading and answering, but its "save a digest" step writes to
`~/github/repos/book-library/digests/` and its book/chapter path calls the
local `books summarize` CLI — both assume the library owner's machine, so
those two things won't work for you unless you're also running the library
locally.

## Install (phone / Claude.ai web)

Skills aren't available on Claude.ai (they're a Claude Code feature), but
the raw library tools are, via a Connector:

1. Open Claude.ai → Settings → Connectors → Add custom connector
2. URL: `https://books-mcp-ulo37etflq-uc.a.run.app/mcp`
3. Authentication: None
4. Ask Claude things like "what books are in my library" or "search the
   library for X" — it has the same `list_books`/`get_structure`/
   `get_pages`/`get_digest`/`list_digests` tools, just without the
   step-by-step retrieval procedure the Skills give Claude Code.

**This URL is not secret-proof** — anyone who has it can read the library.
It isn't published publicly; please don't forward it beyond people you'd
personally hand it to.
