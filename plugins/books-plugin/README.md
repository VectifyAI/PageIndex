# books-plugin

Query and digest a personal PageIndex book library from Claude Code.

## Install (Claude Code, desktop/CLI)

    claude plugin marketplace add utkarsh04agrawal/PageIndex
    claude plugin install books-plugin@book-library-plugins

This gives you the `/book-query` and `/book-digest` skills, already wired
to the shared library — no further setup.

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
