---
name: book-digest
description: Produce a digest from the personal book library (MCP server "books") — of a whole book, one chapter, or a topic across several books — written to ~/github/repos/book-library/digests/. Use when asked to summarize, digest, or synthesize book content.
---

# Book digest

Three shapes; ask which only if it is not obvious from the request.

**Book or chapter digest** (fast path — the library already stores digests):
1. `get_digest(book)` or `get_digest(book, node_id)`. If sections show
   `_(no summary yet)_` or only italic routing summaries, tell the user to run
   `books summarize "<book>" --tier digest` (optionally `--node <id>`) and stop.
2. Otherwise rewrite the returned Markdown into the requested shape (brief /
   essay / flashcards), keeping page references.

**Topic digest across books**:
1. Follow the `book-query` procedure to gather all relevant nodes (read
   `get_digest(book, node)` first, `get_pages` only where the digest lacks
   detail).
2. Write a synthesis: one section per theme, each bullet cited
   `(Book, p. N)`; a closing "Where the books differ" section when they do.

**Output**: write the file with the Write tool to
`~/github/repos/book-library/digests/<topic-or-book-slug>/<name>.md`, then
print the path and a 5-line abstract. Do not invent content that is not in
the pages or digests.
