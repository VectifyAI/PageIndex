---
name: book-query
description: Answer a question from the personal book library (MCP server "books") by navigating each book's tree, reading only the relevant pages, and citing page numbers. Use for any question about what a book says.
---

# Book query

Answer strictly from the library; say so when the books do not cover it.

0. `list_digests` once. If an existing topic digest plausibly covers the
   question, read it (it's a plain file) and use it as a starting point —
   still verify any load-bearing claim against `get_pages` before citing it,
   since a saved digest can predate newly indexed books.
1. `list_books` once. Pick the books whose description matches the question
   (all of them if the question is cross-book). Do not guess a book that is
   not listed.
2. For each candidate: `get_structure(book, depth=2)`. Choose the nodes whose
   title or summary plausibly holds the answer (typically 1–4). If a chosen
   node has children, `get_structure` deeper rather than reading a long range.
3. `get_pages(book, "<start>-<end>")` for each chosen node — tight ranges, at
   most 40 pages per call. Read more nodes only if the answer is incomplete.
4. Answer. Every claim carries a citation like `(Complete Works I, p. 137)`.
   Quote short phrases verbatim when the wording matters. If sources disagree,
   show both.
5. Finish with a "Sources" list: book — node title — pages.

Never answer from memory of the author; the page text is the only source.
