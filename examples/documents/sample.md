# PageIndex Overview

PageIndex turns long documents into a navigable tree of sections, each with a
summary, so agents can reason over structure instead of flat chunks. This
sample doc is used by the incremental update demo.

## 1. What PageIndex Does

PageIndex parses a PDF or Markdown file into a hierarchical structure of nodes.
Each node holds a title, its text, and a generated summary. The tree lets a
retrieval agent walk from the document root down to the exact section that
answers a question, without embedding every chunk into a vector store.

## 2. Indexing

Indexing builds the tree once. For Markdown, headings define the hierarchy; for
PDFs, the table of contents and page layout are used. Every section is
summarized, and the whole document gets a short description. The result is
persisted in a workspace as JSON keyed by a document id.

## 3. Incremental Update

When a document changes, PageIndex avoids rebuilding everything. It hashes the
file and each section: if the file hash is unchanged the update is skipped
entirely, and if only some sections changed, only those (plus their ancestors)
are re-summarized. Unchanged sections reuse their cached summary.

## 4. Vectorless Retrieval

Because the tree carries summaries at every level, an agent can retrieve by
traversing the structure instead of doing nearest-neighbor search over
embeddings. This keeps retrieval explainable and cheap to maintain.

## Appendix: Key Methods

`client.index(path)` builds the tree. `client.update(doc_id)` refreshes it
incrementally. `client.get_doc_id_by_path(path)` resolves an existing document
so the same file is never indexed twice.
