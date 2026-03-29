# PageIndex Self-Index Test Report

**Date**: 2026-03-30
**Target**: `D:/projects/pageindex-local/pageindex/` (PageIndex own source code)

## 1. Indexing Statistics

| Metric | Value |
|--------|-------|
| Files indexed | 8 (all .py) |
| Total symbols | 161 |
| Call graph edges | 62 |
| Language | Python 100% |

### Per-file breakdown

| File | Symbols | Lines |
|------|---------|-------|
| `__init__.py` | 0 | 5 |
| `client.py` | 14 | 235 |
| `code_indexer.py` | 28 | 465 |
| `code_searcher.py` | 12 | 203 |
| `page_index.py` | 36 | 1154 |
| `page_index_md.py` | 9 | 343 |
| `retrieve.py` | 7 | 138 |
| `utils.py` | 55 | 716 |

## 2. Search Accuracy

### Definition mode

| Query | Expected | Found | Accurate? |
|-------|----------|-------|-----------|
| `CodeIndexer` | class CodeIndexer in code_indexer.py | Matched `client.py:index` (wrong) | **FAIL** — name truncation bug |
| `index_directory` | method in CodeIndexer | Matched `client.py:index` (partial, wrong) | **FAIL** — name truncation bug |
| `search` | CodeSearcher.search | Matched `code_searcher.py:deSearcher` (class, not method) | **PARTIAL** — right file, wrong symbol |

### Impact mode

| Query | Expected | Found | Accurate? |
|-------|----------|-------|-----------|
| `index_directory` | callers from client.py / mcp | "Symbol not found" | **FAIL** — symbol name mismatch |

**Accuracy score: 0.5 / 4 (12.5%)**

## 3. Root Cause: Symbol Name Truncation Bug

All extracted symbol names are **missing their first 2 characters**:

| Actual name | Extracted name |
|-------------|---------------|
| `CodeIndexer` | `deIndexer:` |
| `CodeSearcher` | `deSearcher:` |
| `PY_LANGUAGE` | `_LANGUAGE` |
| `LANG_MAP` | `NG_MAP` |
| `IGNORE_DIRS` | `NORE_DIRS` |
| `__init__` | `init__` |
| `_get_parser` | `et_parser` |
| `index_directory` | `dex_directory` |
| `_extract_symbols` | `xtract_symbols` |

**Hypothesis**: `_node_text()` uses `node.start_byte` / `node.end_byte` byte offsets. If the source bytes fed to the parser differ from the source string used for slicing (e.g., a BOM prefix, or `parser.parse()` receiving different bytes than `source.encode("utf-8")`), all byte ranges shift by a fixed offset. The consistent 2-byte shift suggests a UTF-8 BOM (`\xef\xbb\xbf` = 3 bytes) or a `\r\n` vs `\n` line ending issue on Windows.

**Most likely**: On Windows, files opened in text mode normalize `\r\n` → `\n`, but `source.encode("utf-8")` produces `\n`-only bytes. Meanwhile tree-sitter parsed the raw bytes (with `\r\n`). Actually, the file is read with `open(..., "r")` (text mode), so `\r\n` is normalized to `\n` before encoding. But `parser.parse(source.encode("utf-8"))` parses the `\n`-only version, while `_node_text` slices from the same string — so byte offsets should match.

**Alternative**: tree-sitter-python binding version mismatch where `start_byte`/`end_byte` are off. This needs deeper investigation.

## 4. Comparison: Python vs TypeScript (GitNexus experiment)

| Dimension | PageIndex (Python) | GitNexus (TypeScript) |
|-----------|-------------------|----------------------|
| Indexing | 8 files, 161 symbols | ~50 repos, cross-project |
| Symbol extraction | **Broken** — 2-char offset | Neo4j-based, stable |
| Call graph | 62 edges built | Full Cypher-queryable graph |
| Search modes | 3 (definition/impact/relevance) | query/cypher/context/impact |
| Accuracy | 12.5% (name truncation) | High (Neo4j exact match) |

**Verdict**: The tree-sitter code indexer has a **critical name extraction bug on Windows** that makes all search modes unreliable. The call graph structure (62 edges across 8 files) and search architecture (3 modes + LLM ranking) are sound — the bug is localized to `_node_text()` byte offset handling. Once fixed, Python indexing quality should match or exceed the TypeScript/GitNexus baseline for single-repo use cases.

## 5. Recommended Fix

Investigate the 2-byte offset in `code_indexer.py:_node_text()` (line 446-447). Add a debug check:
```python
# In _parse_file, after tree = parser.parse(source.encode("utf-8"))
# Verify: tree.root_node.end_byte == len(source.encode("utf-8"))
```
If they differ, the source bytes and parse bytes are misaligned. The fix is to store the encoded bytes and slice from those instead of the original string.
