# 01_REPO_MAP.md — PageIndex Codebase Map

## Package Exports (`pageindex/__init__.py`)
```python
from .page_index import *           # page_index(), page_index_main()
from .page_index_md import md_to_tree
from .retrieve import get_document, get_document_structure, get_page_content
from .client import PageIndexClient
```

---

## `pageindex/page_index.py` — PDF Indexing (1153 lines)

### TOC Detection & Extraction
| Function | Purpose |
|----------|---------|
| `toc_detector_single_page(content)` | Detects if a page contains a table of contents |
| `find_toc_pages(start_page_index, page_list, opt)` | Scans pages for TOC presence |
| `toc_extractor(page_list, toc_page_list, model)` | Extracts raw TOC text from pages |
| `detect_page_index(toc_content, model)` | Checks if TOC has page numbers |
| `toc_transformer(toc_content, model)` | Transforms raw TOC to JSON structure |
| `extract_toc_content(content, model)` | Full TOC extraction with continuation logic |

### TOC Index Mapping
| Function | Purpose |
|----------|---------|
| `toc_index_extractor(toc, content, model)` | Maps TOC entries to physical page indices |
| `extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index)` | Matches TOC entries with page indices |
| `calculate_page_offset(pairs)` | Computes offset between TOC page numbers and physical indices |
| `add_page_offset_to_toc_json(data, offset)` | Applies offset to TOC entries |

### Title Verification
| Function | Purpose |
|----------|---------|
| `check_title_appearance(item, page_list, start_index, model)` | Async — verifies section title appears on page |
| `check_title_appearance_in_start(title, page_text, model)` | Async — checks if section starts at page beginning |
| `check_title_appearance_in_start_concurrent(structure, page_list, model)` | Async — batch title start verification |

### Tree Building
| Function | Purpose |
|----------|---------|
| `page_list_to_group_text(page_contents, token_lengths, max_tokens, overlap_page)` | Chunks pages into LLM-digestible groups |
| `add_page_number_to_toc(part, structure, model)` | Adds page numbers to partial TOC |
| `remove_first_physical_index_section(text)` | Strips first section between `<physical_index_*>` tags |
| `list_to_tree(data)` | Converts flat list to hierarchical tree |
| `add_preface_if_needed(data)` | Inserts "Preface" node if doc starts after page 1 |
| `post_processing(structure, end_physical_index)` | Converts `physical_index` → `start_index`/`end_index` |

### Verification & Correction
| Function | Purpose |
|----------|---------|
| `verify_toc(page_list, list_result, start_index, N, model)` | Async — checks TOC accuracy via LLM |
| `fix_incorrect_toc(toc_with_page_number, page_list, incorrect_results, ...)` | Async — retries failed TOC items |
| `fix_incorrect_toc_with_retries(...)` | Async — multiple fix attempts |
| `validate_and_truncate_physical_indices(...)` | Removes out-of-bounds page indices |

### Large Node Processing
| Function | Purpose |
|----------|---------|
| `process_large_node_recursively(node, page_list, opt, logger)` | Async — handles oversized nodes by recursive splitting |

### Main Pipeline
| Function | Purpose |
|----------|---------|
| `meta_processor(page_list, mode, toc_content, toc_page_list, start_index, opt, logger)` | Async — orchestrates PDF indexing modes |
| `tree_parser(page_list, opt, doc, logger)` | Async — builds tree structure from pages |
| `page_index_main(doc, opt)` | Synchronous entry point |
| `page_index(doc, model, toc_check_page_num, ...)` | User-facing API |

---

## `pageindex/page_index_md.py` — Markdown Indexing (341 lines)

| Function | Purpose |
|----------|---------|
| `extract_nodes_from_markdown(markdown_content)` | Parses `#` headers into node list |
| `extract_node_text_content(node_list, markdown_lines)` | Extracts text between headers |
| `update_node_list_with_text_token_count(node_list, model)` | Calculates token counts for thinning |
| `tree_thinning_for_index(node_list, min_node_token, model)` | Merges small nodes into parents |
| `build_tree_from_nodes(node_list)` | Converts flat nodes to tree hierarchy |
| `clean_tree_for_output(tree_nodes)` | Removes internal fields |
| `get_node_summary(node, summary_token_threshold, model)` | Async — generates or returns truncated text |
| `generate_summaries_for_structure_md(structure, summary_token_threshold, model)` | Async — batch summary generation |
| `md_to_tree(md_path, if_thinning, min_token_threshold, ...)` | Async — main markdown indexing function |

---

## `pageindex/client.py` — Workspace Client (234 lines)

### `PageIndexClient` Class
| Method | Purpose |
|--------|---------|
| `__init__(api_key, model, retrieve_model, workspace)` | Initializes client, loads workspace |
| `index(file_path, mode)` | Indexes PDF or MD, returns `doc_id` |
| `get_document(doc_id)` | Returns document metadata JSON |
| `get_document_structure(doc_id)` | Returns tree structure JSON |
| `get_page_content(doc_id, pages)` | Returns page content (e.g., `'5-7'`) |

### Internal Helpers
| Method | Purpose |
|--------|---------|
| `_make_meta_entry(doc)` | Builds lightweight meta entry |
| `_read_json(path)` | Safe JSON read |
| `_save_doc(doc_id)` | Persists doc to workspace |
| `_rebuild_meta()` | Scans workspace for docs |
| `_read_meta()` | Reads `_meta.json` |
| `_save_meta(doc_id, entry)` | Updates `_meta.json` |
| `_load_workspace()` | Loads existing docs on init |
| `_ensure_doc_loaded(doc_id)` | Lazy-loads full doc JSON |

### Internal Constants
- `META_INDEX = "_meta.json"` — workspace metadata filename

---

## `pageindex/retrieve.py` — Retrieval Helpers (137 lines)

| Function | Purpose |
|----------|---------|
| `_parse_pages(pages)` | Parses `'5-7'`, `'3,8'`, `'12'` → sorted int list |
| `_count_pages(doc_info)` | Returns PDF page count |
| `_get_pdf_page_content(doc_info, page_nums)` | Extracts text from PDF pages |
| `_get_md_page_content(doc_info, page_nums)` | Extracts text from markdown nodes |
| `get_document(documents, doc_id)` | Returns doc metadata JSON |
| `get_document_structure(documents, doc_id)` | Returns structure JSON (no text) |
| `get_page_content(documents, doc_id, pages)` | Returns page content JSON |

---

## `pageindex/utils.py` — Utilities (710 lines)

### LLM Interface
| Function | Purpose |
|----------|---------|
| `count_tokens(text, model)` | Token counting via litellm |
| `llm_completion(model, prompt, chat_history, return_finish_reason)` | Sync completion with 10 retries |
| `llm_acompletion(model, prompt)` | Async completion with 10 retries |

### JSON Parsing
| Function | Purpose |
|----------|---------|
| `extract_json(content)` | Extracts JSON from ` ```json ` blocks, handles cleanup |
| `get_json_content(response)` | Strips markdown code fences |

### Tree Utilities
| Function | Purpose |
|----------|---------|
| `write_node_id(data, node_id)` | Assigns 4-digit zero-padded IDs |
| `get_nodes(structure)` | Flatten tree to node list |
| `structure_to_list(structure)` | Alias for `get_nodes` |
| `get_leaf_nodes(structure)` | Returns only leaf nodes |
| `is_leaf_node(data, node_id)` | Checks if node is leaf |
| `get_last_node(structure)` | Returns last node |
| `list_to_tree(data)` | Converts flat list to tree |
| `remove_fields(data, fields)` | Recursively removes fields |
| `remove_structure_text(data)` | Removes 'text' field from tree |
| `print_toc(tree, indent)` | Pretty-prints tree |
| `print_json(data, max_len, indent)` | Pretty-prints JSON |

### PDF Utilities
| Function | Purpose |
|----------|---------|
| `extract_text_from_pdf(pdf_path)` | Full PDF text extraction |
| `get_pdf_title(pdf_path)` | Extracts PDF metadata title |
| `get_text_of_pages(pdf_path, start_page, end_page, tag)` | Extracts pages with `<start_index_X>` tags |
| `get_page_tokens(pdf_path, model, pdf_parser)` | Returns `[(text, token_count)]` per page |
| `get_text_of_pdf_pages(pdf_pages, start_page, end_page)` | Gets text from page tuples |
| `get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)` | Gets text with `<physical_index_*>` labels |
| `get_number_of_pages(pdf_path)` | Returns page count |
| `get_pdf_name(pdf_path)` | Extracts sanitized PDF filename |

### Markdown Utilities
| Function | Purpose |
|----------|---------|
| `sanitize_filename(filename, replacement)` | Replaces `/` with replacement |

### Config
| Class | Purpose |
|-------|---------|
| `ConfigLoader` | Loads `config.yaml` with user overrides (from `SimpleNamespace`) |
| `JsonLogger` | JSON file logger for indexing runs |

---

## `run_pageindex.py` — CLI Entry (133 lines)

### Arguments
| Argument | Description |
|----------|-------------|
| `--pdf_path` | Path to PDF file |
| `--md_path` | Path to Markdown file |
| `--model` | Override LLM model |
| `--toc-check-pages` | Max TOC detection pages (PDF) |
| `--max-pages-per-node` | Max pages per tree node (PDF) |
| `--max-tokens-per-node` | Max tokens per node (PDF) |
| `--if-add-node-id` | Add node IDs |
| `--if-add-node-summary` | Add node summaries |
| `--if-add-doc-description` | Add document description |
| `--if-add-node-text` | Include node text |
| `--if-thinning` | Apply tree thinning (MD only) |
| `--thinning-threshold` | Min tokens for thinning (MD only) |
| `--summary-token-threshold` | Summary threshold (MD only) |

### Output
- Writes `{pdf_name}_structure.json` to `./results/` directory

---

## Data Flow Summary

```
PDF/MD Input
    │
    ▼
┌─────────────────────────┐
│  extract_nodes_from_md  │  (page_index_md.py)
│  get_page_tokens        │  (page_index.py / utils.py)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  tree_parser            │  (page_index.py)
│  md_to_tree             │  (page_index_md.py)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  LLM calls via litellm  │
│  - toc_transformer      │
│  - verify_toc           │
│  - fix_incorrect_toc    │
│  - generate_summaries   │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Tree Structure JSON    │
│  {title, node_id,       │
│   start_index, end_index│
│   summary, text, nodes} │
└─────────────────────────┘
```

---

## Config Defaults (`config.yaml`)
```yaml
model: "gpt-4o-2024-11-20"
retrieve_model: "gpt-5.4"
toc_check_page_num: 20
max_page_num_each_node: 10
max_token_num_each_node: 20000
if_add_node_id: "yes"
if_add_node_summary: "yes"
if_add_doc_description: "no"
if_add_node_text: "no"
```

---

## Key Design Patterns

1. **Async LLM Calls**: Heavy use of `asyncio` + `litellm.acompletion` for concurrent API calls
2. **Fallback Modes**: PDF processing has 3 modes (`process_toc_with_page_numbers` → `process_toc_no_page_numbers` → `process_no_toc`)
3. **Token Budgeting**: Groups pages into chunks respecting `max_token_num_each_node` (20k)
4. **Workspace Pattern**: `PageIndexClient` persists indexed documents to a workspace directory
5. **Lazy Loading**: Workspace documents load structure/pages on demand
6. **Retry Logic**: 10 retries with 1s sleep on LLM failures
7. **Verification Loop**: TOC accuracy checked and incorrect entries fixed automatically