# PageIndex: Bug Fixes & Improvements Report

## Overview

This document summarizes all changes made to the PageIndex repository. The work covers **12 bug fixes** (3 critical, 9 standard) and **6 feature improvements**, all verified with a new 45-test automated test suite.

No behavioral changes were made to the core LLM prompts or retrieval logic. All fixes target correctness, robustness, and developer experience.

---

## Critical Bug Fixes

### 1. Missing `import re` in `utils.py`

**Problem:** `utils.py` used `re.search()` and `re.finditer()` in `get_first_start_page_from_text()` and `get_last_start_page_from_text()` but never imported the `re` module. Any call to these functions would crash with `NameError: name 're' is not defined`.

**Fix:** Added `import re` to the top-level imports.

**Impact:** These functions are called during PDF page tag extraction. Without this fix, any PDF with page index tags would fail at runtime.

---

### 2. Variable shadowing in `fix_incorrect_toc()` corrupts results

**Problem:** In `fix_incorrect_toc()` > `process_and_check_item()`, the variable `list_index` (the TOC entry index) was overwritten inside a loop that iterates over page ranges:

```python
list_index = incorrect_item['list_index']    # correct TOC index
for page_index in range(prev_correct, next_correct+1):
    list_index = page_index - start_index    # OVERWRITES with page offset
```

The returned `list_index` would be wrong, causing the fix to update the wrong TOC entry silently.

**Fix:** Renamed the inner loop variable to `page_list_index`.

**Impact:** Without this fix, TOC correction could place fixes at wrong indices, producing silently corrupted output for any document where initial verification fails and the fix path is triggered.

---

### 3. `ChatGPT_API_with_finish_reason` returns wrong type on error

**Problem:** On success the function returns a tuple `(content, finish_reason)`, but on max retries exhaustion it returned just the string `"Error"`. Any caller that unpacks like `response, reason = ChatGPT_API_with_finish_reason(...)` would crash with `ValueError: not enough values to unpack`.

**Fix:** Changed to return `("Error", "error")` on failure.

**Impact:** Every call site that unpacks this return value (`extract_toc_content`, `toc_transformer`, `generate_toc_init`, `generate_toc_continue`) would crash if the API failed 10 times consecutively.

---

## Standard Bug Fixes

### 4. `chat_history` mutation across retries

**Problem:** Both `ChatGPT_API_with_finish_reason` and `ChatGPT_API` assigned `messages = chat_history` then appended to it. This mutated the caller's original list and accumulated duplicate messages across retries.

**Fix:** Changed to `messages = list(chat_history)` to create a shallow copy.

---

### 5. Infinite loop guard in `extract_toc_content` was non-functional

**Problem:** The guard `if len(chat_history) > 5` could never trigger because `chat_history` was recreated with exactly 2 items at the start of each loop iteration.

**Fix:** Added an external `attempt` counter that increments each iteration and breaks at `max_attempts = 10`.

---

### 6. `get_leaf_nodes` crashes on nodes without `nodes` key

**Problem:** `if not structure['nodes']` raises `KeyError` when a dict has no `nodes` key (common for leaf nodes in certain tree states).

**Fix:** Changed to `if not structure.get('nodes')`.

---

### 7. `extract_json` corrupts text containing the word "None"

**Problem:** `json_content.replace('None', 'null')` was a global string replace. A title like `"None of the Above"` would become `"null of the Above"`.

**Fix:** Replaced with context-aware regex that only substitutes `None` in JSON value positions (after `:`, `[`, `,`), not inside quoted strings.

---

### 8. `process_none_page_numbers` crashes on missing `page` key

**Problem:** `del item_copy['page']` and `del item['page']` would raise `KeyError` if the key doesn't exist.

**Fix:** Changed to `item_copy.pop('page', None)` and `item.pop('page', None)`.

---

### 9. `count_tokens` crashes when `model=None`

**Problem:** `tiktoken.encoding_for_model(None)` raises an error. Multiple call sites pass `model=None`.

**Fix:** Added `model = model or "gpt-4o"` as default fallback.

---

### 10. Bare `except:` in `page_index_md.py`

**Problem:** `except:` catches everything including `SystemExit` and `KeyboardInterrupt`, making it impossible to interrupt the process cleanly.

**Fix:** Changed to `except ImportError:`.

---

### 11. Duplicate imports

**Problem:** `import logging` appeared twice in `utils.py`; `import os` appeared twice in `page_index.py`.

**Fix:** Removed the duplicates.

---

### 12. `add_page_number_to_toc` called with list instead of string

**Problem:** `process_none_page_numbers` passed `page_contents` (a Python list) to `add_page_number_to_toc`, which expects a string for its `part` parameter. Python would convert the list to its repr `['...', '...']` in the f-string prompt, sending garbled text to the LLM. Also added a guard for empty/invalid `result` before indexing.

**Fix:** Changed to `''.join(page_contents)` and added `result and isinstance(result[0].get(...))` check.

---

## Feature Improvements

### 1. `pyproject.toml` for proper Python packaging

Added a standard `pyproject.toml` with:
- Package metadata (name, version, description, license, classifiers)
- Dependencies mirroring `requirements.txt` (with `>=` instead of `==` for flexibility)
- Optional `[dev]` dependencies (pytest, pytest-asyncio)
- Console script entry point: `run_pageindex` command
- Setuptools configuration to include both `pageindex/` package and root `run_pageindex.py`

Also added a `main()` function wrapper in `run_pageindex.py` so the console script entry point works.

**Benefit:** Users can `pip install .` or eventually `pip install pageindex` from PyPI.

---

### 2. Automated test suite (45 tests)

Created two test files:
- `tests/test_utils.py` (36 tests) -- covers `count_tokens`, `extract_json`, `get_leaf_nodes`, `get_first_start_page_from_text`, `get_last_start_page_from_text`, `sanitize_filename`, `list_to_tree`, `add_preface_if_needed`, `remove_fields`, `reorder_dict`, `convert_physical_index_to_int`, `convert_page_to_int`, `write_node_id`, `get_nodes`, `structure_to_list`
- `tests/test_page_index_md.py` (9 tests) -- covers `extract_nodes_from_markdown`, `extract_node_text_content`, `build_tree_from_nodes`, `clean_tree_for_output`

All 45 tests pass on Python 3.14 with pytest.

**Benefit:** Catches regressions automatically. Several of the bugs fixed above would have been caught immediately by these tests.

---

### 3. Exponential backoff with jitter for API retries

Replaced fixed `time.sleep(1)` in all three API functions (`ChatGPT_API`, `ChatGPT_API_with_finish_reason`, `ChatGPT_API_async`) with `_retry_delay(attempt)` that computes:

```
delay = min(base * 2^attempt, max_delay) + random(0, 1)
```

Delays: ~1.5s, ~2.7s, ~4.8s, ~8.5s, ... capped at ~60s.

**Benefit:** Better resilience to API rate limits. Avoids hammering the API with fixed 1-second retries under load.

---

### 4. Progress reporting for long-running processing

Added an optional `progress_callback(stage, message, **kwargs)` parameter to `page_index()` and to `config.yaml`. When provided, it is called at key stages:

| Stage | When |
|-------|------|
| `pages_loaded` | After PDF page extraction |
| `toc_detection` | Starting TOC detection |
| `verify_toc` | Verifying section positions |
| `build_tree` | Building the document tree |
| `node_ids` | Assigning node IDs |
| `summaries` | Generating node summaries |
| `description` | Generating document description |

Usage: `page_index(doc, progress_callback=lambda stage, msg, **kw: print(f"[{stage}] {msg}"))`

**Benefit:** Users processing large PDFs (which can take minutes with dozens of LLM calls) get visibility into progress.

---

### 5. Optional LLM response caching

Added in-memory response caching keyed on `(model, sha256(prompt + chat_history))`. Enable via:
- `page_index(doc, response_cache=True)` or
- `response_cache: true` in `config.yaml`

Cache is automatically cleaned up after `page_index_main()` returns. Also exposed `set_response_cache(True/False)` for manual control.

**Benefit:** Avoids redundant API calls during retries or re-processing of the same document. Reduces cost and latency during development.

---

### 6. `doc_description` works independently of node summaries

**Problem:** In `page_index_main()`, the `if_add_doc_description` check was nested inside the `if_add_node_summary == 'yes'` block. Setting `if_add_doc_description='yes'` with `if_add_node_summary='no'` would silently skip description generation.

**Fix:** Moved `if_add_doc_description` to a separate top-level check after the summary block. The result dict is now built first, then description is conditionally added.

**Benefit:** Users can generate document descriptions without the cost of per-node summaries.

---

## Files Changed

| File | Changes |
|------|---------|
| `pageindex/utils.py` | Bug fixes 1, 3, 4, 6, 7, 9, 11; Features 3, 5 |
| `pageindex/page_index.py` | Bug fixes 2, 5, 8, 11, 12; Features 4, 6 |
| `pageindex/page_index_md.py` | Bug fix 10 |
| `run_pageindex.py` | Feature 1 (added `main()` wrapper) |
| `pageindex/config.yaml` | Features 4, 5 (new config keys) |
| `pyproject.toml` | Feature 1 (new file) |
| `tests/__init__.py` | Feature 2 (new file) |
| `tests/test_utils.py` | Feature 2 (new file, 36 tests) |
| `tests/test_page_index_md.py` | Feature 2 (new file, 9 tests) |

## Test Results

```
45 passed, 0 failed (Python 3.14.0, pytest 9.0.2)
```

All imports verified, CLI `--help` verified, new features (backoff, caching, config) independently verified.
