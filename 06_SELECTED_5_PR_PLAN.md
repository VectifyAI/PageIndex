# 06_SELECTED_5_PR_PLAN.md — PageIndex Top 5 PR Implementation Plan

## Selection Criteria
- **High impact** on user experience or reliability
- **Medium-to-low complexity** — achievable in focused sessions
- **Clear scope** — well-defined problem and solution
- **No existing test suite** — write unit tests alongside fixes
- **Avoid duplicating active PRs** — pick where we can add unique value

---

## Selected PR #1: Fix `get_page_content` Over-Collection on Markdown (Issues #279 + #280)

**Issue**: `get_page_content` for Markdown documents returns content from nodes outside the requested page range when using comma-separated lists (e.g., `'3,8'`).
**PR**: #280 already filed (`fix/md-discrete-pages-overcollection`) but needs validation.

### Current Behavior (retrieve.py:56-76)
```python
def _get_md_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    min_line, max_line = min(page_nums), max(page_nums)  # ← BUG: uses min/max not discrete pages
    # ...
```
The function finds nodes where `min_line <= line_num <= max_line`, which groups all nodes between the smallest and largest line numbers.

### Fix Approach
Rewrite `_get_md_page_content` to:
1. Treat each page number as a discrete line number
2. Return only nodes with `line_num == page` for each requested page
3. Use a set to deduplicate if same line appears in multiple requests

### Implementation Steps
1. Read `pageindex/retrieve.py` — understand current `_get_md_page_content`
2. Modify `_get_md_page_content` to collect per-page nodes discretely
3. Add unit test in `tests/test_retrieve_md.py`
4. Test with example markdown files in `examples/documents/`

### Files to Modify
- `pageindex/retrieve.py` — fix `_get_md_page_content`
- `tests/test_retrieve_md.py` — new test file (create tests/ directory)

---

## Selected PR #2: Add Concurrency Throttling for LLM Requests (Issue #283)

**Issue**: Unthrottled concurrent LLM requests cause HTTP 429 rate limits and cascading `KeyError` in tree generation.
**Problem**: Code uses `asyncio.gather(*tasks)` with no limit on concurrent calls. When processing large documents, hundreds of LLM calls fire simultaneously.

### Fix Approach
Add a semaphore to limit concurrent LLM API calls globally.

### Implementation Steps
1. Create a concurrency limiter module `pageindex/concurrency.py`:
   ```python
   import asyncio
   _sem = asyncio.Semaphore(5)  # Configurable via config.yaml
   
   async def limited_acompletion(*args, **kwargs):
       async with _sem:
           return await llm_acompletion(*args, **kwargs)
   ```
2. Update `config.yaml` to add `max_concurrent_llm_calls: 5`
3. Update `ConfigLoader` to parse this new option
4. Replace direct `llm_acompletion` calls with `limited_acompletion` in:
   - `pageindex/page_index.py` — `check_title_appearance`, `check_title_appearance_in_start`, `verify_toc`, `fix_incorrect_toc`
   - `pageindex/page_index_md.py` — `get_node_summary`
5. Add test `tests/test_concurrency.py` that verifies semaphore behavior

### Files to Modify
- `pageindex/concurrency.py` — new file
- `pageindex/utils.py` — export limiter function
- `pageindex/page_index.py` — use limited versions
- `pageindex/page_index_md.py` — use limited versions
- `pageindex/config.yaml` — add `max_concurrent_llm_calls`
- `tests/test_concurrency.py` — new test file

### Complexity: Medium
- Requires understanding async patterns in the codebase
- Must not break existing functionality
- Semaphore should be configurable

---

## Selected PR #3: Fix Dependency Version Conflict (Issue #286)

**Issue**: `litellm==1.83.7` requires `python-dotenv==1.0.1` but `requirements.txt` pins `python-dotenv==1.2.2` — causes installation failure.

### Fix Approach
Upgrade `litellm` to a version compatible with `python-dotenv>=1.2.2`, or downgrade dotenv in requirements. Check litellm changelog for when dotenv constraint was relaxed.

### Implementation Steps
1. Check litellm release notes for dotenv compatibility
2. Update `requirements.txt`:
   - Option A: Bump litellm to latest (check if it supports dotenv 1.2.x)
   - Option B: Pin python-dotenv to 1.0.1 if litellm requires it
3. Test `pip install -r requirements.txt` in fresh virtual environment
4. Verify PageIndex still imports correctly
5. Update README if installation steps change

### Files to Modify
- `requirements.txt` — version adjustment

### Complexity: Low
- Direct dependency version fix
- Should verify on fresh install

---

## Selected PR #4: Add `--output` CLI Option for Scripted Usage (Issue #268)

**Issue**: CLI output path is hardcoded to `./results/{name}_structure.json`, making scripted/automated usage difficult.

### Current Code (run_pageindex.py:72-75)
```python
output_dir = './results'
output_file = f'{output_dir}/{pdf_name}_structure.json'
os.makedirs(output_dir, exist_ok=True)
```

### Fix Approach
Add `--output-dir` and/or `--output-file` CLI arguments.

### Implementation Steps
1. Add to `run_pageindex.py` argument parser:
   ```python
   parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results (default: ./results)')
   parser.add_argument('--output-file', type=str, default=None,
                       help='Output file path (overrides default naming)')
   ```
2. Use `args.output_dir` and `args.output_file` in the output section
3. Handle case where user provides custom path but directory doesn't exist
4. Update README.md usage section with new options
5. Add `tests/test_cli.py` with subprocess tests for new arguments

### Files to Modify
- `run_pageindex.py` — add CLI arguments
- `README.md` — document new options
- `tests/test_cli.py` — new test file

### Complexity: Low
- Straightforward CLI enhancement
- Clear user demand (issue explicitly mentions scripted usage)

---

## Selected PR #5: Markdown Parser Edge Cases (Issues #245, #246, #249, #250)

**Issue Group**: Multiple open issues about markdown parser failures:
- #245: "Markdown parser fails on common edge cases"
- #246: "fix: make markdown parser robust" (PR exists)
- #249: "fix: fix markdown parser edge cases in page_index_md.py" (PR exists)
- #250: "feat: recognize whole-line bold as level-1 heading in markdown parser" (PR exists)

**Problem**: The markdown parser (`page_index_md.py`) has known edge cases that fail:
- Bold headings not recognized as level-1
- Code blocks interfering with header detection
- Edge cases in header level detection

### Fix Approach
Audit `extract_nodes_from_markdown()` in `page_index_md.py` and add robust handling.

### Implementation Steps
1. Read `pageindex/page_index_md.py` lines 32-59 (`extract_nodes_from_markdown`)
2. Create `examples/documents/test_cases.md` with edge case examples:
   - Bold headers: `**Section Title**`
   - Italic headers: `*Section Title*`
   - Code blocks with hash characters
   - Mixed header levels
   - Headers at start/end of code blocks
3. Write failing tests in `tests/test_markdown_parser.py`
4. Fix `extract_nodes_from_markdown` to handle:
   - Skip headers inside code blocks (already done with `in_code_block` flag — verify)
   - Recognize markdown emphasis patterns as headers
5. Run full markdown test suite

### Files to Modify
- `pageindex/page_index_md.py` — fix `extract_nodes_from_markdown`
- `examples/documents/test_cases.md` — new test file with edge cases
- `tests/test_markdown_parser.py` — new test file

### Complexity: Medium
- Requires understanding of regex patterns
- Multiple edge cases to handle carefully

---

## Summary Table

| # | PR Title | Issue(s) | Files | Complexity |
|---|----------|----------|-------|------------|
| 1 | Fix MD page content over-collection | #279, #280 | retrieve.py, tests/ | Low |
| 2 | Add LLM concurrency throttling | #283 | concurrency.py (new), page_index.py, page_index_md.py, config.yaml | Medium |
| 3 | Fix dependency version conflict | #286 | requirements.txt | Low |
| 4 | Add --output CLI option | #268 | run_pageindex.py, README.md | Low |
| 5 | Markdown parser edge cases | #245, #246, #249, #250 | page_index_md.py, tests/ | Medium |

---

## Testing Strategy

Since the project has **no existing test suite**, create a `tests/` directory with:
- `tests/__init__.py` — package marker
- `tests/conftest.py` — pytest fixtures (sample PDF path, sample MD content)
- `tests/test_retrieve_md.py` — markdown retrieval tests
- `tests/test_concurrency.py` — semaphore behavior tests
- `tests/test_cli.py` — CLI argument tests
- `tests/test_markdown_parser.py` — markdown parsing tests

Use `pytest` as the test framework. Add to `requirements.txt` if not present.

---

## CI Integration

Once tests are written, add a test workflow to `.github/workflows/`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: pip install -r requirements.txt pytest
      - name: Run tests
        run: pytest tests/ -v
```