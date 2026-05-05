# PageIndex Indexing Pipeline Summary

This document summarizes the PDF indexing pipeline in PageIndex, with emphasis on how TOC detection, TOC normalization, verification, and repair work before retrieval is used.

For the retrieval side, see [RETRIEVAL_PIPELINE_SUMMARY.md](RETRIEVAL_PIPELINE_SUMMARY.md).

## Main files involved

- `run_pageindex.py` is the CLI entry point for PDF and Markdown indexing.
- `pageindex/client.py` wraps indexing, creates the document ID, stores workspace state, and selects the indexing model.
- `pageindex/page_index.py` contains the PDF indexing pipeline and almost all of the TOC logic.
- `pageindex/page_index_md.py` contains the Markdown indexing pipeline.
- `pageindex/utils.py` contains LiteLLM helpers, token counting, JSON cleanup, config loading, and `.env` loading.
- `pageindex/config.yaml` provides the default models and indexing knobs.

## Where the prompts live

There is no separate prompt directory for indexing. The prompts are embedded directly in Python functions, which makes the flow easy to follow but also means the behavior is scattered across helper functions rather than one prompt registry.

In `pageindex/page_index.py`:

- `toc_detector_single_page` asks a binary question: does this page look like a TOC page or not? It is used as a page-level classifier during TOC scanning.
- `find_toc_pages` loops over early pages and uses `toc_detector_single_page` to decide which consecutive pages belong to the TOC block.
- `check_if_toc_extraction_is_complete` checks whether the extracted TOC text is complete relative to the source pages. It is a guard against truncated generation.
- `check_if_toc_transformation_is_complete` checks whether the structured TOC output is complete after JSON transformation. It is a second guard, this time on the cleaned structure.
- `extract_toc_content` asks the model to copy the raw TOC out of the source text, with retries if the output is incomplete.
- `detect_page_index` checks whether the TOC already contains page numbers or page indices. That decides whether the pipeline can use the more accurate alignment branch.
- `toc_transformer` converts raw TOC text into structured JSON with section hierarchy and page numbers.
- `toc_index_extractor` removes page numbers from the TOC, then matches TOC titles to nearby physical pages so the code can infer actual page positions.
- `add_page_number_to_toc` is used when the TOC has structure but no page numbers. It asks the model to place each TOC item onto the physical page markers in the body text.
- `generate_toc_init` creates the first tree when there is no TOC at all.
- `generate_toc_continue` extends that tree over later page groups when the document is too large to process in one shot.
- `single_toc_item_index_fixer` repairs one bad TOC entry by finding the physical page where that section actually starts.
- `check_title_appearance` verifies a specific section title against a specific page.
- `check_title_appearance_in_start` is a stricter version of the same check that only asks whether the title starts at the beginning of the page.
- `verify_toc` applies the title check across the whole tree or a sample of entries.
- `fix_incorrect_toc` and `fix_incorrect_toc_with_retries` repair only the entries that failed verification.

In `pageindex/page_index_md.py` there are additional prompts for Markdown tree generation and summary extraction, but those are separate from the PDF TOC path.

## The indexing pipeline, end to end

The PDF flow starts in `page_index_main` and builds a tree from the document pages. The broad shape is:

1. Convert the PDF into page text and token counts.
2. Detect whether a TOC exists.
3. If a TOC exists, determine whether it already has page numbers.
4. Build a structured tree from the TOC and align it to real pages.
5. If no TOC exists, generate a tree directly from the document pages.
6. Validate the result against the source pages.
7. Repair incorrect entries if the tree is close enough to trust.
8. Add node IDs, text, summaries, and optionally a document description.

### Step 1: page loading and token counts

`page_index_main` starts by calling `get_page_tokens`, which builds the page list and token lengths used throughout the rest of the pipeline. That page list is the shared data structure passed into TOC detection, verification, repair, and recursive processing.

### Step 2: TOC detection

`check_toc` calls `find_toc_pages`, which scans the early pages looking for a TOC block. The scan stops when the code sees the TOC end or when it reaches the configured TOC check window. This is why TOC detection is cheap compared with the later extraction and repair stages: it is just classifying a handful of pages one by one.

### Step 3: TOC extraction and normalization

Once a TOC is found, `toc_extractor` concatenates those pages and checks whether the TOC already includes page numbers. If it does, the code can move directly into the page-number alignment branch. If not, it has to infer page positions from the body pages, which is slower and more error-prone.

`toc_transformer` then converts the TOC into structured JSON. That output is not yet trusted. It still has to be aligned to the document pages, verified, and possibly repaired.

### Step 4: page alignment

In the TOC-with-page-numbers branch, `toc_index_extractor` looks at the first chunk of content after the TOC and tries to map section titles to actual physical page markers. Then `calculate_page_offset` computes the most likely offset between TOC page numbers and physical pages. That offset is applied to the whole tree, and `process_none_page_numbers` fills in gaps for items that never got a direct match.

### Step 5: validation and repair

After alignment, `validate_and_truncate_physical_indices` removes impossible page references. Then `verify_toc` checks whether each title really appears on the page where the model placed it. If the result is good enough but not perfect, `fix_incorrect_toc_with_retries` attempts to repair the bad entries by narrowing the search window around each one.

### Step 6: recursive processing for large nodes

`process_large_node_recursively` is the second pass over the tree. It looks for nodes that are too large in page and token count, then re-runs `meta_processor` on that node’s page slice in `process_no_toc` mode. This is how the code breaks a large document into nested subtrees instead of forcing the whole document through one giant tree.

## Branches in the PDF path

### Branch 1: TOC exists and already has page numbers

This is the preferred path because it reuses the author’s own outline instead of inferring structure from raw text.

`check_toc` calls `find_toc_pages` to scan the early pages for a TOC. If a TOC is found, `toc_extractor` checks whether the TOC contains page numbers.

If the answer is yes, `meta_processor(..., mode='process_toc_with_page_numbers')` runs:

- `toc_transformer` converts the raw TOC text into a structured JSON tree.
- `remove_page_number` strips the page field to produce a page-free copy.
- `toc_index_extractor` looks at a nearby slice of real pages and maps TOC entries to physical page markers.
- `calculate_page_offset` estimates the offset between TOC page numbers and real page numbers by comparing matched titles.
- `add_page_offset_to_toc_json` shifts the whole tree by that offset.
- `process_none_page_numbers` fills in any missing physical page indices by looking between neighboring known entries.

This branch is fast compared with the fallback branches because the TOC already provides most of the structure. The model still has to do a few alignment and verification passes, but those are much cheaper than inventing the hierarchy from scratch.

### Branch 2: TOC exists but does not have page numbers

If the document has a TOC but the TOC text does not include page numbers, `check_toc` takes the same raw TOC but returns `page_index_given_in_toc = 'no'`.

Then `meta_processor(..., mode='process_toc_no_page_numbers')` runs:

- `toc_transformer` first converts the TOC into structured JSON.
- The code then scans document pages and calls `add_page_number_to_toc` on grouped page text.
- The model tries to match TOC entries to the actual page markers in the page text.
- Missing page numbers are later normalized to integers.

This branch is slower because it has to do an extra pass over the body text to infer page placement. It is still better than no TOC, but it depends on the quality of the model’s title-to-page matching.

### Branch 3: No TOC found

If `find_toc_pages` returns nothing, `check_toc` prints `no toc found` and the pipeline falls back to `meta_processor(..., mode='process_no_toc')`.

That branch does not rely on a TOC at all.

- The document pages are wrapped with `<physical_index_X>` tags.
- Pages are grouped by token budget using `page_list_to_group_text`.
- `generate_toc_init` asks the model to invent the initial tree from the text.
- `generate_toc_continue` extends that tree across later page groups.
- The output is then converted into integer page indices.

This is the most expensive and least reliable branch because the model must infer structure from the body text alone. It can be useful for documents that have no usable outline, but it is inherently less stable.

## What the log lines mean

The log sequence you saw corresponds to the preferred TOC path:

- `start find_toc_pages` means the pipeline is scanning early pages to detect a TOC.
- `toc found` means a TOC page was detected.
- `start detect_page_index` means the code is checking whether the TOC already contains page numbers.
- `index found` means the TOC already has page numbers, so the pipeline can map TOC sections to actual pages.
- `process_toc_with_page_numbers` means the system is using the TOC-with-page-numbers branch.
- `start_index: 1` means the pipeline is indexing physical pages starting at 1.
- `start toc_transformer` means the raw TOC text is being cleaned into structured JSON.
- `start toc_index_extractor` means the code is aligning titles to physical page markers in nearby pages.
- `Document validation: 26 pages, max allowed index: 26` means the code checked that inferred page indices do not exceed the real document length.
- `start verify_toc` means the code is validating whether titles really appear on the pages they were assigned.
- `check all items` means it is checking every entry, not a sample.
- `accuracy: 92.31%` means the automatic title-vs-page verification passed for most entries, but not all.
- `start fix_incorrect_toc` and `start fix_incorrect_toc with 2 incorrect results` mean the repair loop has started for the two bad entries.

## What `verify_toc` does

`verify_toc` rechecks the output against the original page text.

- It finds the last non-`None` physical index.
- It verifies either all items or a sample, depending on `N`.
- For each entry it calls `check_title_appearance` to ask whether the title really appears on that page.
- It returns an accuracy score and a list of incorrect entries.

This is not just bookkeeping. It is a second model pass that catches alignment errors before the tree is finalized.

The code also short-circuits some cases. If the tree has no valid physical indices, or if the last valid page is too early relative to the document length, verification returns early rather than wasting calls on obviously bad output.

## What `fix_incorrect_toc` does

`fix_incorrect_toc_with_retries` is a repair loop around `fix_incorrect_toc`.

For each incorrect entry, the fixer:

1. Finds the nearest correct entry above it.
2. Finds the nearest correct entry below it.
3. Builds a page window between those boundaries.
4. Calls `single_toc_item_index_fixer` to re-predict the physical index of that one section.
5. Calls `check_title_appearance` again to verify the new guess.
6. Keeps the fix only if the verification passes.

If the repaired entry still looks wrong, it remains in the invalid list and the retry loop can run again, up to the configured maximum.

This exists because the TOC alignment step is probabilistic. The model can misplace a section by a page or two, especially when titles are repeated, OCR is noisy, or section starts are ambiguous.

It also matters because the repair step works on a narrower page window than the first alignment pass. That narrower context often makes the model more accurate on the second try.

## What `doc_id` is

`doc_id` is the internal UUID PageIndex assigns to each indexed document.

- It is created in `PageIndexClient.index()`.
- It is not the PDF filename.
- It is the key used to retrieve the indexed document from the client workspace and from the in-memory `documents` map.
- The saved workspace JSON files are stored under that ID.

In the demo script, the code either reuses a cached `doc_id` or creates a new one when indexing the PDF.

The `doc_id` is also what the retrieval tools use later, so the indexing phase and retrieval phase are joined through this identifier.

## Why it takes so long

The slow part is mostly model calls, not Python control flow.

The pipeline is expensive because it may do all of the following:

- Scan multiple early pages to find the TOC.
- Make one or more LLM calls to detect whether the TOC contains page numbers.
- Convert raw TOC text into JSON.
- Align TOC entries to real pages.
- Validate every entry against the page text.
- Retry incorrect entries with tighter page windows.
- Generate node summaries and a document description after the tree is built when those options are enabled.

The no-TOC branch is the slowest because the model has to invent structure from scratch instead of using an existing document outline.

Another source of slowness is repetition: the same text can be passed to multiple prompts during detection, transformation, alignment, verification, and repair. Each pass is solving a narrower problem, but they add up.

## Practical takeaway

For indexing, the best case is a clean PDF with a real TOC and page numbers. That path minimizes guesswork and avoids the expensive fallback branches.

If you want to inspect the exact prompts or tweak the behavior, start in `pageindex/page_index.py` and follow the functions above in order.

## Ideas for very large documents

If the tree is very large, a better retrieval strategy is to avoid handing the agent the entire structure up front.

- Give the agent only the top-level nodes first.
- Add tools that let it expand one node at a time, such as `get_children(node_id)` or `get_node_path(node_id)`.
- Add a `search_tree(query, node_id=None)` tool that can search only the current subtree.
- Let the agent recurse: choose a top-level branch, inspect its children, then descend only where needed.
- Return compact node metadata first, and fetch full text only after the agent has narrowed the scope.
- Cache expanded subtrees so repeated exploration does not require rereading the same structure.

That approach keeps the context window smaller, reduces prompt cost, and makes large-document retrieval feel more like deliberate tree search than bulk serialization.
