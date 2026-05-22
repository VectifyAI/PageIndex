# API Reference

This document provides API reference documentation for the PageIndex package.

## Table of Contents

- [Core Functions](#core-functions)
  - [PDF Processing](#pdf-processing)
  - [Table of Contents](#table-of-contents)
  - [Tree Structure](#tree-structure)
  - [LLM Integration](#llm-integration)
- [Utilities](#utilities)

---

## Core Functions

### PDF Processing

#### `extract_text_from_pdf(pdf_path)`

Extracts all text from a PDF document.

**Parameters:**
- `pdf_path` (str): Path to the PDF file

**Returns:**
- `str`: Combined text from all pages

```python
from pageindex.utils import extract_text_from_pdf

text = extract_text_from_pdf("/path/to/document.pdf")
```

---

#### `get_text_of_pages(pdf_path, start_page, end_page, tag=True)`

Extracts text from a specific range of pages with optional tagging.

**Parameters:**
- `pdf_path` (str): Path to the PDF file
- `start_page` (int): Starting page number (1-indexed)
- `end_page` (int): Ending page number (1-indexed)
- `tag` (bool): Whether to add `<start_index_X>` and `<end_index_X>` tags (default: True)

**Returns:**
- `str`: Text content from the specified page range

```python
text = get_text_of_pages("/path/to/document.pdf", start_page=1, end_page=10)
```

---

#### `get_pdf_title(pdf_path)`

Extracts the title from PDF metadata.

**Parameters:**
- `pdf_path` (str): Path to the PDF file

**Returns:**
- `str`: Document title or 'Untitled' if not available

```python
title = get_pdf_title("/path/to/document.pdf")
```

---

#### `get_page_tokens(pdf_path, model=None, pdf_parser="PyPDF2")`

Gets page text and token counts for all pages in a PDF.

**Parameters:**
- `pdf_path` (str or BytesIO): Path to PDF file or BytesIO object
- `model` (str): LLM model name for token counting (default: None)
- `pdf_parser` (str): PDF parser to use - "PyPDF2" or "PyMuPDF" (default: "PyPDF2")

**Returns:**
- `list`: List of tuples `[(page_text, token_count), ...]`

```python
pages = get_page_tokens("/path/to/document.pdf", model="gpt-4o")
```

---

### Table of Contents

#### `toc_detector_single_page(content, model=None)`

Detects if a page contains a table of contents.

**Parameters:**
- `content` (str): Page text content
- `model` (str): LLM model to use (default: None)

**Returns:**
- `str`: "yes" if TOC detected, "no" otherwise

```python
has_toc = toc_detector_single_page(page_text, model="gpt-4o")
```

---

#### `extract_toc_content(content, model=None)`

Extracts the full table of contents from text.

**Parameters:**
- `content` (str): Text containing the table of contents
- `model` (str): LLM model to use (default: None)

**Returns:**
- `str`: Extracted table of contents content

```python
toc_content = extract_toc_content(text, model="gpt-4o")
```

---

#### `toc_transformer(toc_content, model=None)`

Transforms raw table of contents into structured JSON format.

**Parameters:**
- `toc_content` (str): Raw table of contents text
- `model` (str): LLM model to use (default: None)

**Returns:**
- `list`: List of TOC entries with structure, title, and page fields

```python
toc_json = toc_transformer(raw_toc, model="gpt-4o")
# Returns: [{"structure": "1", "title": "Introduction", "page": 1}, ...]
```

---

#### `toc_index_extractor(toc, content, model=None)`

Adds physical indices to table of contents entries.

**Parameters:**
- `toc` (list): Table of contents in JSON format
- `content` (str): Document pages content with physical_index tags
- `model` (str): LLM model to use (default: None)

**Returns:**
- `list`: TOC entries with added physical_index fields

```python
toc_with_indices = toc_index_extractor(toc_json, document_content, model="gpt-4o")
```

---

#### `find_toc_pages(start_page_index, page_list, opt, logger=None)`

Finds all pages that contain table of contents content.

**Parameters:**
- `start_page_index` (int): Page index to start searching from
- `page_list` (list): List of page tuples `[(text, tokens), ...]`
- `opt`: Configuration object with `model` and `toc_check_page_num` attributes
- `logger` (JsonLogger): Optional logger instance

**Returns:**
- `list`: List of page indices containing TOC content

```python
toc_pages = find_toc_pages(1, page_list, config)
```

---

### Tree Structure

#### `list_to_tree(data)`

Converts a flat list of TOC entries into a hierarchical tree structure.

**Parameters:**
- `data` (list): Flat list of TOC entries with structure indices

**Returns:**
- `list`: Hierarchical tree structure with nested `nodes` arrays

```python
tree = list_to_tree(flat_toc_list)
# Returns nested structure with "nodes" children
```

---

#### `get_leaf_nodes(structure)`

Extracts all leaf nodes from a tree structure.

**Parameters:**
- `structure` (dict or list): Tree structure

**Returns:**
- `list`: List of leaf node dictionaries

```python
leaves = get_leaf_nodes(tree)
```

---

#### `get_nodes(structure)`

Extracts all nodes from a tree structure (including internal nodes).

**Parameters:**
- `structure` (dict or list): Tree structure

**Returns:**
- `list`: List of all node dictionaries

```python
all_nodes = get_nodes(tree)
```

---

#### `add_preface_if_needed(data)`

Adds a "Preface" entry if the document starts after page 1.

**Parameters:**
- `data` (list): List of TOC entries

**Returns:**
- `list`: Updated list with Preface entry if needed

```python
toc_with_preface = add_preface_if_needed(toc_data)
```

---

### LLM Integration

#### `llm_completion(model, prompt, chat_history=None, return_finish_reason=False)`

Sends a completion request to the configured LLM.

**Parameters:**
- `model` (str): Model name (e.g., "gpt-4o")
- `prompt` (str): User prompt text
- `chat_history` (list): Optional list of previous message dicts
- `return_finish_reason` (bool): Whether to return finish reason (default: False)

**Returns:**
- `str` or `tuple`: Response content, or `(content, finish_reason)` if `return_finish_reason=True`

```python
response = llm_completion("gpt-4o", "Extract the main topics.")
content, reason = llm_completion("gpt-4o", prompt, chat_history=history, return_finish_reason=True)
```

---

#### `llm_acompletion(model, prompt)`

Sends an async completion request to the configured LLM.

**Parameters:**
- `model` (str): Model name
- `prompt` (str): User prompt text

**Returns:**
- `str`: Response content

```python
response = await llm_acompletion("gpt-4o", "Check if section appears on this page.")
```

---

#### `count_tokens(text, model=None)`

Counts the number of tokens in text using the specified model.

**Parameters:**
- `text` (str): Text to count tokens for
- `model` (str): Model name for token counting

**Returns:**
- `int`: Token count

```python
num_tokens = count_tokens(long_text, model="gpt-4o")
```

---

## Utilities

#### `extract_json(content)`

Extracts and parses JSON from LLM response text.

**Parameters:**
- `content` (str): LLM response text containing JSON

**Returns:**
- `dict` or `list`: Parsed JSON content

```python
data = extract_json(llm_response)
```

---

#### `JsonLogger`

Logger class for storing processing results in JSON format.

**Methods:**
- `info(message, **kwargs)`: Log info level message
- `error(message, **kwargs)`: Log error level message
- `debug(message, **kwargs)`: Log debug level message
- `exception(message, **kwargs)`: Log exception with traceback

**Usage:**

```python
logger = JsonLogger("/path/to/document.pdf")
logger.info("Processing started")
logger.info({"section": "introduction", "pages": 5})
logger.error("Failed to parse page")
```

---

#### `print_toc(tree, indent=0)`

Prints a tree structure as an indented text outline.

**Parameters:**
- `tree` (list): Tree structure to print
- `indent` (int): Initial indentation level (default: 0)

```python
print_toc(tree)
```

---

#### `print_json(data, max_len=40, indent=2)`

Prints JSON data with automatic truncation of long strings.

**Parameters:**
- `data` (dict or list): Data to print
- `max_len` (int): Maximum string length before truncation (default: 40)
- `indent` (int): JSON indentation spaces (default: 2)

```python
print_json(result)
```

---

#### `sanitize_filename(filename, replacement='-')`

Sanitizes a string to be used as a valid filename.

**Parameters:**
- `filename` (str): Original filename
- `replacement` (str): Character to replace invalid characters with (default: '-')

**Returns:**
- `str`: Sanitized filename

```python
safe_name = sanitize_filename("report/2024.pdf")  # "report-2024.pdf"
```

---

## Configuration

The package uses a `config.yaml` file for default settings. Key configuration options:

- `model`: Default LLM model (default: "gpt-4o-2024-11-20")
- `toc_check_page_num`: Maximum pages to check for TOC (default: 20)
- `max_pages_per_node`: Maximum pages per tree node (default: 10)
- `max_tokens_per_node`: Maximum tokens per node (default: 20000)

Load configuration using:

```python
from pageindex.utils import config
```

---

## Async Functions

PageIndex includes async variants of key functions for concurrent processing:

- `check_title_appearance()` - Check if a title appears on a page
- `check_title_appearance_in_start()` - Check if a title starts at page beginning
- `check_title_appearance_in_start_concurrent()` - Concurrent title checking for all sections

```python
import asyncio

async def process_titles():
    results = await check_title_appearance_in_start_concurrent(
        structure=items,
        page_list=pages,
        model="gpt-4o",
        logger=logger
    )
```