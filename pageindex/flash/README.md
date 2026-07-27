# PageIndex Flash

Builds a PageIndex tree structure from a PDF using layout statistics alone.
No LLM, no API key, no OCR, no network. Runs in seconds, fully offline.

## Usage

```python
from pageindex.flash import page_index_flash

tree = page_index_flash("paper.pdf")
```

```bash
python3 run_pageindex.py --pdf_path document.pdf --flash
```

Accepts a path (`str` or `pathlib.Path`) or an `io.BytesIO` stream. Raises on a
missing, non-PDF, encrypted, empty, or unreadable file.

## Output

```python
{
    "doc_name": str,
    "doc_title": str,
    "structure": [
        {"title": str, "start_index": int, "end_index": int, "nodes": [...]}
    ],
}
```

Page indexes are 1-based. `nodes` nests recursively.

## Limits

- Scanned PDFs without embedded text are not supported.
- Encrypted PDFs need preprocessing first.
- Headings drawn as vector paths, or very decorative layouts, can be missed.
- Titles are taken from the document text as-is.

## Dependencies

`pypdfium2`, `PyPDF2`, `regex`, `sortedcontainers`.
