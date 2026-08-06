# PageIndex Flash

Builds the PageIndex tree structure from a PDF using layout statistics without
an LLM. Augmenting the tree with summaries and refining it for retrieval needs
an LLM.

## Usage

### Python

```python
from pageindex.flash import page_index_flash

tree = page_index_flash("paper.pdf")
tree = page_index_flash("paper.pdf", summary=False)   # tree structure only, no LLM
tree = page_index_flash("paper.pdf", optimize=True)   # refined tree for retrieval
```

Takes a file path or an `io.BytesIO` stream and returns the tree as a dict.
Summaries are on by default and need an LLM API key.

### Command line

```bash
python3 run_pageindex.py --pdf_path document.pdf --flash
python3 run_pageindex.py --pdf_path document.pdf --flash --no-summary  # tree structure only, no LLM
python3 run_pageindex.py --pdf_path document.pdf --flash --optimize    # refined tree for retrieval
```

Writes the tree to `results/<name>_structure_flash.json`.

## Output

```python
{
    "doc_name": str,
    "doc_title": str,
    "structure": [
        {
            "title": str,
            "node_id": str,       # 4-digit, zero-padded
            "start_index": int,   # 1-based, inclusive
            "end_index": int,
            "summary": str,
            "key_items": [str],   # optimize only: titles of subsections merged away
            "nodes": [...],
        }
    ],
}
```

## Benchmark

Nine PDFs, each run end to end with tree optimization: PDF parse, layout
outline, merge, LLM expand, then a summary for every node.

<img src="assets/time_vs_pages.png" alt="Time against document length" width="50%">

| Document | Pages | Input tokens | Output tokens |
|---|---:|---:|---:|
| Bitcoin whitepaper | 9 | 8,715 | 4,673 |
| Attention Is All You Need | 15 | 26,805 | 10,183 |
| KIMI K3 | 47 | 85,704 | 35,217 |
| DeepSeek-R1 | 86 | 68,398 | 26,351 |
| Situational Awareness | 165 | 115,130 | 54,347 |
| Federal Reserve 2023 report | 222 | 280,975 | 136,982 |
| 9/11 Commission Report | 585 | 720,624 | 200,202 |
| Pattern Recognition and Machine Learning | 758 | 857,983 | 277,675 |
| Machine Learning: A Probabilistic Perspective | 1,098 | 1,587,265 | 646,958 |
