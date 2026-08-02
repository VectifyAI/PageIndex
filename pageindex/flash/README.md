# PageIndex Flash

Builds the PageIndex tree structure from a PDF using layout statistics without
LLM. Augmenting the tree with summaries and refining it for retrieval needs an
LLM.

## Usage

### Python

```python
from pageindex.flash import page_index_flash

tree = page_index_flash("paper.pdf", summary=False)   # structure only
tree = page_index_flash("paper.pdf")                  # + a summary per node
tree = page_index_flash("paper.pdf", optimize=True)   # + retrieval refinement
```

Takes a file path or an `io.BytesIO` stream and returns the tree as a dict.
Summaries are on by default and need an API key.

### Command line

```bash
python3 run_pageindex.py --pdf_path document.pdf --flash
python3 run_pageindex.py --pdf_path document.pdf --flash --optimize
```

Writes the tree to `results/<name>_structure_flash.json`.

## Output

```python
{
    "doc_name": str,
    "doc_title": str,
    "has_abstract_or_references_section": bool,
    "structure": [
        {
            "title": str,
            "node_id": str,       # 4-digit, zero-padded
            "start_index": int,   # 1-based, inclusive
            "end_index": int,
            "summary": str,       # with summary
            "key_items": [str],   # with optimize: titles merged away
            "nodes": [...],       # absent on leaves
        }
    ],
}
```

## Benchmark

Nine PDFs, each run end to end with tree optimization: PDF parse, layout
outline, merge, LLM expand, then a summary for every node.

![Time against document length](assets/time_vs_pages.png)

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
| **Total** | **2,985** | **3,751,599** | **1,392,588** |

Measured with `gpt-5.6-luna`.
