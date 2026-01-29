# PageIndex Gemini API - Quick Reference

## Installation ✅
```bash
pip install google-generativeai
```

## Files
| File | Purpose |
|------|---------|
| `gemini_pageindex.py` | Main implementation |
| `gemini_rag_demo.py` | Full RAG example |
| `gemini_chat.py` | Interactive chat |
| `test_gemini.py` | Test setup |
| `.env` | API key |

## Quick Commands

### Generate Tree
```bash
python gemini_pageindex.py --pdf document.pdf
python gemini_pageindex.py --pdf document.pdf --summaries  # With summaries
python gemini_pageindex.py --pdf document.pdf --model gemini-2.5-pro  # Pro model
```

### Interactive Chat
```bash
python gemini_chat.py document.pdf
```

### Test Setup
```bash
python test_gemini.py
```

## Python API

### Generate Index
```python
from gemini_pageindex import generate_pageindex
import asyncio

async def main():
    result = await generate_pageindex("./doc.pdf", add_summaries=True)
    return result

asyncio.run(main())
```

### Query with RAG
```python
from gemini_pageindex import rag_query, get_page_tokens
import asyncio

async def main():
    pages = get_page_tokens("./doc.pdf")
    answer = await rag_query(
        "Your question here",
        tree_structure,
        pages
    )
    print(answer['answer'])

asyncio.run(main())
```

## Models Available
- `gemini-2.0-flash` ⚡ Default, fastest, cheapest
- `gemini-2.5-pro` 🎯 Best quality, higher cost
- `gemini-2.5-flash` Latest Flash version
- `gemini-pro-latest` Latest Pro version

## Cost
| Model | 50-Page PDF |
|-------|-----------|
| Flash | $0.15-0.30 ✅ |
| Pro | $1.50-3.00 |
| GPT-4o | $3-5 (10x more!) |

## Features
- ✅ Tree generation from PDFs
- ✅ Intelligent document retrieval
- ✅ Answer generation with sources
- ✅ Caching for performance
- ✅ Interactive chat interface
- ✅ Batch processing support

## Status
✅ API Key configured  
✅ Dependencies installed  
✅ Ready to use!

Start with: `python test_gemini.py`
