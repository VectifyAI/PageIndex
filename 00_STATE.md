# 00_STATE.md — PageIndex Repository Analysis

## Repository Identity
- **Upstream**: VectifyAI/PageIndex (source)
- **Fork**: okwn/PageIndex (working copy at /root/oss-pr-campaign/repos/pageindex)
- **License**: MIT
- **Archived**: No
- **Language**: Python

## Repository Statistics (Upstream)
- **Stars**: 31,969
- **Forks**: 2,754
- **Open Issues**: 30 (of 158 total)
- **Open PRs**: 3 (dependency bumps)
- **Watchers**: 31,969
- **Default Branch**: main

## Repository Structure
```
pageindex/
├── pageindex/               # Main package
│   ├── __init__.py          # Exports: page_index, md_to_tree, retrieve functions, PageIndexClient
│   ├── page_index.py        # PDF indexing logic (~1150 lines, async LLM-driven)
│   ├── page_index_md.py     # Markdown indexing logic (~340 lines)
│   ├── client.py            # PageIndexClient workspace-based API (~234 lines)
│   ├── retrieve.py          # Document/page retrieval helpers (~137 lines)
│   ├── utils.py             # LLM wrappers, token counting, tree utilities (~710 lines)
│   └── config.yaml          # Default config (gpt-4o-2024-11-20 model, various limits)
├── run_pageindex.py         # CLI entry point for PDF/MD processing
├── requirements.txt         # Dependencies (litellm, pymupdf, PyPDF2, python-dotenv, pyyaml)
├── examples/
│   ├── agentic_vectorless_rag_demo.py
│   ├── documents/
│   │   ├── q1-fy25-earnings.pdf
│   │   ├── four-lectures.pdf
│   │   ├── earthmover.pdf
│   │   └── [other PDFs]
│   │   └── results/         # Pre-generated tree structures
│   └── tutorials/
├── .github/
│   ├── workflows/           # CI: codeql, dependency-review, autoclose, dedupe
│   ├── scripts/             # autoclose-labeled-issues.js, comment-on-duplicates.sh
│   └── dependabot.yml      # Weekly GitHub Actions dependency updates
└── README.md               # Full documentation with examples

```

## Key Upstream Branches
- `main` — stable release
- `dev` — development work
- `feat/markdown-tree`, `feat/md-bold-heading-recognition` — feature branches
- `fix/cloud-poll-status-completed`, `add-pypdfium2-parser` — fix branches
- `dependabot/*` — automated dependency updates

## Current Working Branch
- **Local main** is tracking `upstream/main`
- Fork created via `gh api --method POST repos/VectifyAI/PageIndex/forks`

## Installation
```bash
pip3 install --upgrade -r requirements.txt
# Optional: openai-agents for examples/agentic_vectorless_rag_demo.py
```

## Core Functionality Summary
PageIndex is a **vectorless, reasoning-based RAG** system that:
1. Builds a hierarchical tree index (ToC-style) from PDFs or markdown
2. Uses LLMs to reason over the tree for context-aware retrieval
3. Achieved 98.7% accuracy on FinanceBench (Mafin 2.5 system)

## Package Usage
```bash
# PDF processing
python3 run_pageindex.py --pdf_path /path/to/document.pdf

# Markdown processing
python3 run_pageindex.py --md_path /path/to/document.md

# Via Python API
from pageindex import PageIndexClient
client = PageIndexClient(api_key="...")
doc_id = client.index("document.pdf")
print(client.get_document_structure(doc_id))
```

## No Test Suite Found
- No pytest, unittest, or test files present in the repository
- No CI workflow for running tests

## CI/CD
- **CodeQL**: Security analysis on push/PR to main
- **Dependency Review**: Scans dependency changes on PRs
- **Dependabot**: Weekly GitHub Actions updates (actions/checkout, dependency-review-action, github-script)
- **Autoclose**: Auto-closes issues with specific labels
- **Dedupe**: Issue deduplication workflow

## Health Indicators
- Active upstream (31k stars, 2.7k forks, 158 issues)
- Regular maintenance via Dependabot
- Multiple active branches for features/fixes
- No test suite — notable gap for OSS contribution
- 3 open dependency PRs (unmerged)