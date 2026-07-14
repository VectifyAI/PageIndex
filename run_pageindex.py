import argparse
import os
import json
from pageindex.index.page_index import *
# Reuse the canonical yes/no coercion (as _cli_bool) instead of a second copy —
# a bare ``--flag`` (no value) resolves to True via argparse's ``const``; an
# explicit value keeps the legacy yes/no style working, so ``--flag no`` turns
# it off. argparse only ever passes a str here (const/default bypass type=).
from pageindex.index.page_index_md import md_to_tree, _coerce_bool as _cli_bool
from pageindex.config import IndexConfig


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

    parser.add_argument('--model', type=str, default=None, help='Model to use')

    parser.add_argument('--toc-check-pages', type=int, default=None,
                      help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=None,
                      help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=None,
                      help='Maximum number of tokens per node (PDF only)')

    # Bare flag (e.g. --if-add-node-id) turns the option on; an explicit value
    # keeps the legacy yes/no style, so --if-add-node-id no turns it off.
    parser.add_argument('--if-add-node-id', nargs='?', const=True, type=_cli_bool, default=None,
                      help='Add node IDs (on by default). Bare flag or yes/no, e.g. --if-add-node-id no')
    parser.add_argument('--if-add-node-summary', nargs='?', const=True, type=_cli_bool, default=None,
                      help='Add node summaries (on by default). Bare flag or yes/no')
    parser.add_argument('--if-add-doc-description', nargs='?', const=True, type=_cli_bool, default=None,
                      help='Add a document description (off by default). Bare flag or yes/no')
    parser.add_argument('--if-add-node-text', nargs='?', const=True, type=_cli_bool, default=None,
                      help='Add raw text to nodes (off by default). Bare flag or yes/no')

    # Markdown specific arguments
    parser.add_argument('--if-thinning', nargs='?', const=True, type=_cli_bool, default=None,
                      help='Apply tree thinning (off by default, markdown only). Bare flag or yes/no')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    args = parser.parse_args()

    # Validate that exactly one file type is specified
    if not args.pdf_path and not args.md_path:
        raise ValueError("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")

    # Build IndexConfig from CLI args (None values use defaults)
    config_overrides = {
        k: v for k, v in {
            "model": args.model,
            "toc_check_page_num": args.toc_check_pages,
            "max_page_num_each_node": args.max_pages_per_node,
            "max_token_num_each_node": args.max_tokens_per_node,
            "if_add_node_id": args.if_add_node_id,
            "if_add_node_summary": args.if_add_node_summary,
            "if_add_doc_description": args.if_add_doc_description,
            "if_add_node_text": args.if_add_node_text,
        }.items() if v is not None
    }
    # Legacy config.yaml is the base when present, CLI args win
    try:
        opt = IndexConfig.from_yaml(**config_overrides)
    except FileNotFoundError:
        opt = IndexConfig(**config_overrides)

    if args.pdf_path:
        # Validate PDF file
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")

        # Process the PDF
        toc_with_page_number = page_index_main(args.pdf_path, opt)
        print('Parsing done, saving to file...')

        # Save results
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_dir = './results'
        output_file = f'{output_dir}/{pdf_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2)

        print(f'Tree structure saved to: {output_file}')

    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")

        # Process markdown file
        print('Processing markdown file...')
        import asyncio

        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=bool(args.if_thinning),
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id
        ))

        print('Parsing done, saving to file...')

        # Save results
        md_name = os.path.splitext(os.path.basename(args.md_path))[0]
        output_dir = './results'
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)

        print(f'Tree structure saved to: {output_file}')
