import argparse
import os
import json
from pageindex import *
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')
    parser.add_argument('--flash', action='store_true', help='Use PageIndex Flash (with --pdf_path)')
    parser.add_argument('--embedded-toc', action=argparse.BooleanOptionalAction, default=None,
                      help='Use the PDF\'s embedded bookmarks when trustworthy (default: on with --flash)')
    parser.add_argument('--summary', action=argparse.BooleanOptionalAction, default=None,
                      help='Generate node summaries with an LLM (default: on with --flash)')
    parser.add_argument('--optimize', nargs='?', const='full', choices=['full', 'merge'],
                      default=None,
                      help='Refine the tree for search cost: a deterministic merge, then an '
                           'LLM expansion pass; pass `merge` to run the merge alone (PDF only)')

    parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config.yaml)')
    parser.add_argument('--summary-model', type=str, default=None,
                      help='Model for node summaries (defaults to --model, then config.yaml)')

    parser.add_argument('--toc-check-pages', type=int, default=None,
                      help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=None,
                      help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=None,
                      help='Maximum number of tokens per node (PDF only)')

    parser.add_argument('--if-add-node-id', type=str, default=None,
                      help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default=None,
                      help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default=None,
                      help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-node-text', type=str, default=None,
                      help='Whether to add text to the node')
                      
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
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
    if args.optimize and not (args.pdf_path and args.flash):
        raise ValueError("--optimize requires --flash with --pdf_path")
    if args.embedded_toc is not None and not (args.pdf_path and args.flash):
        raise ValueError("--embedded-toc requires --flash with --pdf_path")
    if args.summary is not None and not (args.pdf_path and args.flash):
        raise ValueError("--summary requires --flash with --pdf_path")

    if args.pdf_path:
        # Validate PDF file
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")
            
        if args.flash:
            from pageindex.flash import page_index_flash
            if args.optimize == 'full':
                from pageindex.tree_optimize import default_model
                from pageindex.utils import _is_openai_model
                expand_model = args.model or default_model()
                if _is_openai_model(expand_model) and not os.getenv("OPENAI_API_KEY"):
                    raise SystemExit(f"OPENAI_API_KEY is not set (expand model: {expand_model}).")
            toc_with_page_number = page_index_flash(
                args.pdf_path,
                optimize=args.optimize is not None,
                optimize_expand=args.optimize == 'full',
                optimize_model=args.model,
                summary_model=args.summary_model or args.model,
                use_embedded_toc=args.embedded_toc if args.embedded_toc is not None else True,
                summary=args.summary if args.summary is not None else True,
            )
            if 'optimize' in toc_with_page_number:
                o = toc_with_page_number['optimize']
                print(f"Optimize: merges={o['merges']} expands={o['expands']}, "
                      f"worst-case search cost "
                      f"{o['before'].get('worst_case_search_complexity')} -> "
                      f"{o['after'].get('worst_case_search_complexity')} pages")
        else:
            # Process PDF file
            user_opt = {
                'model': args.model,
                'toc_check_page_num': args.toc_check_pages,
                'max_page_num_each_node': args.max_pages_per_node,
                'max_token_num_each_node': args.max_tokens_per_node,
                'if_add_node_id': args.if_add_node_id,
                'if_add_node_summary': args.if_add_node_summary,
                'if_add_doc_description': args.if_add_doc_description,
                'if_add_node_text': args.if_add_node_text,
            }
            opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})
            toc_with_page_number = page_index_main(args.pdf_path, opt)

        print('Parsing done, saving to file...')

        # Save results
        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
        suffix = '_structure_flash' if args.flash else '_structure'
        output_dir = './results'
        output_file = f'{output_dir}/{pdf_name}{suffix}.json'
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)

        print(f'Tree structure saved to: {output_file}')
            
    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")
            
        # Process markdown file
        print('Processing markdown file...')
        
        # Process the markdown
        import asyncio
        
        # Use ConfigLoader to get consistent defaults (matching PDF behavior)
        from pageindex.utils import ConfigLoader
        config_loader = ConfigLoader()
        
        # Create options dict with user args
        user_opt = {
            'model': args.model,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_node_id': args.if_add_node_id
        }
        
        # Load config with defaults from config.yaml
        opt = config_loader.load(user_opt)
        
        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
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
