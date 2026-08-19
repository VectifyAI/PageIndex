import argparse
import os
import json
from pageindex import *
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader, _openai_missing_keys

# Keep LiteLLM's import off the network (frozen bundled model-cost map);
# an explicit user setting wins.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')
    parser.add_argument('--mode', choices=['flash', 'standard'], default='flash',
                      help='Processing mode (default: flash)')
    parser.add_argument('--flash', action='store_true', default=False,
                      help=argparse.SUPPRESS)
    parser.add_argument('--embedded-toc', action=argparse.BooleanOptionalAction, default=None,
                      help='Use the PDF\'s embedded bookmarks when trustworthy (default: on in flash mode)')
    parser.add_argument('--summary', action=argparse.BooleanOptionalAction, default=None,
                      help='Generate node summaries with an LLM (default: on in flash mode)')
    parser.add_argument('--optimize', nargs='?', const='full', choices=['full', 'merge', 'off'],
                      default=None,
                      help='Refine the tree for search cost (default: full in flash mode). '
                           '`merge` for deterministic merge only; `off` to disable')

    parser.add_argument('--index-model', type=str, default=None,
                      help='Model used to index the document (overrides config.yaml)')
    parser.add_argument('--model', type=str, default=None,
                      help='(legacy) Same as --index-model')
    parser.add_argument('--summary-model', type=str, default=None,
                      help='Model for node summaries (defaults to --index-model, then --model, then config.yaml)')

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
    if args.flash:
        args.mode = 'flash'

    # Validate that exactly one file type is specified
    if not args.pdf_path and not args.md_path:
        raise ValueError("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")
    if args.optimize is not None and not (args.pdf_path and args.mode == 'flash'):
        raise ValueError("--optimize requires Flash mode with --pdf_path")
    if args.optimize is None:
        args.optimize = 'full' if args.mode == 'flash' else 'off'
    if args.embedded_toc is not None and not (args.pdf_path and args.mode == 'flash'):
        raise ValueError("--embedded-toc requires Flash mode with --pdf_path")
    if args.summary is not None and not (args.pdf_path and args.mode == 'flash'):
        raise ValueError("--summary requires Flash mode with --pdf_path")
    if args.pdf_path and args.mode == 'flash':
        for flag, value in (('--toc-check-pages', args.toc_check_pages),
                            ('--max-pages-per-node', args.max_pages_per_node),
                            ('--max-tokens-per-node', args.max_tokens_per_node),
                            ('--if-add-node-id', args.if_add_node_id),
                            ('--if-add-node-summary', args.if_add_node_summary),
                            ('--if-add-doc-description', args.if_add_doc_description),
                            ('--if-add-node-text', args.if_add_node_text)):
            if value is not None:
                raise ValueError(f"{flag} is not supported in flash mode; use --mode standard")

    if args.pdf_path:
        # Validate PDF file
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")
            
        if args.mode == 'flash':
            from pageindex.flash import page_index_flash
            summary_model = (args.summary_model or args.index_model
                             or args.model
                             or ConfigLoader().load().summary_model)
            will_summarize = args.summary if args.summary is not None else True
            if will_summarize or args.optimize == 'full':
                missing = _openai_missing_keys(summary_model)
                if missing:
                    raise SystemExit(
                        f"Missing API key for {summary_model}: {', '.join(missing)}")
            toc_with_page_number = page_index_flash(
                args.pdf_path,
                optimize=args.optimize if args.optimize != 'off' else False,
                optimize_model=summary_model,
                summary_model=summary_model,
                use_embedded_toc=args.embedded_toc if args.embedded_toc is not None else True,
                summary=will_summarize,
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
                'index_model': args.index_model,
                'model': args.model,
                'summary_model': args.summary_model,
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
        suffix = '_structure'
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
            'index_model': args.index_model,
            'model': args.model,
        }
        
        # Load config with defaults from config.yaml
        opt = config_loader.load({k: v for k, v in user_opt.items() if v is not None})
        
        # if_add_* pass through as given (absent = off, as before this CLI
        # used config.yaml): the PDF defaults there must not switch on LLM
        # passes the markdown CLI never ran.
        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=args.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=args.if_add_doc_description,
            if_add_node_text=args.if_add_node_text,
            if_add_node_id=args.if_add_node_id
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
