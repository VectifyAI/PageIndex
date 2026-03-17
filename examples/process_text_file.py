#!/usr/bin/env python3
"""
Example: Processing .txt files with PageIndex

This example demonstrates how to process plain text files with automatic
section detection using the PageIndex framework.
"""

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pageindex.page_index_txt import txt_to_tree
from pageindex.utils import print_toc, print_json

async def main():
    # Configuration
    TXT_PATH = 'tests/texts/sample.txt'
    MODEL = 'gpt-4o-2024-11-20'
    WINDOW_SIZE = 5000
    OVERLAP = 500
    
    print(f"Processing: {TXT_PATH}")
    print(f"Window size: {WINDOW_SIZE}, Overlap: {OVERLAP}\n")
    
    result = await txt_to_tree(
        txt_path=TXT_PATH,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        if_add_node_summary='yes',
        summary_token_threshold=200,
        model=MODEL,
        if_add_doc_description='yes',
        if_add_node_text='no',
        if_add_node_id='yes',
        max_input_tokens=25000
    )
    
    print('='*60)
    print('TABLE OF CONTENTS')
    print('='*60)
    print_toc(result['structure'])
    
    print('\n' + '='*60)
    print('STRUCTURE')
    print('='*60)
    print_json(result, max_len=100)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("CHATGPT_API_KEY"):
        print("ERROR: Set CHATGPT_API_KEY in .env file")
        exit(1)
    
    asyncio.run(main())
