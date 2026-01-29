"""
Quick test of RedBook queries with v2 tree
"""

import asyncio
import json
from gemini_pageindex import rag_query, get_page_tokens

async def test_queries():
    # Load v2 tree
    with open('C:/Users/akash/Desktop/rovabot-dev/Rovabot/RedBook_tree_v2.json', 'r') as f:
        result = json.load(f)
    
    pdf_pages = get_page_tokens('C:/Users/akash/Desktop/rovabot-dev/Rovabot/RedBook.pdf')
    
    print(f"Loaded tree: {result['doc_name']}")
    print(f"Pages: {result['total_pages']}, Tokens: {result['total_tokens']:,}")
    
    # Count nodes
    def count_nodes(tree):
        count = 0
        if isinstance(tree, list):
            for node in tree:
                count += count_nodes(node)
        elif isinstance(tree, dict):
            count = 1
            if 'nodes' in tree:
                count += count_nodes(tree['nodes'])
        return count
    
    total_nodes = sum(count_nodes(node) for node in result['structure'])
    print(f"Total nodes: {total_nodes}")
    
    # Test queries
    queries = [
        "What is gallons per ft for 10.75\" Casing in 13\" hole?",
        "What is cu ft per Lin ft for 14.5\" hole and 11.75\" casing?",
        "What is the coiled tubing collapse pressure for 1in diameter and 0.080 in wall thickness?"
    ]
    
    print(f"\n{'='*70}")
    print("Testing queries...")
    print('='*70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}]: {query}")
        print("-" * 70)
        
        try:
            answer_result = await rag_query(
                query=query,
                tree=result['structure'],
                pdf_pages=pdf_pages,
                model="gemini-2.0-flash"
            )
            
            print(f"Answer: {answer_result['answer']}")
            print(f"Sources: Nodes {', '.join(answer_result['relevant_nodes'])}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print()

if __name__ == "__main__":
    asyncio.run(test_queries())
