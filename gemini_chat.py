"""
Interactive chat with your PDF using Gemini
"""

import asyncio
import json
from gemini_pageindex import generate_pageindex, rag_query, get_page_tokens

async def chat_with_pdf(pdf_path, model="gemini-1.5-flash"):
    """Interactive chat session"""
    
    print("Loading PDF and generating index...")
    
    # Check for cached tree (try v2 first, then fallback to v1)
    cache_file_v2 = pdf_path.replace('.pdf', '_tree_v2.json')
    cache_file = pdf_path.replace('.pdf', '_tree.json')
    
    try:
        with open(cache_file_v2, 'r') as f:
            result = json.load(f)
        print(f"✓ Loaded cached tree from {cache_file_v2}")
    except:
        try:
            with open(cache_file, 'r') as f:
                result = json.load(f)
            print(f"✓ Loaded cached tree from {cache_file}")
        except:
            result = await generate_pageindex(
                pdf_path=pdf_path,
                model=model,
                add_text=True,
                max_pages_per_node=5,
                max_tokens_per_node=10000
            )
            # Save cache
            with open(cache_file_v2, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Generated and cached tree to {cache_file_v2}")
    
    pdf_pages = get_page_tokens(pdf_path)
    
    print(f"\n{'='*60}")
    print(f"Chat with: {result['doc_name']}")
    print(f"Pages: {result['total_pages']}, Model: {model}")
    print(f"{'='*60}")
    print("Type your questions (or 'quit' to exit)\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("\n🤔 Thinking...")
        
        try:
            answer_result = await rag_query(
                query=query,
                tree=result['structure'],
                pdf_pages=pdf_pages,
                model=model
            )
            
            print(f"\n🤖 PageIndex:\n{answer_result['answer']}\n")
            print(f"📍 Sources: Nodes {', '.join(answer_result['relevant_nodes'])}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gemini_chat.py <pdf_path> [model]")
        print("Example: python gemini_chat.py ./report.pdf gemini-1.5-flash")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gemini-1.5-flash"
    
    asyncio.run(chat_with_pdf(pdf_path, model))
