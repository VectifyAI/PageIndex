"""
Complete RAG demo with Gemini API
"""

import asyncio
import json
from gemini_pageindex import generate_pageindex, rag_query, get_page_tokens

async def main():
    # Configuration
    PDF_PATH = "./my_document.pdf"  # Change this to your PDF
    MODEL = "gemini-1.5-flash"  # or "gemini-1.5-pro"
    
    # Step 1: Generate PageIndex (one-time, cache the result)
    print("="*60)
    print("STEP 1: Generating PageIndex Tree")
    print("="*60)
    
    result = await generate_pageindex(
        pdf_path=PDF_PATH,
        model=MODEL,
        add_summaries=False,  # Set True for better quality
        add_text=True  # Need text for RAG
    )
    
    # Save tree for reuse
    with open('tree_cache.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Generated tree with {len(result['structure'])} top-level sections")
    
    # Extract pages for later use
    pdf_pages = get_page_tokens(PDF_PATH)
    
    # Step 2: Ask questions
    print("\n" + "="*60)
    print("STEP 2: Asking Questions")
    print("="*60)
    
    questions = [
        "What are the main conclusions?",
        "What is the revenue?",
        "What are the key findings?",
    ]
    
    for query in questions:
        print(f"\n📝 Question: {query}")
        print("-"*60)
        
        answer_result = await rag_query(
            query=query,
            tree=result['structure'],
            pdf_pages=pdf_pages,
            model=MODEL
        )
        
        print(f"🎯 Answer:\n{answer_result['answer']}")
        print(f"\n📍 Sources: {', '.join(answer_result['relevant_nodes'])}")
        print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
