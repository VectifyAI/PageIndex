"""
Specialized RedBook Query Interface
Handles large technical documents with tabular data
"""

import asyncio
import json
import re
from pathlib import Path
from gemini_pageindex import gemini_api_async, get_page_tokens

async def search_redbook_pages(query, pdf_pages, max_pages_per_query=20):
    """
    Search through RedBook pages in chunks to find specific data
    """
    # Extract search terms from query
    search_terms = extract_search_terms(query)
    
    print(f"Search terms: {search_terms}")
    
    # Find pages with relevant terms
    relevant_pages = []
    for i, (page_text, tokens) in enumerate(pdf_pages[:100]):  # Search first 100 pages (where most tables are)
        if any(term.lower() in page_text.lower() for term in search_terms):
            relevant_pages.append((i+1, page_text))
    
    print(f"Found {len(relevant_pages)} potentially relevant pages")
    
    if not relevant_pages:
        return "Could not find relevant pages in the document."
    
    # Take first few relevant pages
    pages_to_analyze = relevant_pages[:max_pages_per_query]
    
    # Build context
    context = "\n\n".join([
        f"PAGE {page_num}:\n{text[:5000]}"  # Limit each page to 5000 chars
        for page_num, text in pages_to_analyze
    ])
    
    # Query Gemini with the specific context
    prompt = f"""
Answer this question based on the provided pages from a technical manual:

Question: {query}

Pages:
{context}

Provide a specific, numerical answer if available. Include the page number where you found the information.
"""
    
    answer = await gemini_api_async(prompt, model="gemini-2.0-flash")
    return answer


def extract_search_terms(query):
    """Extract key search terms from query"""
    # Extract numbers and units
    terms = []
    
    # Find measurements like "10.75", "13"", etc.
    measurements = re.findall(r'\d+\.?\d*\s*(?:"|in|inch|ft|psi)', query, re.IGNORECASE)
    terms.extend(measurements)
    
    # Extract key technical terms
    technical_terms = [
        'casing', 'tubing', 'hole', 'diameter', 'pressure', 'collapse',
        'gallons', 'barrel', 'cubic', 'cu ft', 'gal/ft', 'GPF',
        'PDC', 'bit', 'WOB', 'IADC', 'coiled', 'cement', 'bentonite',
        'strength', 'compressive', 'thickening', 'bulk weight', 'fly ash',
        'safety factor', 'SF', 'utilization', 'lube', 'Baro Lube',
        'Diacel', 'Micro', 'polymer'
    ]
    
    for term in technical_terms:
        if term.lower() in query.lower():
            terms.append(term)
    
    return list(set(terms)) if terms else [query]


async def main():
    """Interactive query interface for RedBook"""
    
    # Load PDF
    pdf_path = r"C:\Users\akash\Desktop\rovabot-dev\Rovabot\RedBook.pdf"
    
    print("Loading RedBook PDF...")
    pdf_pages = get_page_tokens(pdf_path)
    print(f"Loaded {len(pdf_pages)} pages")
    
    print("\n" + "="*60)
    print("RedBook Query Interface")
    print("="*60)
    print("Type your questions (or 'quit' to exit)\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("\n🔍 Searching...")
        
        try:
            answer = await search_redbook_pages(query, pdf_pages)
            print(f"\n🤖 Answer:\n{answer}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
