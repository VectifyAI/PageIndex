"""
Gemini-compatible PageIndex implementation
Replaces OpenAI API calls with Gemini API
"""

import os
import json
import time
import asyncio
import re
import copy
import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path

import google.generativeai as genai
import tiktoken
import PyPDF2
import pymupdf

from dotenv import load_dotenv
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Gemini API Wrapper Functions
# ============================================================================

def gemini_api(prompt, model="gemini-2.0-flash", temperature=0, max_retries=5):
    """
    Call Gemini API with retry logic
    """
    for attempt in range(max_retries):
        try:
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 8192,
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Max retries reached. Returning error message.")
                return "Error: API call failed"


async def gemini_api_async(prompt, model="gemini-2.0-flash", temperature=0, max_retries=5):
    """
    Async version of Gemini API call
    """
    # Gemini SDK doesn't have native async, so we wrap it
    return await asyncio.to_thread(gemini_api, prompt, model, temperature, max_retries)


# ============================================================================
# Utility Functions (adapted from pageindex/utils.py)
# ============================================================================

def count_tokens(text, model="gpt-4o"):
    """Count tokens using tiktoken"""
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except:
        # Fallback: estimate 4 chars per token
        return len(text) // 4


def extract_json(content):
    """Extract JSON from Gemini response"""
    try:
        # Try to find JSON in code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.rfind("```")
            json_str = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.rfind("```")
            json_str = content[start:end].strip()
        else:
            json_str = content.strip()
        
        # Clean up
        json_str = json_str.replace('None', 'null')
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Remove trailing commas
        
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"JSON extraction failed: {e}")
        logger.debug(f"Content: {content[:500]}")
        return {}


def get_page_tokens(pdf_path, model="gpt-4o"):
    """Extract text and count tokens from PDF"""
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    page_list = []
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        token_length = count_tokens(page_text, model)
        page_list.append((page_text, token_length))
    
    return page_list


def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    """Get text from page range"""
    text = ""
    for page_num in range(start_page-1, end_page):
        if page_num < len(pdf_pages):
            text += pdf_pages[page_num][0] + "\n"
    return text


# ============================================================================
# TOC Detection & Extraction
# ============================================================================

def detect_toc_page(page_text, model="gemini-1.5-flash"):
    """Detect if page contains table of contents (improved prompt to avoid false positives)"""
    prompt = f"""
Determine if this page is a Table of Contents (TOC) page or a content page.

A TRUE Table of Contents:
- Lists multiple sections/chapters with page numbers
- References OTHER pages in the document (not describing current page)
- Usually has "Table of Contents", "Contents", or similar header
- Example: "Chapter 1: Introduction ... 5"

NOT a Table of Contents:
- A regular content page with a title/heading
- Product descriptions or specifications
- Pages that describe their own content (not referencing other pages)
- Example: "SUPER SEAL™ II FLOAT EQUIPMENT" (this is a content page title)

Page text:
{page_text[:2000]}

Is this a TRUE Table of Contents page that references other pages?
Answer ONLY "yes" or "no".
"""
    response = gemini_api(prompt, model)
    return "yes" if "yes" in response.lower() else "no"


def extract_toc_from_pages(toc_pages_text, model="gemini-1.5-flash"):
    """Extract TOC structure from text"""
    prompt = f"""
Extract the table of contents from the following text and convert it to JSON format.

Text:
{toc_pages_text}

Return a JSON array with this structure:
[
  {{
    "structure": "1",
    "title": "Chapter Title",
    "page": 5
  }},
  {{
    "structure": "1.1",
    "title": "Section Title",
    "page": 8
  }}
]

Only return valid JSON, no other text.
"""
    response = gemini_api(prompt, model)
    return extract_json(response)


def generate_toc_from_content(pages_text, model="gemini-1.5-flash"):
    """Generate TOC from document content (when no TOC exists)"""
    prompt = f"""
Analyze this document and generate a table of contents structure.

Document:
{pages_text[:15000]}

Return JSON array with sections found:
[
  {{
    "structure": "1",
    "title": "Section Title",
    "physical_index": 1
  }}
]

Only return valid JSON.
"""
    response = gemini_api(prompt, model)
    result = extract_json(response)
    return result if isinstance(result, list) else []


# ============================================================================
# Tree Building Functions
# ============================================================================

def list_to_tree(data):
    """Convert flat TOC list to hierarchical tree"""
    def get_parent_structure(structure):
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None
    
    nodes = {}
    root_nodes = []
    
    for item in data:
        structure = item.get('structure', '')
        node = {
            'title': item.get('title', 'Untitled'),
            'start_index': item.get('physical_index', item.get('start_index', 1)),
            'end_index': item.get('end_index', 1),
            'nodes': []
        }
        
        nodes[structure] = node
        parent_structure = get_parent_structure(structure)
        
        if parent_structure and parent_structure in nodes:
            nodes[parent_structure]['nodes'].append(node)
        else:
            root_nodes.append(node)
    
    # Clean empty nodes
    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node
    
    return [clean_node(node) for node in root_nodes]


def post_processing(structure, end_physical_index):
    """Process TOC list into tree with page ranges"""
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index', i + 1)
        if i < len(structure) - 1:
            item['end_index'] = structure[i + 1].get('physical_index', i + 2) - 1
        else:
            item['end_index'] = end_physical_index
    
    tree = list_to_tree(structure)
    return tree if tree else structure


def add_node_ids(tree, counter=0):
    """Add unique node IDs to tree"""
    if isinstance(tree, list):
        for node in tree:
            counter = add_node_ids(node, counter)
    elif isinstance(tree, dict):
        tree['node_id'] = f"{counter:04d}"
        counter += 1
        if 'nodes' in tree:
            counter = add_node_ids(tree['nodes'], counter)
    return counter


def add_node_text(tree, pdf_pages):
    """Add text content to each node"""
    if isinstance(tree, list):
        for node in tree:
            add_node_text(node, pdf_pages)
    elif isinstance(tree, dict):
        start = tree.get('start_index', 1)
        end = tree.get('end_index', 1)
        tree['text'] = get_text_of_pdf_pages(pdf_pages, start, end)
        if 'nodes' in tree:
            add_node_text(tree['nodes'], pdf_pages)


async def generate_node_summary(node, model="gemini-1.5-flash"):
    """Generate summary for a node"""
    prompt = f"""
Summarize the main points of this document section in 2-3 sentences:

{node['text'][:5000]}

Be concise and factual.
"""
    return await gemini_api_async(prompt, model)


async def generate_summaries(tree, model="gemini-1.5-flash"):
    """Generate summaries for all nodes"""
    if isinstance(tree, list):
        for node in tree:
            await generate_summaries(node, model)
    elif isinstance(tree, dict):
        if 'text' in tree and tree['text']:
            tree['summary'] = await generate_node_summary(tree, model)
        if 'nodes' in tree:
            await generate_summaries(tree['nodes'], model)


# ============================================================================
# Main PageIndex Generation Function
# ============================================================================

async def process_large_node_recursively(node, pdf_pages, model="gemini-1.5-flash", 
                                         max_pages=10, max_tokens=20000):
    """
    Recursively split large nodes into smaller sub-nodes
    Based on PageIndex process_large_node_recursively()
    """
    start = node.get('start_index', 1)
    end = node.get('end_index', 1)
    
    # Calculate node size
    node_pages = pdf_pages[start-1:end]
    token_count = sum(p[1] for p in node_pages)
    page_count = end - start + 1
    
    # Check if node needs splitting
    if page_count > max_pages and token_count > max_tokens:
        logger.info(f"Splitting large node: {node.get('title')} (pages {start}-{end}, {token_count:,} tokens)")
        
        # Generate sub-TOC for this section
        section_text = "\n".join([p[0][:3000] for p in node_pages[:20]])  # First 20 pages, truncated
        
        prompt = f"""
This is a large section spanning pages {start} to {end}. 
Analyze the content and identify subsections or logical divisions.

Content:
{section_text}

Return JSON array with subsections:
[
  {{
    "structure": "1",
    "title": "Subsection Title",
    "physical_index": {start}
  }}
]

If no clear subsections exist, split into roughly equal parts by page ranges.
Only return valid JSON.
"""
        
        try:
            response = await gemini_api_async(prompt, model)
            sub_toc = extract_json(response)
            
            if sub_toc and isinstance(sub_toc, list) and len(sub_toc) > 1:
                # Process sub-TOC
                for i, item in enumerate(sub_toc):
                    if 'physical_index' not in item:
                        item['physical_index'] = start + (i * page_count // len(sub_toc))
                
                # Build sub-tree
                sub_tree = post_processing(sub_toc, end)
                
                # Replace current node with sub-tree
                if sub_tree:
                    node['nodes'] = sub_tree
                    node['end_index'] = sub_tree[0].get('start_index', start)
                    
                    # Recursively process child nodes
                    if 'nodes' in node:
                        for child in node['nodes']:
                            await process_large_node_recursively(child, pdf_pages, model, max_pages, max_tokens)
            else:
                logger.warning(f"Could not split node {node.get('title')}, keeping as-is")
        except Exception as e:
            logger.error(f"Error splitting node: {e}")
    
    # Process existing child nodes
    elif 'nodes' in node:
        for child in node['nodes']:
            await process_large_node_recursively(child, pdf_pages, model, max_pages, max_tokens)
    
    return node


async def generate_pageindex(
    pdf_path,
    model="gemini-1.5-flash",
    add_summaries=False,
    add_text=False,
    max_toc_pages=20,
    max_pages_per_node=10,
    max_tokens_per_node=20000
):
    """
    Generate PageIndex tree structure from PDF
    
    Args:
        pdf_path: Path to PDF file
        model: Gemini model to use
        add_summaries: Whether to generate summaries (slower, costs more)
        add_text: Whether to include full text in nodes
        max_toc_pages: Max pages to check for TOC
        max_pages_per_node: Max pages per node before splitting
        max_tokens_per_node: Max tokens per node before splitting
    
    Returns:
        dict with 'doc_name' and 'structure'
    """
    logger.info(f"Processing: {pdf_path}")
    
    # Extract PDF pages
    logger.info("Extracting PDF text...")
    page_list = get_page_tokens(pdf_path)
    total_pages = len(page_list)
    total_tokens = sum(p[1] for p in page_list)
    
    logger.info(f"Pages: {total_pages}, Tokens: {total_tokens:,}")
    
    # Try to find TOC
    logger.info("Detecting table of contents...")
    toc_pages = []
    for i in range(min(max_toc_pages, total_pages)):
        result = detect_toc_page(page_list[i][0], model)
        if result == "yes":
            toc_pages.append(i)
        elif toc_pages:  # Stop after TOC ends
            break
    
    # Build structure
    if toc_pages:
        logger.info(f"Found TOC on pages: {[p+1 for p in toc_pages]}")
        toc_text = "\n".join(page_list[i][0] for i in toc_pages)
        toc_structure = extract_toc_from_pages(toc_text, model)
    else:
        logger.info("No TOC found, generating from content...")
        first_pages = "\n".join(p[0] for p in page_list[:10])
        toc_structure = generate_toc_from_content(first_pages, model)
    
    logger.info(f"Found {len(toc_structure)} sections")
    
    # Build tree
    logger.info("Building tree structure...")
    tree = post_processing(toc_structure, total_pages)
    
    # Add node IDs
    add_node_ids(tree)
    
    # Process large nodes recursively
    logger.info("Processing large nodes...")
    for node in tree:
        await process_large_node_recursively(node, page_list, model, max_pages_per_node, max_tokens_per_node)
    
    # Re-assign node IDs after splitting
    add_node_ids(tree)
    
    # Add text if requested
    if add_text or add_summaries:
        logger.info("Adding text to nodes...")
        add_node_text(tree, page_list)
    
    # Generate summaries if requested
    if add_summaries:
        logger.info("Generating summaries (this may take a while)...")
        await generate_summaries(tree, model)
        
        # Remove text if not requested
        if not add_text:
            def remove_text(node):
                if isinstance(node, list):
                    for n in node:
                        remove_text(n)
                elif isinstance(node, dict):
                    node.pop('text', None)
                    if 'nodes' in node:
                        remove_text(node['nodes'])
            remove_text(tree)
    
    doc_name = Path(pdf_path).stem
    
    result = {
        'doc_name': doc_name,
        'total_pages': total_pages,
        'total_tokens': total_tokens,
        'structure': tree
    }
    
    logger.info("✓ PageIndex generation complete!")
    return result


# ============================================================================
# RAG Query Functions
# ============================================================================

def structure_to_list(tree):
    """Flatten tree to list of nodes"""
    nodes = []
    if isinstance(tree, list):
        for item in tree:
            nodes.extend(structure_to_list(item))
    elif isinstance(tree, dict):
        nodes.append(tree)
        if 'nodes' in tree:
            nodes.extend(structure_to_list(tree['nodes']))
    return nodes


def remove_fields(tree, fields=['text']):
    """Remove specified fields from tree"""
    if isinstance(tree, dict):
        return {k: remove_fields(v, fields) for k, v in tree.items() if k not in fields}
    elif isinstance(tree, list):
        return [remove_fields(item, fields) for item in tree]
    return tree


async def reasoning_based_retrieval(query, tree, model="gemini-2.0-flash"):
    """
    Use Gemini to perform reasoning-based tree search
    """
    tree_no_text = remove_fields(tree, fields=['text'])
    
    prompt = f"""
Given this document structure (tree of contents):

{json.dumps(tree_no_text, indent=2)}

Which node_ids would most likely contain information to answer this question:
"{query}"

Think step by step:
1. What type of information does the query need?
2. Which sections would contain that information?
3. List the relevant node_ids

Return JSON format:
{{
  "thinking": "explanation of reasoning",
  "node_ids": ["0001", "0015", "0023"]
}}

Only return valid JSON.
"""
    
    response = await gemini_api_async(prompt, model)
    result = extract_json(response)
    
    return result.get('node_ids', [])


async def generate_answer(query, context, model="gemini-2.0-flash"):
    """Generate answer based on retrieved context"""
    # Truncate context if it's too large (keep under 500k tokens / ~2M characters)
    max_context_chars = 2000000
    if len(context) > max_context_chars:
        # Keep first and last parts for context
        keep_chars = max_context_chars - 500
        first_chars = keep_chars // 2
        last_chars = keep_chars - first_chars
        context = context[:first_chars] + "\n\n[... content truncated ...]\n\n" + context[-last_chars:]
        logger.info(f"Context truncated to {len(context):,} characters")
    
    prompt = f"""
Answer the following question based on the provided context.

Question: {query}

Context:
{context}

Provide a detailed, accurate answer. Include page references when possible.
"""
    
    return await gemini_api_async(prompt, model)


async def rag_query(query, tree, pdf_pages, model="gemini-2.0-flash"):
    """
    Complete RAG pipeline: retrieve + generate answer
    """
    logger.info(f"Query: {query}")
    
    # Step 1: Reasoning-based retrieval
    logger.info("Searching tree structure...")
    node_ids = await reasoning_based_retrieval(query, tree, model)
    logger.info(f"Found relevant nodes: {node_ids}")
    
    # Step 2: Extract context
    nodes = structure_to_list(tree)
    node_map = {n['node_id']: n for n in nodes}
    
    context_parts = []
    for nid in node_ids:
        if nid in node_map:
            node = node_map[nid]
            title = node.get('title', 'Untitled')
            start = node.get('start_index', 1)
            end = node.get('end_index', 1)
            
            # Get text
            if 'text' not in node:
                node['text'] = get_text_of_pdf_pages(pdf_pages, start, end)
            
            # Limit text per section to avoid token overflow
            section_text = node['text']
            max_section_chars = 100000  # ~25k tokens per section
            if len(section_text) > max_section_chars:
                section_text = section_text[:max_section_chars] + "\n[... truncated ...]"
            
            context_parts.append(
                f"Section: {title} (Pages {start}-{end})\n{section_text}"
            )
    
    context = "\n\n".join(context_parts)
    
    # Step 3: Generate answer
    logger.info("Generating answer...")
    answer = await generate_answer(query, context, model)
    
    return {
        'query': query,
        'relevant_nodes': node_ids,
        'answer': answer
    }


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PageIndex with Gemini API")
    parser.add_argument('--pdf', required=True, help='Path to PDF file')
    parser.add_argument('--model', default='gemini-1.5-flash', 
                       help='Gemini model (gemini-1.5-flash or gemini-1.5-pro)')
    parser.add_argument('--summaries', action='store_true', 
                       help='Generate summaries (slower)')
    parser.add_argument('--text', action='store_true', 
                       help='Include full text in output')
    parser.add_argument('--output', help='Output JSON file path')
    
    args = parser.parse_args()
    
    async def main():
        result = await generate_pageindex(
            pdf_path=args.pdf,
            model=args.model,
            add_summaries=args.summaries,
            add_text=args.text
        )
        
        # Save result
        output_path = args.output or f"./results/{result['doc_name']}_structure.json"
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved to: {output_path}")
    
    asyncio.run(main())
