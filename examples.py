"""
Example Usage Patterns for PageIndex with Gemini API

This file demonstrates common usage patterns and workflows.
Copy and adapt these examples for your own use cases.
"""

import asyncio
import json
import os
from pathlib import Path
from gemini_pageindex import (
    generate_pageindex,
    rag_query,
    get_page_tokens,
    structure_to_list
)


# ============================================================================
# Example 1: Basic Tree Generation
# ============================================================================

async def example_basic_generation():
    """Generate a PageIndex tree from a PDF."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Tree Generation")
    print("="*60)
    
    result = await generate_pageindex(
        pdf_path="./my_document.pdf",
        model="gemini-2.0-flash",
        add_summaries=False,
        add_text=False
    )
    
    print(f"Document: {result['doc_name']}")
    print(f"Pages: {result['total_pages']}")
    print(f"Tokens: {result['total_tokens']:,}")
    print(f"Sections: {len(result['structure'])}")
    
    return result


# ============================================================================
# Example 2: Generation with Summaries
# ============================================================================

async def example_with_summaries():
    """Generate tree with summaries for each section."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Generation with Summaries")
    print("="*60)
    
    result = await generate_pageindex(
        pdf_path="./my_document.pdf",
        model="gemini-2.0-flash",
        add_summaries=True,  # Enable summaries
        add_text=False       # Don't include full text
    )
    
    # Print summaries
    def print_summaries(nodes, level=0):
        indent = "  " * level
        for node in nodes:
            title = node.get('title', 'Untitled')
            summary = node.get('summary', 'No summary')
            print(f"{indent}• {title}")
            if summary:
                print(f"{indent}  → {summary[:100]}...")
            if 'nodes' in node:
                print_summaries(node['nodes'], level + 1)
    
    print_summaries(result['structure'])
    
    return result


# ============================================================================
# Example 3: Single Query
# ============================================================================

async def example_single_query():
    """Ask a single question about the document."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Single Query")
    print("="*60)
    
    # Generate index
    result = await generate_pageindex(
        pdf_path="./my_document.pdf",
        add_text=True
    )
    
    pdf_pages = get_page_tokens("./my_document.pdf")
    
    # Ask question
    answer = await rag_query(
        query="What are the main findings?",
        tree=result['structure'],
        pdf_pages=pdf_pages
    )
    
    print(f"Question: {answer['query']}")
    print(f"\nAnswer:\n{answer['answer']}")
    print(f"\nSources: {', '.join(answer['relevant_nodes'])}")
    
    return answer


# ============================================================================
# Example 4: Multiple Queries (Cached)
# ============================================================================

async def example_multiple_queries():
    """Ask multiple questions using cached tree."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Multiple Queries (Cached)")
    print("="*60)
    
    # Generate index once
    print("Generating index (one-time)...")
    result = await generate_pageindex(
        pdf_path="./my_document.pdf",
        add_text=True
    )
    
    # Cache it
    with open('tree_cache.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("✓ Cached tree")
    
    # Get pages for queries
    pdf_pages = get_page_tokens("./my_document.pdf")
    
    # Ask multiple questions
    questions = [
        "What is the executive summary?",
        "What are the key metrics?",
        "What are the recommendations?",
        "What is the budget?",
        "What are the next steps?"
    ]
    
    answers = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}")
        
        answer = await rag_query(
            query=question,
            tree=result['structure'],
            pdf_pages=pdf_pages
        )
        answers.append(answer)
        
        print(f"✓ Answer retrieved from nodes: {', '.join(answer['relevant_nodes'])}")
    
    # Save all answers
    with open('answers.json', 'w') as f:
        json.dump(answers, f, indent=2)
    print(f"\n✓ Saved {len(answers)} answers to answers.json")
    
    return answers


# ============================================================================
# Example 5: Batch Processing
# ============================================================================

async def example_batch_processing():
    """Process multiple PDFs."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Processing Multiple PDFs")
    print("="*60)
    
    pdf_dir = Path("./pdfs/")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDFs")
    
    # Create output directory
    output_dir = Path("./results/")
    output_dir.mkdir(exist_ok=True)
    
    # Process each PDF
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        result = await generate_pageindex(
            pdf_path=str(pdf_file),
            add_summaries=False,
            add_text=False
        )
        
        # Save result
        output_file = output_dir / f"{result['doc_name']}_tree.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Saved to {output_file}")
    
    print(f"\n✓ Processed {len(pdf_files)} documents")


# ============================================================================
# Example 6: Custom Model Selection
# ============================================================================

async def example_different_models():
    """Compare different models."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Different Models Comparison")
    print("="*60)
    
    pdf = "./my_document.pdf"
    
    models = [
        ("gemini-2.0-flash", "Fast & Cheap"),
        ("gemini-2.5-pro", "Better Quality"),
    ]
    
    results = {}
    
    for model, description in models:
        print(f"\nTesting {model} ({description})...")
        
        result = await generate_pageindex(
            pdf_path=pdf,
            model=model,
            add_summaries=False
        )
        
        results[model] = {
            'model': model,
            'description': description,
            'sections': len(result['structure']),
            'tokens': result['total_tokens']
        }
        
        print(f"  • Sections: {result['structure'].__len__()}")
        print(f"  • Tokens: {result['total_tokens']:,}")
    
    # Display comparison
    print("\n" + "="*40)
    for model, info in results.items():
        print(f"{info['model']}: {info['sections']} sections, {info['tokens']:,} tokens")
    
    return results


# ============================================================================
# Example 7: Search Within Document
# ============================================================================

async def example_keyword_search():
    """Search document for specific keywords."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Keyword Search")
    print("="*60)
    
    # Generate tree
    result = await generate_pageindex(
        pdf_path="./my_document.pdf",
        add_text=True
    )
    
    pdf_pages = get_page_tokens("./my_document.pdf")
    
    # Search for keywords
    keywords = ["revenue", "profit", "loss", "forecast"]
    
    for keyword in keywords:
        print(f"\nSearching for: {keyword}")
        
        answer = await rag_query(
            query=f"Find all information about {keyword}",
            tree=result['structure'],
            pdf_pages=pdf_pages
        )
        
        print(f"Found in nodes: {', '.join(answer['relevant_nodes'])}")
    
    return answer


# ============================================================================
# Example 8: Hierarchical Navigation
# ============================================================================

async def example_hierarchical_navigation():
    """Navigate document structure hierarchically."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Hierarchical Navigation")
    print("="*60)
    
    # Generate tree
    result = await generate_pageindex(
        pdf_path="./my_document.pdf",
        add_summaries=True,
        add_text=False
    )
    
    # Function to print hierarchy
    def print_hierarchy(nodes, level=0):
        indent = "  " * level
        for node in nodes:
            title = node.get('title', 'Untitled')
            pages = f"pp. {node.get('start_index', '?')}-{node.get('end_index', '?')}"
            print(f"{indent}├─ {title} [{pages}]")
            
            if 'nodes' in node:
                print_hierarchy(node['nodes'], level + 1)
    
    print("\nDocument Structure:")
    print_hierarchy(result['structure'])
    
    return result


# ============================================================================
# Example 9: Export to Different Formats
# ============================================================================

async def example_export_formats():
    """Export tree structure in different formats."""
    print("\n" + "="*60)
    print("EXAMPLE 9: Export Formats")
    print("="*60)
    
    # Generate tree
    result = await generate_pageindex(
        pdf_path="./my_document.pdf",
        add_text=False
    )
    
    # 1. JSON (default)
    with open('export.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("✓ Exported to export.json")
    
    # 2. CSV
    import csv
    
    def flatten_tree(nodes, parent_id=""):
        rows = []
        for node in nodes:
            row = {
                'node_id': node.get('node_id'),
                'title': node.get('title'),
                'start_page': node.get('start_index'),
                'end_page': node.get('end_index'),
                'parent_id': parent_id
            }
            rows.append(row)
            if 'nodes' in node:
                rows.extend(flatten_tree(node['nodes'], node.get('node_id')))
        return rows
    
    rows = flatten_tree(result['structure'])
    with open('export.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['node_id', 'title', 'start_page', 'end_page', 'parent_id'])
        writer.writeheader()
        writer.writerows(rows)
    print("✓ Exported to export.csv")
    
    # 3. Markdown outline
    def to_markdown(nodes, level=0):
        lines = []
        for node in nodes:
            title = node.get('title', 'Untitled')
            pages = f"{node.get('start_index', '?')}-{node.get('end_index', '?')}"
            lines.append(f"{'#' * (level + 1)} {title} (pp. {pages})")
            
            if 'nodes' in node:
                lines.extend(to_markdown(node['nodes'], level + 1))
        return lines
    
    with open('export.md', 'w') as f:
        f.write("# Document Outline\n\n")
        f.write("\n".join(to_markdown(result['structure'])))
    print("✓ Exported to export.md")
    
    return result


# ============================================================================
# Example 10: Advanced Analysis
# ============================================================================

async def example_advanced_analysis():
    """Advanced document analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 10: Advanced Analysis")
    print("="*60)
    
    # Generate tree
    result = await generate_pageindex(
        pdf_path="./my_document.pdf",
        add_text=True
    )
    
    pdf_pages = get_page_tokens("./my_document.pdf")
    
    # Get all nodes
    all_nodes = structure_to_list(result['structure'])
    print(f"\nTotal nodes: {len(all_nodes)}")
    
    # Calculate statistics
    total_pages = result['total_pages']
    total_tokens = result['total_tokens']
    avg_tokens_per_page = total_tokens / total_pages if total_pages > 0 else 0
    
    print(f"Pages: {total_pages}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Avg tokens/page: {avg_tokens_per_page:.0f}")
    
    # Find largest sections
    print("\nLargest sections:")
    nodes_by_size = sorted(
        all_nodes,
        key=lambda n: (n.get('end_index', 0) - n.get('start_index', 0)),
        reverse=True
    )[:5]
    
    for node in nodes_by_size:
        size = node.get('end_index', 0) - node.get('start_index', 0)
        print(f"  • {node.get('title', 'Untitled')}: {size} pages")
    
    # Query for overall document summary
    summary = await rag_query(
        query="Provide a comprehensive summary of the entire document",
        tree=result['structure'],
        pdf_pages=pdf_pages
    )
    
    print(f"\nDocument Summary:\n{summary['answer']}")
    
    return result


# ============================================================================
# Main Runner
# ============================================================================

async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("PageIndex Gemini API - Usage Examples")
    print("="*60)
    
    examples = [
        (example_basic_generation, "Basic Tree Generation"),
        (example_with_summaries, "Generation with Summaries"),
        (example_single_query, "Single Query"),
        (example_multiple_queries, "Multiple Queries"),
        # Uncomment to run:
        # (example_batch_processing, "Batch Processing"),
        # (example_different_models, "Different Models"),
        # (example_keyword_search, "Keyword Search"),
        # (example_hierarchical_navigation, "Hierarchical Navigation"),
        # (example_export_formats, "Export Formats"),
        # (example_advanced_analysis, "Advanced Analysis"),
    ]
    
    print("\nAvailable examples:")
    for i, (func, name) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nNote: Modify this file to run individual examples.")
    print("Uncomment examples in the main() function to test them.\n")


if __name__ == "__main__":
    asyncio.run(main())
