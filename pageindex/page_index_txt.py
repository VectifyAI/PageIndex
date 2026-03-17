import asyncio
import json
import re
import os
try:
    from .utils import *
except:
    from utils import *


async def detect_semantic_sections(text, model=None, max_input_tokens=None):
    """
    Use LLM to detect semantic sections (headings and subheadings) in unstructured text.
    Returns a list of sections with their titles, levels, and character positions.
    """
    # For large texts, we need to process in chunks
    # Use character-based chunking to maintain accurate position tracking
    text_length = len(text)
    
    # If text is small enough, process it all at once
    if not max_input_tokens or count_tokens(text, model=model) <= max_input_tokens:
        chunks = [text]
        chunk_positions = [0]
    else:
        # Chunk by characters, not tokens, to maintain accurate position tracking
        # Estimate chars per token (~4) and chunk accordingly
        estimated_chars_per_chunk = max_input_tokens * 3  # Conservative estimate
        overlap_chars = 500  # Character overlap between chunks
        
        chunks = []
        chunk_positions = []
        start = 0
        
        while start < text_length:
            end = min(start + estimated_chars_per_chunk, text_length)
            
            # Try to break at sentence boundary if not at the end
            if end < text_length:
                search_start = max(end - 200, start)
                sentence_endings = [m.end() for m in re.finditer(r'[.!?]\s+', text[search_start:end])]
                if sentence_endings:
                    end = search_start + sentence_endings[-1]
            
            chunks.append(text[start:end])
            chunk_positions.append(start)
            
            if end >= text_length:
                break
            start = end - overlap_chars
    
    all_sections = []
    
    for chunk_idx, (chunk, chunk_start_pos) in enumerate(zip(chunks, chunk_positions)):
        prompt = f"""You are an expert document analyzer. Your task is to identify semantic sections (headings and subheadings) in the following unstructured text.

Analyze the text and identify natural section boundaries, titles, and their hierarchical levels (1 for main sections, 2 for subsections, 3 for sub-subsections, etc.).

Look for:
- Topic changes or transitions
- Natural paragraph groupings
- Semantic breaks in content
- Logical organization of ideas

Text to analyze:
{chunk}

Return a JSON array of sections with the following structure:
[
  {{
    "title": "Section Title",
    "level": 1,
    "char_start": 0,
    "char_end": 500
  }},
  ...
]

Important:
- char_start and char_end should be the character positions within THIS chunk
- Assign appropriate hierarchical levels based on content importance and structure
- Create descriptive titles that summarize each section's content
- Ensure sections don't overlap
- Include at least one section, even if the text appears uniform

Directly return ONLY the JSON array, no other text."""

        response = await ChatGPT_API_async(model, prompt)
        sections = extract_json(response)
        
        if not isinstance(sections, list):
            sections = []
        
        # Adjust character positions to be relative to the original text
        for section in sections:
            if 'char_start' in section and 'char_end' in section:
                section['char_start'] += chunk_start_pos
                section['char_end'] += chunk_start_pos
                section['chunk_idx'] = chunk_idx
                all_sections.append(section)
    
    # If no sections were detected, create a single section for the entire text
    if not all_sections:
        all_sections = [{
            'title': 'Document Content',
            'level': 1,
            'char_start': 0,
            'char_end': len(text)
        }]
    
    return all_sections


def window_text_with_overlap(text, window_size=5000, overlap=500):
    """
    Split text into overlapping windows by character count.
    
    Args:
        text: The text to split
        window_size: Size of each window in characters
        overlap: Number of characters to overlap between windows
        
    Returns:
        List of text windows with their start and end positions
    """
    if len(text) <= window_size:
        return [{'text': text, 'start': 0, 'end': len(text)}]
    
    windows = []
    start = 0
    
    while start < len(text):
        end = min(start + window_size, len(text))
        
        # Try to break at sentence boundary if not at the end
        if end < len(text):
            # Look for sentence ending within last 200 chars of window
            search_start = max(end - 200, start)
            sentence_endings = [m.end() for m in re.finditer(r'[.!?]\s+', text[search_start:end])]
            if sentence_endings:
                end = search_start + sentence_endings[-1]
        
        window_text = text[start:end]
        windows.append({
            'text': window_text,
            'start': start,
            'end': end
        })
        
        # Move to next window with overlap
        if end >= len(text):
            break
        start = end - overlap
    
    return windows


def merge_overlapping_sections(sections):
    """
    Remove duplicate sections that have the same or very similar positions.
    Note: This function is designed to handle sections detected from overlapping windows,
    not to flatten hierarchical structures. Sections at different levels are kept separate.
    """
    if not sections:
        return []
    
    # Sort by start position, then by level
    sorted_sections = sorted(sections, key=lambda x: (x['char_start'], x.get('level', 1)))
    
    merged = []
    seen_positions = set()
    
    for section in sorted_sections:
        # Create a position key that's unique enough to detect duplicates
        # but allows for hierarchy (same start, different levels)
        position_key = (section['char_start'], section.get('level', 1), section.get('title', ''))
        
        # Skip if we've seen this exact section before
        if position_key in seen_positions:
            continue
        
        seen_positions.add(position_key)
        merged.append(section)
    
    return merged


def extract_section_text(text, sections):
    """
    Extract text content for each section based on character positions.
    """
    nodes = []
    
    for i, section in enumerate(sections):
        start = section.get('char_start', 0)
        # End is either the start of next section or specified char_end
        if i + 1 < len(sections):
            end = sections[i + 1].get('char_start', section.get('char_end', len(text)))
        else:
            end = section.get('char_end', len(text))
        
        section_text = text[start:end].strip()
        
        node = {
            'title': section.get('title', f'Section {i+1}'),
            'level': section.get('level', 1),
            'char_start': start,
            'char_end': end,
            'text': section_text
        }
        nodes.append(node)
    
    return nodes


def build_tree_from_txt_nodes(node_list):
    """
    Build a hierarchical tree structure from a flat list of nodes with levels.
    Similar to the markdown tree building but for text nodes.
    """
    if not node_list:
        return []
    
    stack = []
    root_nodes = []
    node_counter = 1
    
    for node in node_list:
        current_level = node['level']
        
        tree_node = {
            'title': node['title'],
            'node_id': str(node_counter).zfill(4),
            'text': node['text'],
            'char_start': node['char_start'],
            'char_end': node['char_end'],
            'nodes': []
        }
        node_counter += 1
        
        # Pop from stack until we find a parent (lower level number = higher hierarchy)
        while stack and stack[-1][1] >= current_level:
            stack.pop()
        
        if not stack:
            root_nodes.append(tree_node)
        else:
            parent_node, parent_level = stack[-1]
            parent_node['nodes'].append(tree_node)
        
        stack.append((tree_node, current_level))
    
    return root_nodes


async def get_node_summary_txt(node, summary_token_threshold=200, model=None, max_input_tokens=None):
    """
    Generate summary for a text node, similar to markdown node summary generation.
    """
    node_text = node.get('text', '')
    num_tokens = count_tokens(node_text, model=model)
    
    if num_tokens < summary_token_threshold:
        return node_text
    else:
        return await generate_node_summary(node, model=model, max_input_tokens=max_input_tokens)


async def generate_summaries_for_txt_structure(structure, summary_token_threshold, model=None, max_input_tokens=None):
    """
    Generate summaries for all nodes in the tree structure.
    """
    nodes = structure_to_list(structure)
    tasks = [get_node_summary_txt(node, summary_token_threshold=summary_token_threshold, 
                                   model=model, max_input_tokens=max_input_tokens) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    
    for node, summary in zip(nodes, summaries):
        if not node.get('nodes'):
            node['summary'] = summary
        else:
            node['prefix_summary'] = summary
    
    return structure


async def txt_to_tree(txt_path, window_size=5000, overlap=500, if_add_node_summary='no', 
                      summary_token_threshold=200, model=None, if_add_doc_description='no', 
                      if_add_node_text='no', if_add_node_id='yes', max_input_tokens=None):
    """
    Convert a .txt file to a tree structure by:
    1. Reading the text
    2. Using windowing approach with overlap
    3. Detecting semantic sections using LLM
    4. Building tree structure
    5. Generating summaries if requested
    
    Args:
        txt_path: Path to the .txt file
        window_size: Size of each window in characters for processing
        overlap: Number of characters to overlap between windows
        if_add_node_summary: Whether to add summaries to nodes ('yes' or 'no')
        summary_token_threshold: Token threshold for generating summaries
        model: LLM model to use
        if_add_doc_description: Whether to add document description ('yes' or 'no')
        if_add_node_text: Whether to include text in output ('yes' or 'no')
        if_add_node_id: Whether to add node IDs ('yes' or 'no')
        max_input_tokens: Maximum tokens per LLM request
    """
    # Read the text file
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Processing text file with {len(text)} characters...")
    print(f"Using windowing approach with window_size={window_size}, overlap={overlap}")
    
    # Detect semantic sections in the text
    print("Detecting semantic sections...")
    sections = await detect_semantic_sections(text, model=model, max_input_tokens=max_input_tokens)
    
    print(f"Found {len(sections)} semantic sections")
    
    # Merge any overlapping sections
    sections = merge_overlapping_sections(sections)
    
    # Extract text for each section
    print("Extracting section content...")
    nodes_with_content = extract_section_text(text, sections)
    
    # Build tree structure
    print("Building tree structure...")
    tree_structure = build_tree_from_txt_nodes(nodes_with_content)
    
    # Add node IDs if requested
    if if_add_node_id == 'yes':
        write_node_id(tree_structure)
    
    print("Formatting tree structure...")
    
    # Handle summaries and text formatting
    if if_add_node_summary == 'yes':
        # Always include text for summary generation
        tree_structure = format_structure(tree_structure, 
                                         order=['title', 'node_id', 'summary', 'prefix_summary', 
                                               'text', 'char_start', 'char_end', 'nodes'])
        
        print("Generating summaries for each node...")
        tree_structure = await generate_summaries_for_txt_structure(
            tree_structure, 
            summary_token_threshold=summary_token_threshold, 
            model=model, 
            max_input_tokens=max_input_tokens
        )
        
        if if_add_node_text == 'no':
            # Remove text after summary generation if not requested
            tree_structure = format_structure(tree_structure, 
                                             order=['title', 'node_id', 'summary', 'prefix_summary', 
                                                   'char_start', 'char_end', 'nodes'])
        
        if if_add_doc_description == 'yes':
            print("Generating document description...")
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=model, 
                                                      max_input_tokens=max_input_tokens)
            return {
                'doc_name': os.path.splitext(os.path.basename(txt_path))[0],
                'doc_description': doc_description,
                'structure': tree_structure,
            }
    else:
        # No summaries needed, format based on text preference
        if if_add_node_text == 'yes':
            tree_structure = format_structure(tree_structure, 
                                             order=['title', 'node_id', 'summary', 'prefix_summary', 
                                                   'text', 'char_start', 'char_end', 'nodes'])
        else:
            tree_structure = format_structure(tree_structure, 
                                             order=['title', 'node_id', 'summary', 'prefix_summary', 
                                                   'char_start', 'char_end', 'nodes'])
    
    return {
        'doc_name': os.path.splitext(os.path.basename(txt_path))[0],
        'structure': tree_structure,
    }


if __name__ == "__main__":
    import os
    import json
    
    # Example usage
    TXT_NAME = 'sample'
    TXT_PATH = os.path.join(os.path.dirname(__file__), '..', 'tests/texts/', f'{TXT_NAME}.txt')
    
    MODEL = "gpt-4o-2024-11-20"
    WINDOW_SIZE = 5000
    OVERLAP = 500
    SUMMARY_TOKEN_THRESHOLD = 200
    IF_SUMMARY = True
    
    tree_structure = asyncio.run(txt_to_tree(
        txt_path=TXT_PATH,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        if_add_node_summary='yes' if IF_SUMMARY else 'no',
        summary_token_threshold=SUMMARY_TOKEN_THRESHOLD,
        model=MODEL
    ))
    
    print('\n' + '='*60)
    print('TREE STRUCTURE')
    print('='*60)
    print_json(tree_structure)
    
    print('\n' + '='*60)
    print('TABLE OF CONTENTS')
    print('='*60)
    print_toc(tree_structure['structure'])
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', f'{TXT_NAME}_structure.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree_structure, f, indent=2, ensure_ascii=False)
    
    print(f"\nTree structure saved to: {output_path}")
