import copy
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from .llm import count_tokens, ChatGPT_API, ChatGPT_API_async

# Type aliases for tree structures
Node = Dict[str, Any]
Tree = List[Node]
Structure = Union[Node, List[Any]] # Recursive definition limitation in MyPy, using Any for nested

def write_node_id(data: Structure, node_id: int = 0) -> int:
    """
    Recursively assign sequential node_ids to a tree structure.
    
    Args:
        data (Structure): The tree or node to process.
        node_id (int): The starting ID.
        
    Returns:
        int: The next available node_id.
    """
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id

def get_nodes(structure: Structure) -> List[Node]:
    """
    Flatten the tree into a list of nodes, excluding their children 'nodes' list from the copy.
    
    Args:
        structure (Structure): The tree structure.
        
    Returns:
        List[Node]: A flat list of node dictionaries (without 'nodes' key).
    """
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop('nodes', None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if 'nodes' in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    return []
    
def structure_to_list(structure: Structure) -> List[Node]:
    """
    Flatten the tree into a list of references to all nodes (including containers).
    
    Args:
        structure (Structure): The tree structure.
        
    Returns:
        List[Node]: Flat list of all nodes.
    """
    if isinstance(structure, dict):
        nodes = []
        nodes.append(structure)
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes
    return []

    
def get_leaf_nodes(structure: Structure) -> List[Node]:
    """
    Get all leaf nodes (nodes with no children).
    
    Args:
        structure (Structure): The tree structure.
        
    Returns:
        List[Node]: List of leaf node copies (without 'nodes' key).
    """
    if isinstance(structure, dict):
        if not structure.get('nodes'):
            structure_node = copy.deepcopy(structure)
            structure_node.pop('nodes', None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if 'nodes' in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes
    return []

def is_leaf_node(data: Structure, node_id: str) -> bool:
    """
    Check if a node with specific ID is a leaf node.
    
    Args:
        data (Structure): The tree structure.
        node_id (str): The ID to check.
        
    Returns:
        bool: True if node exists and has no children.
    """
    # Helper function to find the node by its node_id
    def find_node(data: Structure, node_id: str) -> Optional[Node]:
        if isinstance(data, dict):
            if data.get('node_id') == node_id:
                return data
            for key in data.keys():
                if 'nodes' in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    # Find the node with the given node_id
    node = find_node(data, node_id)

    # Check if the node is a leaf node
    if node and not node.get('nodes'):
        return True
    return False

def get_last_node(structure: List[Any]) -> Any:
    """Get the last element of a list structure."""
    return structure[-1]

def list_to_tree(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a flat list of nodes with dot-notation 'structure' keys (e.g., '1.1') 
    into a nested tree.
    
    Args:
        data (List[Dict[str, Any]]): List of node dictionaries.
        
    Returns:
        List[Dict[str, Any]]: The nested tree structure.
    """
    def get_parent_structure(structure: Optional[str]) -> Optional[str]:
        """Helper function to get the parent structure code"""
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None
    
    # First pass: Create nodes and track parent-child relationships
    nodes: Dict[str, Dict[str, Any]] = {}
    root_nodes: List[Dict[str, Any]] = []
    
    for item in data:
        structure = str(item.get('structure', ''))
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': []
        }
        
        nodes[structure] = node
        
        # Find parent
        parent_structure = get_parent_structure(structure)
        
        if parent_structure:
            # Add as child to parent if parent exists
            if parent_structure in nodes:
                nodes[parent_structure]['nodes'].append(node)
            else:
                root_nodes.append(node)
        else:
            # No parent, this is a root node
            root_nodes.append(node)
    
    # Helper function to clean empty children arrays
    def clean_node(node: Dict[str, Any]) -> Dict[str, Any]:
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node
    
    # Clean and return the tree
    return [clean_node(node) for node in root_nodes]

def add_preface_if_needed(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Inject a Preface node if the first node starts after page 1.
    """
    if not isinstance(data, list) or not data:
        return data

    if data[0].get('physical_index') is not None and data[0]['physical_index'] > 1:
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data


def post_processing(structure: List[Dict[str, Any]], end_physical_index: int) -> Union[List[Dict[str, Any]], List[Any]]:
    """
    Calculate start/end indices based on 'physical_index' and convert to tree if possible.
    
    Args:
        structure: List of flat nodes.
        end_physical_index: Total pages or end index.
        
    Returns:
        Tree or List.
    """
    # First convert page_number to start_index in flat list
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index')
        if i < len(structure) - 1:
            if structure[i + 1].get('appear_start') == 'yes':
                item['end_index'] = structure[i + 1]['physical_index']-1
            else:
                item['end_index'] = structure[i + 1]['physical_index']
        else:
            item['end_index'] = end_physical_index
    tree = list_to_tree(structure)
    if len(tree)!=0:
        return tree
    else:
        ### remove appear_start 
        for node in structure:
            node.pop('appear_start', None)
            node.pop('physical_index', None)
        return structure

def clean_structure_post(data: Structure) -> Structure:
    """Recursively clean internal processing fields from structure."""
    if isinstance(data, dict):
        data.pop('page_number', None)
        data.pop('start_index', None)
        data.pop('end_index', None)
        if 'nodes' in data:
            clean_structure_post(data['nodes'])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data

def remove_fields(data: Structure, fields: List[str] = ['text']) -> Structure:
    """Recursively remove specified fields from the structure."""
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data

def print_toc(tree: List[Dict[str, Any]], indent: int = 0) -> None:
    """Print Table of Contents to stdout."""
    for node in tree:
        print('  ' * indent + str(node.get('title', '')))
        if node.get('nodes'):
            print_toc(node['nodes'], indent + 1)

def print_json(data: Any, max_len: int = 40, indent: int = 2) -> None:
    """Pretty print JSON with truncated strings."""
    def simplify_data(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + '...'
        else:
            return obj
    
    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def print_wrapped(text: Any, width: int = 100) -> None:
    """Print text wrapped to specified width."""
    import textwrap

    if text is None:
        return
    for line in str(text).splitlines():
        if not line.strip():
            print()
            continue
        for wrapped in textwrap.wrap(line, width=width):
            print(wrapped)


def print_tree(tree: List[Dict[str, Any]], exclude_fields: Optional[List[str]] = None, indent: int = 0, max_summary_len: int = 120) -> None:
    """Print tree structure with node IDs and summaries."""
    if exclude_fields:
        # Cast to Any to satisfy mypy since remove_fields returns Structure
        tree = remove_fields(tree, fields=exclude_fields) # type: ignore
        
    for node in tree:
        node_id = node.get('node_id', '')
        title = node.get('title', '')
        start = node.get('start_index')
        end = node.get('end_index')
        summary = node.get('summary') or node.get('prefix_summary')
        page_range = None
        if start is not None and end is not None:
            page_range = start if start == end else f"{start}-{end}"
        line = f"{node_id}\t{page_range}\t{title}" if page_range else f"{node_id}\t{title}"
        if summary:
            short_summary = summary if len(summary) <= max_summary_len else summary[:max_summary_len] + '...'
            line = f"{line} — {short_summary}"
        print('  ' * indent + line)
        if node.get('nodes'):
            print_tree(node['nodes'], exclude_fields=exclude_fields, indent=indent + 1, max_summary_len=max_summary_len)


def create_node_mapping(tree: List[Dict[str, Any]], include_page_ranges: bool = False, max_page: Optional[int] = None) -> Dict[str, Any]:
    """Create a dictionary mapping node_ids to nodes."""
    mapping = {}

    def clamp_page(value: Optional[int]) -> Optional[int]:
        if value is None or max_page is None:
            return value
        return max(1, min(value, max_page))

    def visit(node: Dict[str, Any]) -> None:
        node_id = node.get('node_id')
        if node_id:
            if include_page_ranges:
                start = clamp_page(node.get('start_index'))
                end = clamp_page(node.get('end_index'))
                mapping[node_id] = {
                    'node': node,
                    'start_index': start,
                    'end_index': end,
                }
            else:
                mapping[node_id] = node
        for child in node.get('nodes') or []:
            visit(child)

    for root in tree:
        visit(root)

    return mapping


def remove_structure_text(data: Structure) -> Structure:
    """Recursively remove 'text' field."""
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure: Structure, limit: int = 110000) -> None:
    """Check if any node exceeds the token limit."""
    flat_list = structure_to_list(structure)
    for node in flat_list:
        text = node.get('text', '')
        num_tokens = count_tokens(text, model='gpt-4o')
        if num_tokens > limit:
            print(f"Node ID: {node.get('node_id')} has {num_tokens} tokens")
            print("Start Index:", node.get('start_index'))
            print("End Index:", node.get('end_index'))
            print("Title:", node.get('title'))
            print("\n")


def convert_physical_index_to_int(data: Any) -> Any:
    """Convert physical_index strings (e.g., '<physical_index_5>') to integers inplace."""
    if isinstance(data, list):
        for i in range(len(data)):
            # Check if item is a dictionary and has 'physical_index' key
            if isinstance(data[i], dict) and 'physical_index' in data[i]:
                if isinstance(data[i]['physical_index'], str):
                    if data[i]['physical_index'].startswith('<physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].rstrip('>').strip())
                    elif data[i]['physical_index'].startswith('physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].strip())
    elif isinstance(data, str):
        if data.startswith('<physical_index_'):
            data = int(data.split('_')[-1].rstrip('>').strip())
        elif data.startswith('physical_index_'):
            data = int(data.split('_')[-1].strip())
        # Check data is int
        if isinstance(data, int):
            return data
        else:
            return None
    return data


def convert_page_to_int(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert 'page' field to int if possible."""
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                # Keep original value if conversion fails
                pass
    return data

from .pdf import get_text_of_pdf_pages, get_text_of_pdf_pages_with_labels

def add_node_text(node: Structure, pdf_pages: List[Any]) -> None:
    """Recursively add text to nodes from pdf_pages list based on page range."""
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        if start_page is not None and end_page is not None:
             node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)
    return


def add_node_text_with_labels(node: Structure, pdf_pages: List[Any]) -> None:
    """Recursively add text with physical index labels."""
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        if start_page is not None and end_page is not None:
             node['text'] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text_with_labels(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)
    return


async def generate_node_summary(node: Dict[str, Any], model: Optional[str] = None) -> str:
    """Generate summary for a node using LLM."""
    # Ensure text exists
    text = node.get('text', '')
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {text}
    
    Directly return the description, do not include any other text.
    """
    # Note: model name should ideally be passed, default handled in API
    response = await ChatGPT_API_async(model or "gpt-4o", prompt)
    return response


async def generate_summaries_for_structure(structure: Structure, model: Optional[str] = None) -> Structure:
    """Generate summaries for all nodes in the structure."""
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    return structure


def create_clean_structure_for_description(structure: Structure) -> Structure:
    """
    Create a clean structure for document description generation,
    excluding unnecessary fields like 'text'.
    """
    if isinstance(structure, dict):
        clean_node: Dict[str, Any] = {}
        # Only include essential fields for description
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]
        
        # Recursively process child nodes
        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_description(structure['nodes'])
        
        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure] # type: ignore
    else:
        return structure


def generate_doc_description(structure: Structure, model: str = "gpt-4o") -> str:
    """Generate a one-sentence description for the entire document structure."""
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.
        
    Document Structure: {structure}
    
    Directly return the description, do not include any other text.
    """
    response = ChatGPT_API(model, prompt)
    return response


def reorder_dict(data: Dict[str, Any], key_order: List[str]) -> Dict[str, Any]:
    """Reorder dictionary keys."""
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure: Structure, order: Optional[List[str]] = None) -> Structure:
    """Recursively format and reorder keys in the structure."""
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure] # type: ignore
    return structure
