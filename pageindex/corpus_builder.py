import json
import os
try:
    from .utils import write_node_id
except ImportError:
    from utils import write_node_id

def build_corpus(corpus_files: list, corpus_name: str = "Corpus") -> dict:
    """
    Takes a list of file paths to existing '*_structure.json' files,
    merges them into a single corpus tree under a root node, and renumbers
    the node_ids sequentially.
    """
    corpus_root = {
        "title": corpus_name,
        "nodes": []
    }
    
    total_line_count = 0

    for path in corpus_files:
        if not os.path.exists(path):
            print(f"Warning: Corpus file not found: {path}")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            try:
                doc_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse JSON from {path}")
                continue
        
        # Get the document title and structure
        doc_title = doc_data.get("doc_name", os.path.basename(path))
        structure = doc_data.get("structure", [])
        total_line_count += doc_data.get("line_count", 0)
        
        doc_node = {
            "title": doc_title,
            "nodes": structure
        }
        
        corpus_root["nodes"].append(doc_node)
        
    # Renumber the entire tree sequentially starting from 0001
    # write_node_id expects a list (or a dict) and modifies in place, 
    # and we want it to start from 1 (the default in write_node_id is 0 but it increments before stringifying, wait, let's check).
    # In utils.py:
    # def write_node_id(data, node_id=0):
    #     if isinstance(data, dict):
    #         data['node_id'] = str(node_id).zfill(4)
    #         node_id += 1 ...
    # So if we pass node_id=1, the root gets 0001.
    
    write_node_id([corpus_root], node_id=1)
    
    return {
        "doc_name": corpus_name,
        "line_count": total_line_count,
        "structure": [corpus_root]
    }
