import os
try:
    from .utils import *
except:
    from utils import *

from .page_index_md import generate_summaries_for_structure_md


def _read_text_file(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(txt_path, 'r', encoding='latin-1') as f:
            return f.read()


async def txt_to_tree(txt_path, if_add_node_summary='no', summary_token_threshold=200, model=None, if_add_doc_description='no', if_add_node_text='yes', if_add_node_id='yes'):
    text = _read_text_file(txt_path)
    line_count = text.count('\n') + 1

    doc_name = os.path.splitext(os.path.basename(txt_path))[0]
    tree_structure = [{
        'title': doc_name,
        'node_id': '0001',
        'text': text,
        'line_num': 1,
        'nodes': [],
    }]

    if if_add_node_id == 'yes':
        write_node_id(tree_structure)

    if if_add_node_summary == 'yes':
        tree_structure = format_structure(tree_structure, order=['title', 'node_id', 'line_num', 'summary', 'prefix_summary', 'text', 'nodes'])
        tree_structure = await generate_summaries_for_structure_md(tree_structure, summary_token_threshold=summary_token_threshold, model=model)

        if if_add_node_text == 'no':
            tree_structure = format_structure(tree_structure, order=['title', 'node_id', 'line_num', 'summary', 'prefix_summary', 'nodes'])

        if if_add_doc_description == 'yes':
            clean_structure = create_clean_structure_for_description(tree_structure)
            doc_description = generate_doc_description(clean_structure, model=model)
            return {
                'doc_name': doc_name,
                'doc_description': doc_description,
                'line_count': line_count,
                'structure': tree_structure,
            }
    else:
        if if_add_node_text == 'yes':
            tree_structure = format_structure(tree_structure, order=['title', 'node_id', 'line_num', 'summary', 'prefix_summary', 'text', 'nodes'])
        else:
            tree_structure = format_structure(tree_structure, order=['title', 'node_id', 'line_num', 'summary', 'prefix_summary', 'nodes'])

    return {
        'doc_name': doc_name,
        'line_count': line_count,
        'structure': tree_structure,
    }
