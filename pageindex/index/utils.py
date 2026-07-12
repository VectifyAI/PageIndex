import litellm
import logging
import os
import textwrap
import time
import json
import copy
import re
import asyncio
import threading
import PyPDF2
import pymupdf
import yaml
from datetime import datetime
from io import BytesIO
from pathlib import Path
from pprint import pprint
# Aliased with a leading underscore so `from .utils import *` (used by the
# page_index modules) doesn't export a name `config` that would shadow the real
# `pageindex.config` submodule for those modules.
from types import SimpleNamespace as _config

from contextlib import asynccontextmanager, contextmanager

from ..config import (
    get_llm_params,
    get_max_concurrency,
    _max_concurrency_scope_semaphore,
    _process_wide_max_concurrency,
)
from ..tokens import count_tokens  # re-exported for backward compat

logger = logging.getLogger(__name__)


# TRUE process-wide ceiling on concurrent in-flight LLM calls, shared across
# EVERY thread and event loop (a plain threading.Semaphore, not an
# asyncio.Semaphore — those are bound to the loop that created them, so one per
# loop would let N concurrently-indexing threads each get their own full-size
# cap and multiply the effective bound by N). Resized lazily when the
# process-wide default changes; resizing isn't perfectly atomic against
# in-flight acquires, which is fine since it only happens on an explicit
# set_max_concurrency() config change, not on the hot path.
_PROCESS_LLM_SEMAPHORE: threading.Semaphore | None = None
_PROCESS_LLM_SEMAPHORE_SIZE: int | None = None
_PROCESS_LLM_SEMAPHORE_LOCK = threading.Lock()


def _process_ceiling_semaphore() -> threading.Semaphore:
    global _PROCESS_LLM_SEMAPHORE, _PROCESS_LLM_SEMAPHORE_SIZE
    size = _process_wide_max_concurrency()
    with _PROCESS_LLM_SEMAPHORE_LOCK:
        if _PROCESS_LLM_SEMAPHORE is None or _PROCESS_LLM_SEMAPHORE_SIZE != size:
            _PROCESS_LLM_SEMAPHORE = threading.Semaphore(size)
            _PROCESS_LLM_SEMAPHORE_SIZE = size
        return _PROCESS_LLM_SEMAPHORE


@asynccontextmanager
async def _llm_semaphore():
    """Bound concurrent in-flight LLM calls to a TRUE process-wide ceiling,
    optionally narrowed further by an active max_concurrency_scope() override.

    Acquired only around the leaf ``litellm.acompletion`` call in
    ``llm_acompletion`` — sync calls use ``_sync_llm_semaphore`` below — so the
    cap holds no matter how deeply the indexing gathers nest
    (``tree_parser`` → ``process_large_node_recursively`` → …) AND no matter how
    many threads are each running their own indexing job concurrently. Bounding
    at the leaf rather than at each gather call site is also deadlock-free: a
    parent coroutine awaiting its children holds no slot, so children can
    always acquire one.

    The process ceiling (threading.Semaphore, shared cross-thread) is sized from
    the process-wide default only; a narrower max_concurrency_scope() override
    is enforced as a second, nested context-local restriction — it can only
    *tighten* the effective cap for its own call tree, never widen it past the
    ceiling. Without the outer bound a many-node document opens one socket per
    node at once and exhausts the process file-descriptor limit (Errno 24).
    """
    ceiling_sem = _process_ceiling_semaphore()
    # A blocking ceiling_sem.acquire() run via asyncio.to_thread() would be
    # unsafe under cancellation: the worker thread can't be interrupted, so if
    # this coroutine is cancelled (Ctrl-C, an outer timeout) while the thread
    # is still parked inside acquire(), the thread can go on to actually
    # acquire a permit *after* we've already unwound — leaking it forever,
    # since the matching finally: release() below never runs for that attempt.
    # Poll with the non-blocking form instead: each check returns immediately
    # (no OS-level wait), so it's safe to call straight from the event loop
    # thread and there's no window for a background acquire to succeed after
    # we've already given up on it.
    while not ceiling_sem.acquire(False):
        await asyncio.sleep(0.05)
    # Only set once the permit is actually held, so a cancellation while polling
    # for it doesn't make the finally release a permit we never acquired (which
    # would inflate the scoped cap — the mirror of the ceiling leak fixed above).
    scoped_sem = None
    try:
        effective = get_max_concurrency()
        ceiling = _process_wide_max_concurrency()
        if effective < ceiling:
            candidate = _max_concurrency_scope_semaphore()
            if candidate is not None:
                while not candidate.acquire(False):
                    await asyncio.sleep(0.05)
                scoped_sem = candidate
        yield
    finally:
        if scoped_sem is not None:
            scoped_sem.release()
        ceiling_sem.release()


@contextmanager
def _sync_llm_semaphore():
    """Synchronous companion to ``_llm_semaphore`` for ``llm_completion``.

    It uses the same process-wide ceiling so sync and async LLM calls share one
    real cap. A scoped override can only narrow that cap for the active context.

    A *blocking* ceiling acquire is only safe OFF the event-loop thread. Several
    sync LLM helpers (``check_toc`` → ``toc_detector_single_page``,
    ``process_no_toc`` → ``generate_toc_init``, ``toc_transformer``, …) are
    called synchronously from inside async coroutines (``meta_processor`` →
    ``process_large_node_recursively``), i.e. ON the running loop. There, the
    async ``_llm_semaphore`` holders own the ceiling permits and can only
    release them by resuming on that same loop — so a blocking acquire here
    would freeze the loop and *deadlock*: the permit it waits for can never be
    freed. When we detect a running loop we therefore take a slot only if one is
    immediately free (non-blocking) and otherwise proceed without it. That's
    safe: a sync call monopolizes the loop thread while it runs, so it's already
    serialized on this loop and can't multiply the in-flight count beyond one
    extra per loop.
    """
    try:
        asyncio.get_running_loop()
        on_event_loop = True
    except RuntimeError:
        on_event_loop = False

    ceiling_sem = _process_ceiling_semaphore()
    # Blocking acquire() (off-loop) always returns True; acquire(False) (on-loop)
    # may return False, meaning "no free permit — proceed without one" rather
    # than block the loop into a deadlock.
    held_ceiling = ceiling_sem.acquire(False) if on_event_loop else ceiling_sem.acquire()
    # Only track a permit we actually hold (mirrors _llm_semaphore): guards
    # against releasing one we never acquired.
    scoped_sem = None
    try:
        effective = get_max_concurrency()
        ceiling = _process_wide_max_concurrency()
        if effective < ceiling:
            candidate = _max_concurrency_scope_semaphore()
            if candidate is not None:
                # Same rule for the scoped cap: never block the loop for it.
                if on_event_loop:
                    if candidate.acquire(False):
                        scoped_sem = candidate
                else:
                    candidate.acquire()
                    scoped_sem = candidate
        yield
    finally:
        if scoped_sem is not None:
            scoped_sem.release()
        if held_ceiling:
            ceiling_sem.release()


def llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
    if model:
        model = model.removeprefix("litellm/")
    max_retries = 10
    messages = list(chat_history) + [{"role": "user", "content": prompt}] if chat_history else [{"role": "user", "content": prompt}]
    for i in range(max_retries):
        try:
            # Hold a concurrency slot only around the actual network call, not
            # retry backoff, so sync completions obey the same cap as async ones.
            with _sync_llm_semaphore():
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    # Per-call litellm kwargs (default temperature=0, drop_params=True);
                    # configure via config.set_llm_params(...) — never the litellm global.
                    **get_llm_params(),
                )
            content = response.choices[0].message.content
            if return_finish_reason:
                finish_reason = "max_output_reached" if response.choices[0].finish_reason == "length" else "finished"
                return content, finish_reason
            return content
        except Exception as e:
            logger.warning("Retrying LLM completion (%d/%d)", i + 1, max_retries)
            logger.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)
            else:
                # Degrade gracefully instead of aborting the whole index: a single
                # persistently-failing call returns an empty result so callers can
                # skip that step (extract_json('') -> {} -> .get(default)) and the
                # rest of the document still gets indexed. Logged at WARNING so the
                # failure is visible, not silent.
                logger.warning(
                    "LLM completion failed after %d retries; degrading to an empty "
                    "result so the caller can skip this step. Last error: %s",
                    max_retries, e,
                )
                return ("", "error") if return_finish_reason else ""



async def llm_acompletion(model, prompt):
    if model:
        model = model.removeprefix("litellm/")
    max_retries = 10
    messages = [{"role": "user", "content": prompt}]
    for i in range(max_retries):
        try:
            # Hold a concurrency slot only around the actual network call — not
            # across retry backoff — so the cap counts real in-flight requests.
            async with _llm_semaphore():
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    **get_llm_params(),  # per-call kwargs; never the litellm global
                )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("Retrying async LLM completion (%d/%d)", i + 1, max_retries)
            logger.error(f"Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                # Degrade gracefully (see llm_completion): return an empty result
                # so the caller skips this step and the rest of the document still
                # indexes. The gather sites still keep return_exceptions=True to
                # absorb any non-LLM error. WARNING so it's visible, not silent.
                logger.warning(
                    "Async LLM completion failed after %d retries; degrading to an "
                    "empty result so the caller can skip this step. Last error: %s",
                    max_retries, e,
                )
                return ""


def extract_json(content):
    try:
        # First, try to extract JSON enclosed within ```json and ```
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7  # Adjust index to start after the delimiter
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            # If no delimiters, assume entire content could be JSON
            json_content = content.strip()

        # Clean up common issues that might cause parsing errors
        json_content = json_content.replace('None', 'null')  # Replace Python None with JSON null
        json_content = json_content.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
        json_content = ' '.join(json_content.split())  # Normalize whitespace

        # Attempt to parse and return the JSON object
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to extract JSON: {e}")
        # Try to clean up the content further if initial parsing fails
        try:
            # Remove any trailing commas before closing brackets/braces
            json_content = json_content.replace(',]', ']').replace(',}', '}')
            return json.loads(json_content)
        except Exception:
            logging.error("Failed to parse JSON even after cleanup")
            return {}
    except Exception as e:
        logging.error(f"Unexpected error while extracting JSON: {e}")
        return {}


def get_json_content(response):
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]

    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]

    json_content = response.strip()
    return json_content


def write_node_id(data, node_id=0):
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


def remove_fields(data, fields=None, max_len=None):
    fields = fields or ["text"]
    if isinstance(data, dict):
        return {k: remove_fields(v, fields, max_len)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields, max_len) for item in data]
    elif isinstance(data, str):
        return data[:max_len] + '...' if max_len is not None and len(data) > max_len else data
    return data


def structure_to_list(structure):
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


def get_nodes(structure):
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


def get_leaf_nodes(structure):
    if isinstance(structure, dict):
        # .get() — clean_node deletes the 'nodes' key on leaf nodes, so direct
        # indexing raises KeyError on a standard tree (issue #330 / #331).
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


async def generate_node_summary(node, model=None):
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {node['text']}

    Directly return the description, do not include any other text.
    """
    response = await llm_acompletion(model, prompt)
    return response


async def generate_summaries_for_structure(structure, model=None):
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    # return_exceptions=True: one node's summary failing (e.g. a transient LLM
    # error) must not abort summarization for the whole document — fall back
    # to the node's own raw text so retrieval still has something usable.
    raw_summaries = await asyncio.gather(*tasks, return_exceptions=True)
    summaries = [
        node.get('text', '') if isinstance(s, Exception) else s
        for node, s in zip(nodes, raw_summaries)
    ]

    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    return structure


def generate_doc_description(structure, model=None):
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.

    Document Structure: {structure}

    Directly return the description, do not include any other text.
    """
    response = llm_completion(model, prompt)
    return response


def list_to_tree(data):
    def get_parent_structure(structure):
        """Helper function to get the parent structure code"""
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None

    # First pass: Create nodes and track parent-child relationships
    nodes = {}
    root_nodes = []

    for item in data:
        structure = item.get('structure')
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
    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node

    # Clean and return the tree
    return [clean_node(node) for node in root_nodes]


def post_processing(structure, end_physical_index):
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


def reorder_dict(data, key_order):
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


def create_clean_structure_for_description(structure):
    """
    Create a clean structure for document description generation,
    excluding unnecessary fields like 'text'.
    """
    if isinstance(structure, dict):
        clean_node = {}
        # Only include essential fields for description
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]

        # Recursively process child nodes
        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_description(structure['nodes'])

        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    else:
        return structure


def _get_text_of_pages(page_list, start_page, end_page):
    """Concatenate text from page_list for pages [start_page, end_page] (1-indexed)."""
    text = ""
    for page_num in range(start_page - 1, end_page):
        text += page_list[page_num][0]
    return text


def add_node_text(node, page_list):
    """Recursively add 'text' field to each node from page_list content.

    Each node must have 'start_index' and 'end_index' (1-indexed page numbers).
    page_list is [(page_text, token_count), ...].
    """
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        if start_page is not None and end_page is not None:
            node['text'] = _get_text_of_pages(page_list, start_page, end_page)
        if 'nodes' in node:
            add_node_text(node['nodes'], page_list)
    elif isinstance(node, list):
        for item in node:
            add_node_text(item, page_list)


def remove_structure_text(data):
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


# ── Functions migrated from retrieve.py ──────────────────────────────────────

_MAX_PAGES = 1000


def parse_pages(pages: str) -> list[int]:
    """Parse a pages string like '5-7', '3,8', or '12' into a sorted list of ints."""
    result = []
    for part in pages.split(','):
        part = part.strip()
        if '-' in part:
            start, end = int(part.split('-', 1)[0].strip()), int(part.split('-', 1)[1].strip())
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
            # Bound the span BEFORE materializing range() into the list. Checking
            # len(result) only after `result.extend(range(...))` is too late: a
            # single huge span like '1-2000000000' allocates billions of ints
            # and exhausts memory before the cap is ever reached (DoS). page_nums
            # is attacker/LLM-reachable via get_page_content.
            span = end - start + 1
            if span > _MAX_PAGES or len(result) + span > _MAX_PAGES:
                raise ValueError(f"Page range too large: max {_MAX_PAGES} pages")
            result.extend(range(start, end + 1))
        else:
            if len(result) + 1 > _MAX_PAGES:
                raise ValueError(f"Page range too large: max {_MAX_PAGES} pages")
            result.append(int(part))
    result = [p for p in result if p >= 1]
    result = sorted(set(result))
    if len(result) > _MAX_PAGES:
        raise ValueError(f"Page range too large: {len(result)} pages (max {_MAX_PAGES})")
    return result


def get_pdf_page_content(file_path: str, page_nums: list[int]) -> list[dict]:
    """Extract text for specific PDF pages (1-indexed), opening the PDF once."""
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total = len(pdf_reader.pages)
        valid_pages = [p for p in page_nums if 1 <= p <= total]
        return [
            {'page': p, 'content': pdf_reader.pages[p - 1].extract_text() or ''}
            for p in valid_pages
        ]


def get_md_page_content(structure: list, page_nums: list[int]) -> list[dict]:
    """
    For Markdown documents, 'pages' are line numbers.
    Return only the nodes whose line_num is one of ``page_nums`` (exact match),
    mirroring the PDF path. A non-contiguous spec like [5, 100] returns just
    those two lines, not the whole [5, 100] range.
    """
    if not page_nums:
        return []
    wanted = set(page_nums)
    results = []
    seen = set()

    def _traverse(nodes):
        for node in nodes:
            ln = node.get('line_num')
            if ln in wanted and ln not in seen:
                seen.add(ln)
                results.append({'page': ln, 'content': node.get('text', '')})
            if node.get('nodes'):
                _traverse(node['nodes'])

    _traverse(structure)
    results.sort(key=lambda x: x['page'])
    return results



# ─────────────────────────────────────────────────────────────────────
# Legacy 0.2.x / OSS utility API — kept here so this module is the single
# source of truth for the indexing pipeline. Previously duplicated in the
# top-level pageindex/utils.py (now a deprecation shim re-exporting this).
# ─────────────────────────────────────────────────────────────────────

async def call_llm(prompt, api_key, model="gpt-4.1", temperature=0):
    """Call an LLM to generate a response to a prompt.

    Kept for compatibility with the pageindex 0.2.x SDK utility API.
    """
    import openai

    async with openai.AsyncOpenAI(api_key=api_key) as client:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
    return response.choices[0].message.content.strip()


def is_leaf_node(data, node_id):
    # Helper function to find the node by its node_id
    def find_node(data, node_id):
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


def get_last_node(structure):
    return structure[-1]


def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    ###return text not list 
    text=""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text+=page.extract_text()
    return text


def get_pdf_title(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title


def get_text_of_pages(pdf_path, start_page, end_page, tag=True):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page-1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num+1}>\n{page_text}\n<end_index_{page_num+1}>\n"
        else:
            text += page_text
    return text


def get_first_start_page_from_text(text):
    start_page = -1
    start_page_match = re.search(r'<start_index_(\d+)>', text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page


def get_last_start_page_from_text(text):
    start_page = -1
    # Find all matches of start_index tags
    start_page_matches = re.finditer(r'<start_index_(\d+)>', text)
    # Convert iterator to list and get the last match if any exist
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename, replacement='-'):
    # In Linux, only '/' and '\0' (null) are invalid in filenames.
    # Null can't be represented in strings, so we only handle '/'.
    return filename.replace('/', replacement)


def get_pdf_name(pdf_path):
    # Extract PDF name
    if isinstance(pdf_path, (str, Path)):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else 'Untitled'
        pdf_name = sanitize_filename(pdf_name)
    else:
        pdf_name = 'Untitled'
    return pdf_name


class JsonLogger:
    def __init__(self, file_path):
        # Extract PDF name for logger name
        pdf_name = get_pdf_name(file_path)
            
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        # Initialize empty list to store all messages
        self.log_data = []

    def log(self, level, message, **kwargs):
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({'message': message})
        # Add new message to the log data
        
        # Write entire log data to file
        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def info(self, message, **kwargs):
        self.log("INFO", message, **kwargs)

    def error(self, message, **kwargs):
        self.log("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        self.log("DEBUG", message, **kwargs)

    def exception(self, message, **kwargs):
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self):
        return os.path.join("logs", self.filename)


def add_preface_if_needed(data):
    if not isinstance(data, list) or not data:
        return data

    if data[0]['physical_index'] is not None and data[0]['physical_index'] > 1:
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data


def get_page_tokens(pdf_path, model=None, pdf_parser="PyPDF2"):
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = litellm.token_counter(model=model, text=page_text)
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            pdf_stream = pdf_path
            doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = litellm.token_counter(model=model, text=page_text)
            page_list.append((page_text, token_length))
        return page_list
    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")


def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += pdf_pages[page_num][0]
    return text


def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
    return text


def get_number_of_pages(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    num = len(pdf_reader.pages)
    return num


def clean_structure_post(data):
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


def print_toc(tree, indent=0):
    for node in tree:
        print('  ' * indent + node['title'])
        if node.get('nodes'):
            print_toc(node['nodes'], indent + 1)


def print_json(data, max_len=40, indent=2):
    def simplify_data(obj):
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


def check_token_limit(structure, limit=110000):
    list = structure_to_list(structure)
    for node in list:
        num_tokens = count_tokens(node['text'], model=None)
        if num_tokens > limit:
            print(f"Node ID: {node['node_id']} has {num_tokens} tokens")
            print("Start Index:", node['start_index'])
            print("End Index:", node['end_index'])
            print("Title:", node['title'])
            print("\n")


def convert_physical_index_to_int(data):
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


def convert_page_to_int(data):
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                # Keep original value if conversion fails
                pass
    return data


def add_node_text_with_labels(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text_with_labels(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)
    return


class ConfigLoader:
    """Legacy 0.2.x config helper. Defaults now come from IndexConfig — the
    old ``config.yaml`` no longer ships. Prefer ``pageindex.IndexConfig``.
    """

    def __init__(self, default_path=None):
        from ..config import IndexConfig
        self._default_dict = IndexConfig().model_dump()

    def _validate_keys(self, user_dict):
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> _config:
        """Merge user options over IndexConfig defaults, returning a namespace."""
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, _config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        # Route through IndexConfig so legacy 'yes'/'no' string overrides get
        # pydantic's bool coercion (a bare 'no' is otherwise a truthy string —
        # page_index_main's `if opt.if_add_node_summary:` checks would silently
        # invert the caller's intent).
        from ..config import IndexConfig
        validated = IndexConfig(**merged)
        return _config(**validated.model_dump())


def create_node_mapping(tree, include_page_ranges=False, max_page=None):
    """Create a mapping of node_id to node for quick lookup.

    The optional page-range arguments are kept for compatibility with the
    pageindex 0.2.x SDK utility API.
    """
    def get_all_nodes(nodes):
        if isinstance(nodes, dict):
            return [nodes] + [
                child_node
                for child in nodes.get('nodes', [])
                for child_node in get_all_nodes(child)
            ]
        elif isinstance(nodes, list):
            return [
                child_node
                for item in nodes
                for child_node in get_all_nodes(item)
            ]
        return []

    all_nodes = get_all_nodes(tree)

    if not include_page_ranges:
        return {node["node_id"]: node for node in all_nodes if node.get("node_id")}

    mapping = {}
    for i, node in enumerate(all_nodes):
        if not node.get("node_id"):
            continue
        start_page = node.get("page_index", node.get("start_index"))
        if node.get("end_index") is not None:
            end_page = node.get("end_index")
        elif i + 1 < len(all_nodes):
            next_node = all_nodes[i + 1]
            end_page = next_node.get("page_index", next_node.get("start_index"))
        else:
            end_page = max_page

        mapping[node["node_id"]] = {
            "node": node,
            "start_index": start_page,
            "end_index": end_page,
        }

    return mapping


def print_tree(tree, exclude_fields=None, indent=None):
    if exclude_fields is None:
        exclude_fields = ['text', 'page_index']
    if isinstance(exclude_fields, int):
        indent = exclude_fields
        exclude_fields = None
    if indent is None and exclude_fields is not None:
        cleaned_tree = remove_fields(copy.deepcopy(tree), exclude_fields, max_len=40)
        pprint(cleaned_tree, sort_dicts=False, width=100)
        return

    indent = indent or 0
    for node in tree:
        summary = node.get('summary') or node.get('prefix_summary', '')
        summary_str = f"  —  {summary[:60]}..." if summary else ""
        print('  ' * indent + f"[{node.get('node_id', '?')}] {node.get('title', '')}{summary_str}")
        if node.get('nodes'):
            print_tree(node['nodes'], exclude_fields=exclude_fields, indent=indent + 1)


def print_wrapped(text, width=100):
    for line in text.splitlines():
        print(textwrap.fill(line, width=width))
