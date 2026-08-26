import contextvars
import logging
import os
import sys
import textwrap
from datetime import datetime
import time
import json
import PyPDF2
import copy
import asyncio
import heapq
from contextlib import asynccontextmanager
from io import BytesIO
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(usecwd=True))
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace as config
import re

# litellm is imported inside the functions that use it; eager import is slow
# and fetches a remote model-cost map.


# The indexing lane's connection overrides, scoped by LocalAPI around each
# indexing operation — a contextvar, so the value reaches this module's
# helpers and their asyncio tasks without threading it through every call.
_llm_backend: contextvars.ContextVar = contextvars.ContextVar(
    "pageindex_llm_backend", default=None)


def _repair_litellm_types() -> None:
    """litellm 1.97.0's Message/Delta annotations carry nested forward refs
    Python 3.10 cannot resolve (BerriAI/litellm#36384), so every completion
    dies constructing its response. Rebuild them once with the defining
    modules' names; no-op on 3.11+ and on fixed litellm releases."""
    if sys.version_info >= (3, 11):
        return
    try:
        import litellm.types.llms.openai as openai_types
        import litellm.types.utils as litellm_types
        namespace = {**vars(openai_types), **vars(litellm_types)}
        litellm_types.Message.model_rebuild(_types_namespace=namespace)
        litellm_types.Delta.model_rebuild(_types_namespace=namespace)
    except Exception:
        pass  # best-effort: a failed repair leaves litellm's own error


def _mute_litellm_bridge_usage_warning() -> None:
    """litellm's chat→Responses bridge (e.g. OpenAI gpt-5.4+ with function
    tools) logs a chat-shaped usage dict inside a ResponseAPIUsage field
    (litellm_logging._get_assembled_streaming_response, 1.97–1.98), and
    pydantic reports it on every streamed turn. Hide exactly that message;
    every other warning still surfaces."""
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r"Pydantic serializer warnings:\s+"
                r"(PydanticSerializationUnexpectedValue\()?Expected `ResponseAPIUsage`")

# Backward compatibility: support CHATGPT_API_KEY as alias for OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
    import warnings
    warnings.warn("CHATGPT_API_KEY is deprecated — set OPENAI_API_KEY "
                  "instead.", FutureWarning)
    os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")

def count_tokens(text, model=None):
    if not text:
        return 0
    import litellm
    return litellm.token_counter(model=model, text=text)


def _strip_prefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    return s


def run_off_loop(func, *args):
    """Run func now, or on a worker thread when this thread already runs an
    asyncio loop (func may itself call asyncio.run)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return func(*args)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(func, *args).result()


def _litellm_model(model):
    """Normalize to LiteLLM's grammar (``litellm/`` strips, bare names get
    the ``openai/`` wire form — same as the chat lane) and refuse an
    unknown provider with the 404 the retry loop treats as unrecoverable.
    Credentials are LiteLLM's own call, made at the first completion."""
    if not model:
        return model
    model = _strip_prefix(model, "litellm/")
    if "/" not in model:
        model = f"openai/{model}"
    import litellm
    provider = model.split("/", 1)[0]
    providers = getattr(litellm, "provider_list", None)
    # custom_provider_map providers join provider_list only at call time.
    custom = {entry.get("provider") for entry
              in getattr(litellm, "custom_provider_map", None) or []}
    if providers and provider not in providers and provider not in custom:
        raise litellm.NotFoundError(
            f"'{model}' routes through LiteLLM, but '{provider}' is not a "
            f"LiteLLM provider. For an OpenAI-compatible server serving "
            f"this model id, use 'openai/{model}' and point "
            f"OPENAI_BASE_URL at the server.",
            llm_provider=None, model=model)
    return model


# Misconfiguration: no retry can fix a rejected key or a model that does not
# exist, and every later call fails the same way. An unknown status is a
# transport failure and stays retryable.
_UNRECOVERABLE_STATUS = frozenset({401, 403, 404})

# A 400 (context_length_exceeded) is equally unfixable by retry — the prompt
# will not shrink — but it is per-prompt: the ladder raises it immediately
# and consumers absorb it instead of failing the run.
_NO_RETRY_STATUS = _UNRECOVERABLE_STATUS | frozenset({400})


class LLMRetriesExhausted(RuntimeError):
    """The retry ladder gave up; carries the last error's status_code."""

    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _is_unrecoverable(exc: Exception) -> bool:
    if isinstance(exc, LLMRetriesExhausted):
        # 400 carries context_length_exceeded, the per-prompt failure the
        # caller absorbs (see above); any other exhausted ladder is fatal.
        return exc.status_code != 400
    return getattr(exc, "status_code", None) in _UNRECOVERABLE_STATUS


def llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
    import litellm
    max_retries = 10
    messages = list(chat_history) + [{"role": "user", "content": prompt}] if chat_history else [{"role": "user", "content": prompt}]
    backend = _llm_backend.get()
    model = _litellm_model(model)
    _repair_litellm_types()
    for i in range(max_retries):
        try:
            response = litellm.completion(**{
                "model": model,
                "messages": messages,
                "drop_params": True,
                # the loop is the retry policy; the merge lets a backend override win
                "max_retries": 0,
                **(backend or {}),
            })
            content = response.choices[0].message.content
            if return_finish_reason:
                finish_reason = "max_output_reached" if response.choices[0].finish_reason == "length" else "finished"
                return content, finish_reason
            return content
        except Exception as e:
            if getattr(e, "status_code", None) in _NO_RETRY_STATUS:
                raise
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)
            else:
                raise LLMRetriesExhausted(
                    f"LLM completion failed after {max_retries} retries: {e}",
                    status_code=getattr(e, "status_code", None),
                ) from e


async def llm_acompletion(model, prompt):
    import litellm
    max_retries = 10
    messages = [{"role": "user", "content": prompt}]
    backend = _llm_backend.get()
    model = _litellm_model(model)
    _repair_litellm_types()
    for i in range(max_retries):
        try:
            response = await litellm.acompletion(**{
                "model": model,
                "messages": messages,
                "drop_params": True,
                "max_retries": 0,
                **(backend or {}),
            })
            return response.choices[0].message.content
        except Exception as e:
            if getattr(e, "status_code", None) in _NO_RETRY_STATUS:
                raise
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                raise LLMRetriesExhausted(
                    f"LLM completion failed after {max_retries} retries: {e}",
                    status_code=getattr(e, "status_code", None),
                ) from e


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

    
def get_leaf_nodes(structure):
    if isinstance(structure, dict):
        if not structure['nodes']:
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
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else 'Untitled'
        pdf_name = sanitize_filename(pdf_name)
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
    import litellm
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
        import pymupdf
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
    if start_page is None or end_page is None:
        return ""
    text = ""
    for page_num in range(start_page-1, end_page):
        text += pdf_pages[page_num][0]
    return text

def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    if start_page is None or end_page is None:
        return ""
    text = ""
    for page_num in range(start_page-1, end_page):
        text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
    return text

def get_number_of_pages(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    num = len(pdf_reader.pages)
    return num



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

def remove_fields(data, fields=['text'], max_len=None):
    if isinstance(data, dict):
        return {k: remove_fields(v, fields, max_len)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields, max_len) for item in data]
    elif isinstance(data, str):
        return data[:max_len] + '...' if max_len is not None and len(data) > max_len else data
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


def remove_structure_text(data):
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


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


def add_node_text(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)
    return


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
    summaries = await asyncio.gather(*tasks, return_exceptions=True)

    for node, summary in zip(nodes, summaries):
        if isinstance(summary, Exception) and _is_unrecoverable(summary):
            raise summary
        node['summary'] = "" if isinstance(summary, BaseException) else summary
    if nodes and not any(node['summary'] for node in nodes):
        raise RuntimeError(
            "Summary generation failed for all nodes "
            "(every summary call failed or returned empty; "
            "check the model and its context limits)"
        )
    return structure


SUMMARY_CONCURRENCY = 64        # simultaneous summary model calls
SUMMARY_RAW_TEXT_TOKENS = 200   # leaves under this reuse their raw text as the summary
SUMMARY_INTRO_MAX_PAGES = 3     # cap on leading pages fed into a parent summary


class _PriorityGate:
    """Semaphore that admits the highest-priority waiter first, FIFO within a priority."""

    def __init__(self, permits):
        if permits < 1:
            raise ValueError("permits must be >= 1")
        self._free = permits
        self._waiters = []
        self._seq = 0

    @asynccontextmanager
    async def slot(self, prio):
        await self.acquire(prio)
        try:
            yield
        finally:
            self.release()

    async def acquire(self, prio):
        if self._free > 0 and not self._waiters:
            self._free -= 1
            return
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (-prio, self._seq, fut))
        self._seq += 1
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()  # granted while cancelling: pass the permit on
            raise

    def release(self):
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)  # the permit moves straight to this waiter
                return
        self._free += 1


def get_intro_text(node, pdf_pages, max_pages=SUMMARY_INTRO_MAX_PAGES):
    """Pages of the node covered by no child: from its start to just before the
    first child starts. Empty when the first child opens on the node's own page."""
    children = node.get('nodes') or []
    first = children[0].get('start_index') if children else None
    if not isinstance(first, int) or first <= node['start_index']:
        return ""
    end = min(first - 1, node['start_index'] + max_pages - 1)
    return get_text_of_pdf_pages(pdf_pages, node['start_index'], end)


def _reply_json(reply):
    """The JSON object in a model reply, or None when none of it parses.

    Not extract_json: that rewrites `None` to `null` and collapses whitespace in
    replies that parse as written.
    """
    if not isinstance(reply, str) or not reply.strip():
        return None
    text = reply.strip()
    if '```' in text:
        text = re.sub(r'^.*?```(?:json)?\s*', '', text, flags=re.S).split('```')[0]
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end <= start:
        return None
    obj = text[start:end + 1]
    collapsed = ' '.join(obj.split())
    # repairs, tried only once the reply fails to parse as written
    for candidate in (obj, collapsed, collapsed.replace(',]', ']').replace(',}', '}')):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def parse_summary(reply):
    """The `summary` field of a model reply, or the reply itself when there is no
    such field."""
    if not isinstance(reply, str) or not reply.strip():
        return ""
    parsed = _reply_json(reply)
    if isinstance(parsed, dict) and 'summary' in parsed:
        summary = parsed['summary']
        if isinstance(summary, list):
            summary = ' '.join(str(item).strip() for item in summary if str(item).strip())
        return str(summary).strip() if summary else ""
    return reply.strip()


def parse_title(reply):
    """The `title` field of a model reply, or "" when it is absent or unusable.

    Unlike parse_summary there is no falling back to the raw reply: a title that
    did not come back as a named field is not a title, and the caller keeps the
    deterministic one it already has.
    """
    parsed = _reply_json(reply)
    if not isinstance(parsed, dict):
        return ""
    title = parsed.get('title')
    if isinstance(title, list):
        title = ' '.join(str(item).strip() for item in title if str(item).strip())
    return ' '.join(str(title).split()) if title else ""


def strip_internal_keys(structure):
    """Drop the bookkeeping keys the optimize/summary passes leave behind."""
    nodes = structure if isinstance(structure, list) else [structure]
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node.pop('_same_page', None)
        if node.get('nodes'):
            strip_internal_keys(node['nodes'])
    return structure


async def summarize_tree(structure, pdf_pages, model=None,
                         small_node_tokens=SUMMARY_RAW_TEXT_TOKENS,
                         max_intro_pages=SUMMARY_INTRO_MAX_PAGES, concurrency=None):
    """Bottom-up summaries: leaves from their own pages, parents composed from
    child summaries plus the pages no child covers. A parent's summary describes
    its whole subtree (end_index union semantics). Nodes that already carry a
    summary are left untouched; leaves under `small_node_tokens` use their raw
    text as the summary without a model call. Queued model calls are admitted
    deepest node first: depth counts the calls left on a node's path to the
    root, its own included."""
    gate = _PriorityGate(concurrency or SUMMARY_CONCURRENCY)
    asked = answered = False

    async def ask(prompt, prio):
        nonlocal asked, answered
        asked = True
        async with gate.slot(prio):
            reply = await llm_acompletion(model, prompt)
        if reply:
            answered = True
        return reply

    async def leaf_summary(node, prio):
        text = get_text_of_pdf_pages(pdf_pages, node['start_index'], node['end_index'])
        if count_tokens(text, model="gpt-4o") < small_node_tokens:
            return text.strip()

        # A node merged from same-page siblings carries a title joined from theirs.
        # This call already has the page text in front of it, so the better title
        # costs no extra call; every other node keeps the heading the document
        # printed, and its prompt stays byte-identical to the one without this.
        retitle = bool(node.get('_same_page'))
        titles = "; ".join(node.get('key_items') or [])
        ask_title = (f"\n    The text is one page holding several short sections: {titles}. "
                     f"Also return a short title, at most 12 words, naming what the "
                     f"whole page covers." if retitle else "")
        title_field = ('\n        "title": <a short title naming what the whole page covers>,'
                       if retitle else "")

        prompt = f"""You are given a text chunk from a document.
    Your task is to generate a concise description of everything that is covered in the text, summarizing all its points without omitting any type of content.
    Keep the description concise and to the point, avoiding unnecessary details.{ask_title}

    Given Text: {text}

    Reply strictly in the following JSON format:
    {{{title_field}
        "points": <a list of points covered in the text>,
        "summary": <a concise description of everything that is covered in the text, summarizing all its points without omitting any type of content>
    }}

    Follow strictly the above JSON return format. Do not include any other text!
    """
        reply = await ask(prompt, prio)
        if retitle:
            written = parse_title(reply)
            if written:
                node['title'] = written
        return parse_summary(reply)

    async def parent_summary(node, prio):
        children = node['nodes']
        intro = get_intro_text(node, pdf_pages, max_pages=max_intro_pages)
        listing = json.dumps(
            [{'title': c.get('title', ''), 'summary': c.get('summary', '')} for c in children],
            ensure_ascii=False)
        prompt = f"""You are given a section of a document: the text that opens the section (possibly empty) and the titles and summaries of its subsections.
    Your task is to generate a concise description of everything that is covered in the whole section, summarizing all its points without omitting any type of content.
    Keep the description concise and to the point, avoiding unnecessary details.

    Section Title: {node.get('title', '')}

    Opening Text: {intro}

    Subsection Titles and Summaries: {listing}

    Reply strictly in the following JSON format:
    {{
        "points": <a list of points covered in the section>,
        "summary": <a concise description of everything that is covered in the section, summarizing all its points without omitting any type of content>
    }}

    Follow strictly the above JSON return format. Do not include any other text!
    """
        return parse_summary(await ask(prompt, prio))

    async def visit(node, depth=1):
        children = node.get('nodes') or []
        if children:
            done = await asyncio.gather(*(visit(child, depth + 1) for child in children),
                                        return_exceptions=True)
            for result in done:
                if isinstance(result, Exception) and _is_unrecoverable(result):
                    raise result
        if node.get('summary'):
            return
        try:
            node['summary'] = await (parent_summary(node, depth) if children
                                     else leaf_summary(node, depth))
        except Exception as e:
            node['summary'] = ""
            if _is_unrecoverable(e):
                raise

    results = await asyncio.gather(*(visit(root) for root in structure),
                                    return_exceptions=True)
    for r in results:
        if isinstance(r, Exception) and _is_unrecoverable(r):
            raise r

    # Raw-text leaves summarize without the model, so they cannot vouch for
    # it: a run whose every model call failed still fails loud.
    def _any_summary(nodes):
        return any(n.get('summary') or _any_summary(n.get('nodes') or [])
                   for n in nodes)
    if (asked and not answered) or not _any_summary(structure):
        raise RuntimeError(
            "Summary generation failed for all nodes "
            "(every summary call failed or returned empty; "
            "check the model and its context limits)"
        )

    strip_internal_keys(structure)
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


def generate_doc_description(structure, model=None):
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.
        
    Document Structure: {structure}
    
    Directly return the description, do not include any other text.
    """
    try:
        return llm_completion(model, prompt)
    except Exception as e:
        # Per-prompt 400: the unbounded whole-tree prompt overran the
        # context; the indexed document survives with no description.
        if getattr(e, "status_code", None) == 400:
            return ""
        raise


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


def page_level_thinning(structure, thinning_threshold_node_num=20, min_pages_for_large_tree=3):
    """Legacy; superseded by tree_optimize.merge_tree."""
    def count_nodes(nodes):
        total = 0
        for node in nodes:
            total += 1
            if node.get('nodes'):
                total += count_nodes(node['nodes'])
        return total

    def get_subtree_end(node):
        while node.get('nodes'):
            node = node['nodes'][-1]
        return node.get('end_index', 0)

    def thin(nodes, total_nodes):
        for node in nodes:
            children = node.get('nodes')
            if not children:
                continue
            end_index = get_subtree_end(node)
            page_count = end_index - node.get('start_index', 0) + 1
            if page_count == 1 or (total_nodes > thinning_threshold_node_num and page_count < min_pages_for_large_tree):
                node['end_index'] = end_index
                node.pop('nodes', None)
            else:
                thin(children, total_nodes)

    nodes = structure if isinstance(structure, list) else [structure]
    total = count_nodes(nodes)
    thin(nodes, total)
    return structure


DEFAULT_INDEX_MODEL = "gpt-5.6-luna"
DEFAULT_CHAT_MODEL = "gpt-5.6-sol"

# Each of the five names has shipped in a release; all stay accepted.
_MODEL_KEYS = ("model", "summary_model", "retrieve_model",
               "index_model", "chat_model")


def _resolve_models(merged: dict) -> None:
    """Fill the model roles from whichever names were given: new names win
    over old, specific over general, ``model`` sets every role, and the
    built-in defaults close each chain. Idempotent, so already-resolved
    config objects can round-trip through load()."""
    given = {key: merged.get(key) for key in _MODEL_KEYS}
    index = given["index_model"] or given["model"] or DEFAULT_INDEX_MODEL
    summary = (given["summary_model"] or given["index_model"]
               or given["model"] or DEFAULT_INDEX_MODEL)
    chat = (given["chat_model"] or given["retrieve_model"]
            or given["model"] or DEFAULT_CHAT_MODEL)
    merged.update(model=index, index_model=index, summary_model=summary,
                  chat_model=chat, retrieve_model=chat)


class ConfigLoader:
    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        unknown_keys = (set(user_dict) - set(self._default_dict)
                        - set(_MODEL_KEYS))
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """
        Load the configuration, merging user options with default values.
        """
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        _resolve_models(merged)
        return config(**merged)

def create_node_mapping(tree, include_page_ranges=False, max_page=None):
    """Map node_id to node; with include_page_ranges, to {"node", "start_index",
    "end_index"} (end = next node's page_index, or max_page for the last node)."""
    def get_all_nodes(tree):
        if isinstance(tree, dict):
            return [tree] + [node for child in tree.get('nodes', []) for node in get_all_nodes(child)]
        elif isinstance(tree, list):
            return [node for item in tree for node in get_all_nodes(item)]
        return []

    all_nodes = get_all_nodes(tree)
    if not include_page_ranges:
        return {node["node_id"]: node for node in all_nodes if node.get("node_id")}
    mapping = {}
    for i, node in enumerate(all_nodes):
        if node.get("node_id"):
            end_page = all_nodes[i + 1].get("page_index") if i + 1 < len(all_nodes) else max_page
            mapping[node["node_id"]] = {
                "node": node,
                "start_index": node["page_index"],
                "end_index": end_page,
            }
    return mapping

def print_tree(tree, exclude_fields=None, indent=0):
    """Outline view; passing exclude_fields gives the 0.2.8 pprint view."""
    if exclude_fields is not None:
        from pprint import pprint
        pprint(remove_fields(tree, exclude_fields, max_len=40), sort_dicts=False, width=100)
        return
    for node in tree:
        summary = node.get('summary') or node.get('prefix_summary', '')
        summary_str = f"  —  {summary[:60]}..." if summary else ""
        print('  ' * indent + f"[{node.get('node_id', '?')}] {node.get('title', '')}{summary_str}")
        if node.get('nodes'):
            print_tree(node['nodes'], indent=indent + 1)

def print_wrapped(text, width=100):
    for line in text.splitlines():
        print(textwrap.fill(line, width=width))

