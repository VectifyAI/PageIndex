import asyncio
import copy
import json
import logging
import os
import re
import textwrap
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as config

import litellm
import pymupdf
import pypdf
import yaml
from dotenv import load_dotenv

load_dotenv()

# Backward compatibility: support CHATGPT_API_KEY as alias for OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")

litellm.drop_params = True


def count_tokens(text, model=None):
    if not text:
        return 0
    return litellm.token_counter(model=model, text=text)


def _build_messages(prompt, chat_history=None):
    base = [{"role": "user", "content": prompt}]
    return list(chat_history) + base if chat_history else base


def _extract_completion_result(response, return_finish_reason):
    content = response.choices[0].message.content
    if return_finish_reason:
        finish_reason = (
            "max_output_reached"
            if response.choices[0].finish_reason == "length"
            else "finished"
        )
        return content, finish_reason
    return content


def llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
    if model:
        model = model.removeprefix("litellm/")
    messages = _build_messages(prompt, chat_history)
    max_retries = 10
    for i in range(max_retries):
        try:
            response = litellm.completion(model=model, messages=messages, temperature=0)
            return _extract_completion_result(response, return_finish_reason)
        except Exception as e:
            print("************* Retrying *************")
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error("Max retries reached for prompt: " + prompt)
                return ("", "error") if return_finish_reason else ""


async def llm_acompletion(model, prompt):
    if model:
        model = model.removeprefix("litellm/")
    max_retries = 10
    messages = [{"role": "user", "content": prompt}]
    for i in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print("************* Retrying *************")
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error("Max retries reached for prompt: " + prompt)
                return ""


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
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            json_content = content.strip()

        json_content = json_content.replace("None", "null")
        json_content = json_content.replace("\n", " ").replace("\r", " ")
        json_content = " ".join(json_content.split())

        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to extract JSON: {e}")
        try:
            json_content = json_content.replace(",]", "]").replace(",}", "}")
            return json.loads(json_content)
        except Exception:
            logging.error("Failed to parse JSON even after cleanup")
            return {}
    except Exception as e:
        logging.error(f"Unexpected error while extracting JSON: {e}")
        return {}


def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data["node_id"] = str(node_id).zfill(4)
        node_id += 1
        for key in data.keys():
            if "nodes" in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id


def get_nodes(structure):
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop("nodes", None)
        nodes = [structure_node]
        for key in structure.keys():
            if "nodes" in key:
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
        if "nodes" in structure:
            nodes.extend(structure_to_list(structure["nodes"]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes


def get_leaf_nodes(structure):
    if isinstance(structure, dict):
        if not structure["nodes"]:
            structure_node = copy.deepcopy(structure)
            structure_node.pop("nodes", None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in structure.keys():
                if "nodes" in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes


def _find_node_by_id(data, node_id):
    """Recursively search for a node with the given node_id."""
    if isinstance(data, dict):
        if data.get("node_id") == node_id:
            return data
        for key in data.keys():
            if "nodes" in key:
                result = _find_node_by_id(data[key], node_id)
                if result:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = _find_node_by_id(item, node_id)
            if result:
                return result
    return None


def is_leaf_node(data, node_id):
    node = _find_node_by_id(data, node_id)
    return bool(node and not node.get("nodes"))


def get_last_node(structure):
    return structure[-1]


def extract_text_from_pdf(pdf_path):
    pdf_reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


def get_pdf_title(pdf_path):
    pdf_reader = pypdf.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else "Untitled"
    return title


def get_text_of_pages(pdf_path, start_page, end_page, tag=True):
    pdf_reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page - 1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num + 1}>\n{page_text}\n<end_index_{page_num + 1}>\n"
        else:
            text += page_text
    return text


def get_first_start_page_from_text(text):
    start_page = -1
    start_page_match = re.search(r"<start_index_(\d+)>", text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page


def get_last_start_page_from_text(text):
    start_page = -1
    start_page_matches = re.finditer(r"<start_index_(\d+)>", text)
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename, replacement="-"):
    return filename.replace("/", replacement)


def get_pdf_name(pdf_path):
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = pypdf.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else "Untitled"
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


class JsonLogger:
    def __init__(self, file_path):
        pdf_name = get_pdf_name(file_path)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        self.log_data = []

    def log(self, _level, message, **kwargs):
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({"message": message})
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


def _clean_tree_node(node):
    """Remove empty 'nodes' arrays from a tree node recursively."""
    if not node["nodes"]:
        del node["nodes"]
    else:
        for child in node["nodes"]:
            _clean_tree_node(child)
    return node


def list_to_tree(data):
    def get_parent_structure(structure):
        if not structure:
            return None
        parts = str(structure).split(".")
        return ".".join(parts[:-1]) if len(parts) > 1 else None

    nodes = {}
    root_nodes = []

    for item in data:
        structure = item.get("structure")
        node = {
            "title": item.get("title"),
            "start_index": item.get("start_index"),
            "end_index": item.get("end_index"),
            "nodes": [],
        }
        nodes[structure] = node
        parent_structure = get_parent_structure(structure)
        if parent_structure:
            if parent_structure in nodes:
                nodes[parent_structure]["nodes"].append(node)
            else:
                root_nodes.append(node)
        else:
            root_nodes.append(node)

    return [_clean_tree_node(node) for node in root_nodes]


def add_preface_if_needed(data):
    if not isinstance(data, list) or not data:
        return data
    if data[0]["physical_index"] is not None and data[0]["physical_index"] > 1:
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data


def get_page_tokens(pdf_path, model=None, pdf_parser="PyPDF2"):
    if pdf_parser == "PyPDF2":
        pdf_reader = pypdf.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = litellm.token_counter(model=model, text=page_text)
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            doc = pymupdf.open(stream=pdf_path, filetype="pdf")
        elif (
            isinstance(pdf_path, str)
            and os.path.isfile(pdf_path)
            and pdf_path.lower().endswith(".pdf")
        ):
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
    for page_num in range(start_page - 1, end_page):
        text += pdf_pages[page_num][0]
    return text


def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page - 1, end_page):
        text += f"<physical_index_{page_num + 1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num + 1}>\n"
    return text


def get_number_of_pages(pdf_path):
    pdf_reader = pypdf.PdfReader(pdf_path)
    num = len(pdf_reader.pages)
    return num


def post_processing(structure, end_physical_index):
    for i, item in enumerate(structure):
        item["start_index"] = item.get("physical_index")
        if i < len(structure) - 1:
            if structure[i + 1].get("appear_start") == "yes":
                item["end_index"] = structure[i + 1]["physical_index"] - 1
            else:
                item["end_index"] = structure[i + 1]["physical_index"]
        else:
            item["end_index"] = end_physical_index
    tree = list_to_tree(structure)
    if len(tree) != 0:
        return tree
    else:
        for node in structure:
            node.pop("appear_start", None)
            node.pop("physical_index", None)
        return structure


def clean_structure_post(data):
    if isinstance(data, dict):
        data.pop("page_number", None)
        data.pop("start_index", None)
        data.pop("end_index", None)
        if "nodes" in data:
            clean_structure_post(data["nodes"])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data


def remove_fields(data, fields=None):
    if fields is None:
        fields = ["text"]
    if isinstance(data, dict):
        return {k: remove_fields(v, fields) for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data


def print_toc(tree, indent=0):
    for node in tree:
        print("  " * indent + node["title"])
        if node.get("nodes"):
            print_toc(node["nodes"], indent + 1)


def print_json(data, max_len=40, indent=2):
    def simplify_data(obj):
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + "..."
        else:
            return obj

    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def remove_structure_text(data):
    if isinstance(data, dict):
        data.pop("text", None)
        if "nodes" in data:
            remove_structure_text(data["nodes"])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure, limit=110000):
    nodes = structure_to_list(structure)
    for node in nodes:
        num_tokens = count_tokens(node["text"], model=None)
        if num_tokens > limit:
            print(f"Node ID: {node['node_id']} has {num_tokens} tokens")
            print("Start Index:", node["start_index"])
            print("End Index:", node["end_index"])
            print("Title:", node["title"])
            print("\n")


def _parse_physical_index_str(value):
    """Parse a physical_index string tag into an integer, or return None."""
    if value.startswith("<physical_index_"):
        return int(value.split("_")[-1].rstrip(">").strip())
    if value.startswith("physical_index_"):
        return int(value.split("_")[-1].strip())
    return None


def convert_physical_index_to_int(data):
    if isinstance(data, list):
        for i in range(len(data)):
            if isinstance(data[i], dict) and "physical_index" in data[i]:
                if isinstance(data[i]["physical_index"], str):
                    parsed = _parse_physical_index_str(data[i]["physical_index"])
                    if parsed is not None:
                        data[i]["physical_index"] = parsed
    elif isinstance(data, str):
        parsed = _parse_physical_index_str(data)
        return parsed if parsed is not None else (data if isinstance(data, int) else None)
    return data


def convert_page_to_int(data):
    for item in data:
        if "page" in item and isinstance(item["page"], str):
            try:
                item["page"] = int(item["page"])
            except ValueError:
                pass
    return data


def add_node_text(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get("start_index")
        end_page = node.get("end_index")
        node["text"] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if "nodes" in node:
            add_node_text(node["nodes"], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)


def add_node_text_with_labels(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get("start_index")
        end_page = node.get("end_index")
        node["text"] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if "nodes" in node:
            add_node_text_with_labels(node["nodes"], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)


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
    summaries = await asyncio.gather(*tasks)

    for node, summary in zip(nodes, summaries):
        node["summary"] = summary
    return structure


def create_clean_structure_for_description(structure):
    """
    Create a clean structure for document description generation,
    excluding unnecessary fields like 'text'.
    """
    if isinstance(structure, dict):
        clean_node = {}
        for key in ["title", "node_id", "summary", "prefix_summary"]:
            if key in structure:
                clean_node[key] = structure[key]
        if "nodes" in structure and structure["nodes"]:
            clean_node["nodes"] = create_clean_structure_for_description(structure["nodes"])
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
    response = llm_completion(model, prompt)
    return response


def reorder_dict(data, key_order):
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    if not order:
        return structure
    if isinstance(structure, dict):
        if "nodes" in structure:
            structure["nodes"] = format_structure(structure["nodes"], order)
        if not structure.get("nodes"):
            structure.pop("nodes", None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


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
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """Load configuration, merging user options with default values."""
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
        return config(**merged)


def create_node_mapping(tree):
    """Create a flat dict mapping node_id to node for quick lookup."""
    mapping = {}

    def _traverse(nodes):
        for node in nodes:
            if node.get("node_id"):
                mapping[node["node_id"]] = node
            if node.get("nodes"):
                _traverse(node["nodes"])

    _traverse(tree)
    return mapping


def print_tree(tree, indent=0):
    for node in tree:
        summary = node.get("summary") or node.get("prefix_summary", "")
        summary_str = f"  —  {summary[:60]}..." if summary else ""
        print(
            "  " * indent
            + f"[{node.get('node_id', '?')}] {node.get('title', '')}{summary_str}"
        )
        if node.get("nodes"):
            print_tree(node["nodes"], indent + 1)


def print_wrapped(text, width=100):
    for line in text.splitlines():
        print(textwrap.fill(line, width=width))
