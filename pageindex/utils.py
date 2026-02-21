import tiktoken
import re
import openai
import logging
import os
from datetime import datetime
import time
import json
import PyPDF2
import copy
import asyncio
import pymupdf
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace as config
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption


CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")
BASE_URL =  os.getenv("BASE_URL","https://api.openai.com/v1/chat/completions")

class PDFReader:
    def __init__(self, pdf_path, parser="docling", use_gpu=False, num_threads=8, do_ocr=True, do_table_structure=True, do_cell_matching=True):
        self.pdf_path = pdf_path
        self.parser = parser
        self.options = {
            "use_gpu": use_gpu,
            "num_threads": num_threads,
            "do_ocr": do_ocr,
            "do_table_structure": do_table_structure,
            "do_cell_matching": do_cell_matching
        }
        self._doc_content = None # Generic content holder
        self._pages_list = None
        self._full_text = None

    def _load_docling(self):
        try:
            if self.options["use_gpu"]:
                print("Using GPU")
                device = AcceleratorDevice.GPU
            else:
                print("Using CPU")
                device = AcceleratorDevice.CPU            
            accelerator_options = AcceleratorOptions(
                num_threads=self.options["num_threads"],
                device=device
            )

            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = accelerator_options
            pipeline_options.do_ocr = self.options["do_ocr"]
            pipeline_options.do_table_structure = self.options["do_table_structure"]
            pipeline_options.table_structure_options = TableStructureOptions(
                do_cell_matching=self.options["do_cell_matching"]
            )

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )

            settings.debug.profile_pipeline_timings = True
            
            # Convert the document
            conversion_result = converter.convert(self.pdf_path)
            doc = conversion_result.document
            
            # Helper to get pages
            self._full_text = doc.export_to_markdown()
            self._pages_list = doc.export_to_markdown(page_break_placeholder="<!-- PAGE_BREAK -->").split("<!-- PAGE_BREAK -->")
            
            doc_conversion_secs = conversion_result.timings["pipeline_total"].times
            print(f"Conversion secs: {doc_conversion_secs}")

        except Exception as e:
            print(f"Error extracting text from PDF with Docling: {e}")
            raise e

    def _load_pypdf2(self):
        try:
            pdf_reader = PyPDF2.PdfReader(self.pdf_path)
            num_pages = len(pdf_reader.pages)
            page_texts = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    page_texts.append(page_text)
                else:
                    page_texts.append("")
            self._pages_list = page_texts
            self._full_text = "\n".join(page_texts)
        except Exception as e:
            print(f"Error extracting text from PDF with PyPDF2: {e}")
            raise e

    def _load_pymupdf(self):
        try:
            if isinstance(self.pdf_path, BytesIO):
                doc = pymupdf.open(stream=self.pdf_path, filetype="pdf")
            else:
                doc = pymupdf.open(self.pdf_path)
                
            num_pages = len(doc)
            page_texts = []
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    page_texts.append(page_text)
                else:
                    page_texts.append("")
            doc.close()
            self._pages_list = page_texts
            self._full_text = "\n".join(page_texts)
        except Exception as e:
            print(f"Error extracting text from PDF with PyMuPDF: {e}")
            raise e

    def load(self):
        if self._full_text is not None and self._pages_list is not None:
             return self
             
        if self.parser == "docling":
            print("Using Docling")
            self._load_docling()
        elif self.parser == "PyMuPDF":
            print("Using PyMuPDF")
            self._load_pymupdf()
        elif self.parser == "PyPDF2":
            print("Using PyPDF2")
            self._load_pypdf2()
        else:
            raise ValueError(f"Unsupported PDF parser: {self.parser}. Use 'docling', 'PyMuPDF', or 'PyPDF2'")
        return self

    def export_to_markdown(self):
        self.load()
        return self._full_text

    def get_pages(self):
        self.load()
        return self._pages_list


def extract_text_from_pdf(input_doc_path, pdf_parser="docling", use_gpu=False, num_threads=8, do_ocr=True, do_table_structure=True, do_cell_matching=True):
    reader = PDFReader(
        input_doc_path, 
        parser=pdf_parser, 
        use_gpu=use_gpu, 
        num_threads=num_threads, 
        do_ocr=do_ocr, 
        do_table_structure=do_table_structure, 
        do_cell_matching=do_cell_matching
    )
    return reader.export_to_markdown()

def read_pdf(input_doc_path, pdf_parser="docling", output_format="full", use_gpu=False, num_threads=8, do_ocr=True, do_table_structure=True, do_cell_matching=True):
    reader = PDFReader(
        input_doc_path, 
        parser=pdf_parser, 
        use_gpu=use_gpu, 
        num_threads=num_threads, 
        do_ocr=do_ocr, 
        do_table_structure=do_table_structure, 
        do_cell_matching=do_cell_matching
    )
    
    if output_format == "pages":
        return reader.get_pages()
    else:
        return reader.export_to_markdown()

def count_tokens(text, tokenizer_model="o200k_base"):
    if not text:
        return 0
    enc = tiktoken.get_encoding(tokenizer_model)
    tokens = enc.encode(text)
    return len(tokens)

def ChatGPT_API_with_finish_reason(model, prompt, response_format=None, api_key=CHATGPT_API_KEY, base_url=BASE_URL, chat_history=None):
    max_retries = 10
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]
            
            if response_format:
                # Use parse() for Pydantic models
                response = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    temperature=0,
                    response_format=response_format,
                )
                if response.choices[0].finish_reason == "length":
                    finish_reason = "max_output_reached"
                else:
                    finish_reason = "finished"
                return response.choices[0].message.parsed, finish_reason
            else:
                # Use create() for regular text responses
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                )
                if response.choices[0].finish_reason == "length":
                    finish_reason = "max_output_reached"
                else:
                    finish_reason = "finished"
                return response.choices[0].message.content, finish_reason

        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)  # Wait for 1秒 before retrying
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                return "Error", "error"



def ChatGPT_API(model, prompt, response_format=None, api_key=CHATGPT_API_KEY, base_url=BASE_URL, chat_history=None):
    max_retries = 10
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]
            
            if response_format:
                # Use parse() for Pydantic models
                response = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    temperature=0,
                    response_format=response_format,
                )
                return response.choices[0].message.parsed
            else:
                # Use create() for regular text responses
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                )
                return response.choices[0].message.content
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)  # Wait for 1秒 before retrying
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                return "Error"
            

async def ChatGPT_API_async(model, prompt, response_format=None, api_key=CHATGPT_API_KEY, base_url=BASE_URL):
    max_retries = 10
    messages = [{"role": "user", "content": prompt}]
    for i in range(max_retries):
        try:
            async with openai.AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
                if response_format:
                    # Use parse() for Pydantic models
                    response = await client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        temperature=0,
                        response_format=response_format,
                    )
                    return response.choices[0].message.parsed
                else:
                    # Use create() for regular text responses
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0,
                    )
                    return response.choices[0].message.content
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)  # Wait for 1s before retrying
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                return "Error"  
            
            
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
        except:
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


def get_pdf_title(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title

def get_text_of_pages(pdf_path, start_page, end_page, tag=True, pdf_parser="PyPDF2", use_gpu=False, num_threads=8, do_ocr=True, do_table_structure=True):
    reader = PDFReader(
        pdf_path, 
        parser=pdf_parser, 
        use_gpu=use_gpu, 
        num_threads=num_threads, 
        do_ocr=do_ocr, 
        do_table_structure=do_table_structure
    )
    pages = reader.get_pages()
    
    text = ""
    # Adjust for 0-based indexing and handle page limits
    num_pages = len(pages)
    
    # Ensure start_page is at least 1
    start_idx = max(0, start_page - 1)
    # Ensure end_page does not exceed number of pages
    end_idx = min(num_pages, end_page)
    
    for i in range(start_idx, end_idx):
        page_text = pages[i]
        if tag:
            text += f"<start_index_{i+1}>\n{page_text}\n<end_index_{i+1}>\n"
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



def get_page_tokens(pdf_path, tokenizer_model="o200k_base", pdf_parser="docling", use_gpu=False, do_ocr=True, do_table_structure=True, do_cell_matching=True):
    enc = tiktoken.get_encoding(tokenizer_model)
    
    reader = PDFReader(
        pdf_path, 
        parser=pdf_parser, 
        use_gpu=use_gpu, 
        do_ocr=do_ocr, 
        do_table_structure=do_table_structure,
        do_cell_matching=do_cell_matching
    )
    pages = reader.get_pages()
    
    page_list = []
    for page_text in pages:
        token_length = len(enc.encode(page_text))
        page_list.append((page_text, token_length))
    return page_list

        

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

def remove_fields(data, fields=['text']):
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
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
        num_tokens = count_tokens(node['text'], model='gpt-4o')
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
                    try:
                        if data[i]['physical_index'].startswith('<physical_index_'):
                            data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].rstrip('>').strip())
                        elif data[i]['physical_index'].startswith('physical_index_'):
                            data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].strip())
                    except ValueError:
                        # Keep original value if conversion fails (e.g., Roman numerals like 'ii')
                        pass
    elif isinstance(data, str):
        try:
            if data.startswith('<physical_index_'):
                data = int(data.split('_')[-1].rstrip('>').strip())
            elif data.startswith('physical_index_'):
                data = int(data.split('_')[-1].strip())
        except ValueError:
            # Return None if conversion fails (e.g., Roman numerals like 'ii')
            return None
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
    response = await ChatGPT_API_async(model, prompt)
    return response


async def generate_summaries_for_structure(structure, model=None):
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
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
    response = ChatGPT_API(model, prompt)
    return response


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
        return config(**merged)