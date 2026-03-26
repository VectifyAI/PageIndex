import PyPDF2
import pymupdf
import re
import os
import tiktoken
from io import BytesIO
from typing import List, Tuple, Union, Optional
from .llm import count_tokens

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text from a PDF file using PyPDF2.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        str: Concatenated text from all pages.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def get_pdf_title(pdf_path: Union[str, BytesIO]) -> str:
    """
    Extract the title from PDF metadata.
    
    Args:
         pdf_path (Union[str, BytesIO]): Path to PDF or BytesIO object.
         
    Returns:
        str: Title of the PDF or 'Untitled'.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title

def get_text_of_pages(pdf_path: str, start_page: int, end_page: int, tag: bool = True) -> str:
    """
    Get text from a specific range of pages in a PDF.
    
    Args:
        pdf_path (str): Path to the PDF file.
        start_page (int): Start page number (1-based).
        end_page (int): End page number (1-based).
        tag (bool): If True, wraps page text in <start_index_N>... tags.
        
    Returns:
        str: Extracted text.
    """
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page-1, end_page):
        if page_num < len(pdf_reader.pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if tag:
                text += f"<start_index_{page_num+1}>\n{page_text}\n<end_index_{page_num+1}>\n"
            else:
                text += page_text
    return text

def get_first_start_page_from_text(text: str) -> int:
    """
    Extract the first page index tag found in text.
    
    Args:
        text (str): Text containing <start_index_N> tags.
        
    Returns:
        int: Page number or -1 if not found.
    """
    start_page = -1
    start_page_match = re.search(r'<start_index_(\d+)>', text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page

def get_last_start_page_from_text(text: str) -> int:
    """
    Extract the last page index tag found in text.
    
    Args:
        text (str): Text containing <start_index_N> tags.
        
    Returns:
        int: Page number or -1 if not found.
    """
    start_page = -1
    start_page_matches = re.finditer(r'<start_index_(\d+)>', text)
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename: str, replacement: str = '-') -> str:
    """Replace illegal characters in filename."""
    return filename.replace('/', replacement)

def get_pdf_name(pdf_path: Union[str, BytesIO]) -> str:
    """
    Get a sanitized name for the PDF file.
    
    Args:
        pdf_path (Union[str, BytesIO]): Path or file object.
        
    Returns:
        str: Filename or logical title.
    """
    pdf_name = "Untitled.pdf"
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        if meta and meta.title:
            pdf_name = meta.title
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


def get_page_tokens(
    pdf_path: Union[str, BytesIO], 
    model: str = "gpt-4o-2024-11-20", 
    pdf_parser: str = "PyPDF2"
) -> List[Tuple[str, int]]:
    """
    Extract text and token counts for each page.
    
    Args:
        pdf_path (Union[str, BytesIO]): Path to PDF.
        model (str): Model name for token counting.
        pdf_parser (str): "PyPDF2" or "PyMuPDF".
        
    Returns:
        List[Tuple[str, int]]: List of (page_text, token_count).
    """
    enc = tiktoken.encoding_for_model(model)
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            pdf_stream = pdf_path
            doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        else:
             raise ValueError(f"Invalid pdf path for PyMuPDF: {pdf_path}")
             
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")

        

def get_text_of_pdf_pages(pdf_pages: List[Tuple[str, int]], start_page: int, end_page: int) -> str:
    """
    Combine text from a list of page tuples [1-based range].
    
    Args:
        pdf_pages (List[Tuple[str, int]]): Output from get_page_tokens.
        start_page (int): Start page (1-based).
        end_page (int): End page (1-based, inclusive).
        
    Returns:
        str: Combined text.
    """
    text = ""
    # Safe indexing
    total_pages = len(pdf_pages)
    for page_num in range(start_page-1, end_page):
        if 0 <= page_num < total_pages:
            text += pdf_pages[page_num][0]
    return text

def get_text_of_pdf_pages_with_labels(pdf_pages: List[Tuple[str, int]], start_page: int, end_page: int) -> str:
    """
    Combine text from pages with <physical_index_N> tags.
    """
    text = ""
    total_pages = len(pdf_pages)
    for page_num in range(start_page-1, end_page):
        if 0 <= page_num < total_pages:
            text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
    return text

def get_number_of_pages(pdf_path: Union[str, BytesIO]) -> int:
    """Get total page count of a PDF."""
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    return len(pdf_reader.pages)
