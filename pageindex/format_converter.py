import os

def convert_docx_to_md(docx_path: str) -> str:
    """
    Reads a .docx file and converts it to markdown, preserving heading levels.
    Requires python-docx.
    """
    try:
        import docx
    except ImportError:
        raise ImportError("The 'python-docx' package is required to parse .docx files. Please install it.")

    doc = docx.Document(docx_path)
    md_lines = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        style_name = para.style.name if para.style else ""
        
        # Check if the style is a heading
        if style_name.startswith('Heading'):
            try:
                # e.g., "Heading 1" -> level 1
                level_str = style_name.replace('Heading', '').strip()
                level = int(level_str)
                # Cap the level at 6 for markdown
                level = min(level, 6)
                md_lines.append(f"{'#' * level} {text}\n")
            except ValueError:
                # Fallback if parsing fails
                md_lines.append(f"{text}\n")
        elif style_name == 'Title':
            md_lines.append(f"# {text}\n")
        else:
            md_lines.append(f"{text}\n")
            
    return "\n".join(md_lines)


def convert_html_to_md(html_path: str) -> str:
    """
    Reads an .html file and converts it to markdown using BeautifulSoup and markdownify.
    """
    try:
        from bs4 import BeautifulSoup
        import markdownify
    except ImportError:
        raise ImportError("The 'beautifulsoup4' and 'markdownify' packages are required to parse .html files.")

    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
        script.extract()

    # Convert to markdown with ATX headings (### instead of underlines)
    md_text = markdownify.markdownify(str(soup), heading_style="ATX")
    
    # Clean up excessive newlines
    import re
    md_text = re.sub(r'\n{3,}', '\n\n', md_text).strip()
    return md_text


def convert_txt_to_md(txt_path: str) -> str:
    """
    Reads a plain .txt file. It doesn't need actual conversion, but we wrap it here for consistency.
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


def convert_to_markdown(file_path: str) -> str:
    """
    Determines the file type and returns its markdown representation.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.docx':
        return convert_docx_to_md(file_path)
    elif ext in ['.html', '.htm']:
        return convert_html_to_md(file_path)
    elif ext == '.txt':
        return convert_txt_to_md(file_path)
    elif ext in ['.md', '.markdown']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format for conversion: {ext}")
