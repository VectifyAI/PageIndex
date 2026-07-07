import pytest
from pathlib import Path
from pageindex.parser.markdown import MarkdownParser
from pageindex.parser.protocol import ContentNode, ParsedDocument

@pytest.fixture
def sample_md(tmp_path):
    md = tmp_path / "test.md"
    md.write_text("""# Chapter 1
Some intro text.

## Section 1.1
Details here.

## Section 1.2
More details.

# Chapter 2
Another chapter.
""")
    return str(md)

def test_supported_extensions():
    parser = MarkdownParser()
    exts = parser.supported_extensions()
    assert ".md" in exts
    assert ".markdown" in exts

def test_parse_returns_parsed_document(sample_md):
    parser = MarkdownParser()
    result = parser.parse(sample_md)
    assert isinstance(result, ParsedDocument)
    assert result.doc_name == "test"

def test_parse_nodes_have_level(sample_md):
    parser = MarkdownParser()
    result = parser.parse(sample_md)
    assert len(result.nodes) == 4
    assert result.nodes[0].level == 1
    assert result.nodes[0].title == "Chapter 1"
    assert result.nodes[1].level == 2
    assert result.nodes[1].title == "Section 1.1"
    assert result.nodes[3].level == 1

def test_parse_nodes_have_content(sample_md):
    parser = MarkdownParser()
    result = parser.parse(sample_md)
    assert "Some intro text" in result.nodes[0].content
    assert "Details here" in result.nodes[1].content

def test_parse_nodes_have_index(sample_md):
    parser = MarkdownParser()
    result = parser.parse(sample_md)
    for node in result.nodes:
        assert node.index is not None


def test_preamble_before_first_header_is_kept(tmp_path):
    md = tmp_path / "pre.md"
    md.write_text("Abstract: important preamble text.\n\n# Chapter 1\nBody.\n")
    result = MarkdownParser().parse(str(md))
    assert result.nodes[0].title == "pre"
    assert "important preamble text" in result.nodes[0].content
    assert result.nodes[1].title == "Chapter 1"


def test_headerless_file_yields_single_node(tmp_path):
    md = tmp_path / "plain.md"
    md.write_text("Just some text.\nNo headings at all.\n")
    result = MarkdownParser().parse(str(md))
    assert len(result.nodes) == 1
    assert result.nodes[0].title == "plain"
    assert "No headings at all" in result.nodes[0].content
