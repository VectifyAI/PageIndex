import asyncio
import os
import pytest

from pageindex.page_index_txt import (
    extract_nodes_from_txt,
    extract_node_text_content,
    txt_to_tree,
)


def test_extract_nodes_chapter_and_numbered_headings():
    content = """
Chapter 1: Getting Started
Here is the opening text for chapter 1.

1.1 Installation
Follow these steps to install the package.

1.2 Configuration
Set your environment variables.

Chapter 2: Advanced Usage
Deeper dive into the internals.
"""
    nodes, lines = extract_nodes_from_txt(content)
    assert len(nodes) == 4
    assert nodes[0]["node_title"] == "Chapter 1: Getting Started"
    assert nodes[0]["level"] == 1
    assert nodes[1]["node_title"] == "1.1 Installation"
    assert nodes[1]["level"] == 2
    assert nodes[2]["node_title"] == "1.2 Configuration"
    assert nodes[2]["level"] == 2
    assert nodes[3]["node_title"] == "Chapter 2: Advanced Usage"
    assert nodes[3]["level"] == 1


def test_extract_nodes_underlined_headings():
    content = """Main Title
==========
Some introductory text under the main title.

Sub Section
-----------
Some text under the subsection.
"""
    nodes, lines = extract_nodes_from_txt(content)
    assert len(nodes) == 2
    assert nodes[0]["node_title"] == "Main Title"
    assert nodes[0]["level"] == 1
    assert nodes[1]["node_title"] == "Sub Section"
    assert nodes[1]["level"] == 2


def test_extract_nodes_all_caps_headings():
    content = """
INTRODUCTION

This is the introduction section with some details.

METHODOLOGY

Here is the description of our methods and data.
"""
    nodes, lines = extract_nodes_from_txt(content)
    assert len(nodes) == 2
    assert nodes[0]["node_title"] == "INTRODUCTION"
    assert nodes[1]["node_title"] == "METHODOLOGY"


def test_extract_nodes_plain_paragraph_fallback():
    content = """First paragraph of a document without any explicit section headers.
It contains some useful information across several sentences.

Second paragraph providing additional context and conclusions.
All information is presented sequentially.
"""
    nodes, lines = extract_nodes_from_txt(content)
    assert len(nodes) == 2
    assert nodes[0]["node_title"].startswith("Section 1")
    assert nodes[1]["node_title"].startswith("Section 2")


def test_extract_node_text_content():
    content = """Chapter 1
Line 1
Line 2

Chapter 2
Line 3
Line 4"""
    nodes, lines = extract_nodes_from_txt(content)
    nodes_with_content = extract_node_text_content(nodes, lines)
    assert len(nodes_with_content) == 2
    assert "Line 1" in nodes_with_content[0]["text"]
    assert "Line 2" in nodes_with_content[0]["text"]
    assert "Line 3" in nodes_with_content[1]["text"]


def test_txt_to_tree_end_to_end(tmp_path):
    txt_file = tmp_path / "sample_doc.txt"
    txt_file.write_text("""Chapter 1: Overview
This is the overview chapter.

1.1 Background
Historical context and background information.

1.2 Objectives
The main objectives of this document.

Chapter 2: Implementation
Technical implementation details.
""", encoding="utf-8")

    tree_dict = asyncio.run(txt_to_tree(
        txt_path=str(txt_file),
        if_thinning=False,
        if_add_node_summary="no",
        if_add_node_text="yes",
        if_add_node_id="yes",
    ))

    assert tree_dict["doc_name"] == "sample_doc"
    assert tree_dict["line_count"] > 0
    structure = tree_dict["structure"]
    assert len(structure) == 2
    assert structure[0]["title"] == "Chapter 1: Overview"
    assert len(structure[0]["nodes"]) == 2
    assert structure[0]["nodes"][0]["title"] == "1.1 Background"
    assert structure[0]["nodes"][1]["title"] == "1.2 Objectives"
    assert structure[1]["title"] == "Chapter 2: Implementation"
