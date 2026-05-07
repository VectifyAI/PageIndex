import asyncio

from pageindex import page_index_md as md


def test_extract_nodes_from_markdown_ignores_code_blocks_and_blanks():
    content = "# Title\n\n```python\n# Not Header\n```\n## Child\ntext"

    nodes, lines = md.extract_nodes_from_markdown(content)

    assert nodes == [
        {"node_title": "Title", "line_num": 1},
        {"node_title": "Child", "line_num": 6},
    ]
    assert lines[2] == "```python"


def test_extract_node_text_content_warns_on_bad_line(capsys):
    nodes = [{"node_title": "Bad", "line_num": 1}, {"node_title": "Good", "line_num": 2}]
    result = md.extract_node_text_content(nodes, ["plain text", "## Good", "body"])

    assert result == [{"title": "Good", "line_num": 2, "level": 2, "text": "## Good\nbody"}]
    assert "does not contain a valid header" in capsys.readouterr().out


def test_token_count_update_and_tree_thinning(monkeypatch):
    monkeypatch.setattr(md, "count_tokens", lambda text, model=None: len(text.split()))
    nodes = [
        {"title": "A", "level": 1, "text": "one", "line_num": 1},
        {"title": "B", "level": 2, "text": "two words", "line_num": 2},
        {"title": "C", "level": 1, "text": "three", "line_num": 3},
    ]

    counted = md.update_node_list_with_text_token_count(nodes)
    assert counted[0]["text_token_count"] == 3

    thinned = md.tree_thinning_for_index(counted, min_node_token=5)
    assert [node["title"] for node in thinned] == ["A", "C"]
    assert "two words" in thinned[0]["text"]

    chain = [
        {"title": "A", "level": 1, "text": "a", "line_num": 1, "text_token_count": 1},
        {"title": "B", "level": 2, "text": "b", "line_num": 2, "text_token_count": 1},
        {"title": "C", "level": 3, "text": "c", "line_num": 3, "text_token_count": 1},
    ]
    assert [node["title"] for node in md.tree_thinning_for_index(chain, min_node_token=10)] == ["A"]

    no_children = [{"title": "A", "level": 1, "text": "", "line_num": 1, "text_token_count": 1}]
    assert md.tree_thinning_for_index(no_children, min_node_token=10)[0]["text"] == ""


def test_build_tree_and_clean_output():
    tree = md.build_tree_from_nodes(
        [
            {"title": "A", "level": 1, "text": "a", "line_num": 1},
            {"title": "B", "level": 2, "text": "b", "line_num": 2},
            {"title": "C", "level": 1, "text": "c", "line_num": 3},
        ]
    )

    assert tree[0]["nodes"][0]["title"] == "B"
    assert tree[1]["title"] == "C"
    assert md.clean_tree_for_output(tree)[0]["nodes"][0]["text"] == "b"
    assert md.build_tree_from_nodes([]) == []


def test_get_node_summary_short_and_long(monkeypatch):
    async def fake_summary(node, model=None):
        return f"summary:{node['text']}"

    monkeypatch.setattr(md, "count_tokens", lambda text, model=None: len(text.split()))
    monkeypatch.setattr(md, "generate_node_summary", fake_summary)

    assert asyncio.run(md.get_node_summary({"text": "few words"}, summary_token_threshold=3)) == "few words"
    assert asyncio.run(md.get_node_summary({"text": "many many words"}, summary_token_threshold=3)) == "summary:many many words"


def test_generate_summaries_marks_leaf_summary_and_parent_prefix(monkeypatch):
    async def fake_node_summary(node, summary_token_threshold=200, model=None):
        return f"s:{node['title']}"

    monkeypatch.setattr(md, "get_node_summary", fake_node_summary)
    structure = [{"title": "A", "nodes": [{"title": "B", "nodes": []}]}]

    result = asyncio.run(md.generate_summaries_for_structure_md(structure, 10))

    assert result[0]["prefix_summary"] == "s:A"
    assert result[0]["nodes"][0]["summary"] == "s:B"


def test_md_to_tree_modes(monkeypatch, tmp_path):
    path = tmp_path / "sample.md"
    path.write_text("# Title\nbody\n## Child\nmore\n", encoding="utf-8")
    monkeypatch.setattr(md, "count_tokens", lambda text, model=None: len(text.split()))

    async def fake_summary(structure, summary_token_threshold, model=None):
        for node in md.structure_to_list(structure):
            node["summary"] = f"sum {node['title']}"
        return structure

    monkeypatch.setattr(md, "generate_summaries_for_structure_md", fake_summary)
    monkeypatch.setattr(md, "generate_doc_description", lambda structure, model=None: "description")

    no_text = asyncio.run(md.md_to_tree(str(path), if_add_node_summary="no", if_add_node_text="no"))
    assert "text" not in no_text["structure"][0]

    with_text = asyncio.run(md.md_to_tree(str(path), if_add_node_summary="no", if_add_node_text="yes"))
    assert with_text["structure"][0]["text"].startswith("# Title")

    summarized = asyncio.run(
        md.md_to_tree(
            str(path),
            if_thinning=True,
            min_token_threshold=100,
            if_add_node_summary="yes",
            summary_token_threshold=1,
            if_add_doc_description="yes",
            if_add_node_text="no",
        )
    )
    assert summarized["doc_description"] == "description"
    assert "text" not in summarized["structure"][0]

    summary_with_text = asyncio.run(
        md.md_to_tree(
            str(path),
            if_add_node_summary="yes",
            summary_token_threshold=1,
            if_add_doc_description="no",
            if_add_node_text="yes",
            if_add_node_id="no",
        )
    )
    assert summary_with_text["structure"][0]["summary"].startswith("sum")
    assert summary_with_text["structure"][0]["node_id"] == "0001"
