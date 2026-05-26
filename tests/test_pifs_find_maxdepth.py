import json
from pathlib import Path

import pytest


def _register_find_fixture(tmp_path: Path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    filesystem.metadata.register_schema({"fields": {"department": "string"}})

    def add_file(
        filename: str,
        *,
        folder_path: str,
        external_id: str,
        title: str,
        domain: str,
    ) -> None:
        source = source_dir / filename
        source.write_text(f"{title} fixture text", encoding="utf-8")
        filesystem.register_file(
            storage_uri=source.as_uri(),
            source_path=f"docs/{filename}",
            folder_path=folder_path,
            external_id=external_id,
            title=title,
            content=source.read_text(encoding="utf-8"),
            metadata={"department": domain},
        )

    add_file(
        "root.txt",
        folder_path="/documents",
        external_id="doc_root",
        title="Root document",
        domain="ops",
    )
    add_file(
        "child.txt",
        folder_path="/documents/team",
        external_id="doc_child",
        title="Child document",
        domain="ops",
    )
    add_file(
        "deep.txt",
        folder_path="/documents/team/deep",
        external_id="doc_deep",
        title="Deep document",
        domain="ops",
    )
    add_file(
        "other.txt",
        folder_path="/documents/team",
        external_id="doc_other",
        title="Other document",
        domain="finance",
    )
    return PIFSCommandExecutor(filesystem, json_output=True)


def _data(output: str):
    return json.loads(output)["data"]


def test_find_maxdepth_one_returns_direct_files_only(tmp_path):
    executor = _register_find_fixture(tmp_path)

    rows = _data(executor.execute("find /documents -maxdepth 1 -type f"))

    assert [row["external_id"] for row in rows] == ["doc_root"]


def test_find_output_is_path_first_without_session_refs(tmp_path):
    executor = _register_find_fixture(tmp_path)
    executor.json_output = False

    output = executor.execute("find /documents -maxdepth 1 -type f")

    assert output.startswith("/documents/Root document id=doc_root file_ref=file_")
    assert "ref_1" not in output
    assert "title=Root document" in output


def test_stable_path_targets_work_without_session_refs(tmp_path):
    executor = _register_find_fixture(tmp_path)
    executor.json_output = False

    stat = executor.execute("stat '/documents/Root document'")
    text = executor.execute("cat '/documents/Root document' --all")

    assert "target: /documents/Root document" in stat
    assert "document_id: doc_root" in stat
    assert "Root document fixture text" in text


def test_find_maxdepth_zero_type_directory_returns_start_folder(tmp_path):
    executor = _register_find_fixture(tmp_path)

    rows = _data(executor.execute("find /documents -maxdepth 0 -type d"))

    assert [row["path"] for row in rows] == ["/documents"]


def test_find_directory_output_renders_root_without_double_slash(tmp_path):
    executor = _register_find_fixture(tmp_path)
    executor.json_output = False

    output = executor.execute("find / -maxdepth 1 -type d")

    assert output.splitlines()[0] == "/ folders=1 files=0"
    assert "//" not in output
    assert "/documents/ folders=1 files=1" in output


def test_find_maxdepth_combines_with_where_and_limit(tmp_path):
    executor = _register_find_fixture(tmp_path)

    rows = _data(
        executor.execute(
            """find /documents -maxdepth 2 -type f --where '{"department":"ops"}' --limit 1"""
        )
    )

    assert len(rows) == 1
    assert rows[0]["metadata"]["department"] == "ops"
    assert rows[0]["folder_path"] in {"/documents", "/documents/team"}


def test_find_maxdepth_rejects_invalid_values_and_unsupported_options(tmp_path):
    from pageindex.filesystem.commands import PIFSCommandError

    executor = _register_find_fixture(tmp_path)

    with pytest.raises(PIFSCommandError, match="find -maxdepth requires an integer >= 0"):
        executor.execute("find /documents -maxdepth nope -type f")
    with pytest.raises(PIFSCommandError, match="find -maxdepth requires an integer >= 0"):
        executor.execute("find /documents -maxdepth -1 -type f")
    with pytest.raises(PIFSCommandError, match="Unsupported find option: -exec"):
        executor.execute("find /documents -maxdepth 1 -type f -exec")


def test_find_maxdepth_is_advertised_to_agents(tmp_path):
    executor = _register_find_fixture(tmp_path)

    assert "-maxdepth N -type f|d" in executor.describe_available_command_surfaces()
    assert executor.command_capabilities()["retrieval"]["lexical"]["find_maxdepth"] is True


def test_where_path_error_points_to_folder_scope(tmp_path):
    from pageindex.filesystem.commands import PIFSCommandError

    executor = _register_find_fixture(tmp_path)

    with pytest.raises(PIFSCommandError) as exc_info:
        executor.execute("""find --where '{"path":"/documents"}'""")

    message = str(exc_info.value)
    assert "Folder paths are positional PIFS paths" in message
    assert "find /documents -type f" in message
    assert "stat --schema" in message
