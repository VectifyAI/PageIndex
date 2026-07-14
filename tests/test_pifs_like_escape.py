from pathlib import Path

from tests.pifs_markdown_fixture import register_markdown


def _register_file(
    filesystem,
    tmp_path: Path,
    filename: str,
    *,
    folder_path: str,
    external_id: str,
    metadata: dict[str, str] | None = None,
) -> None:
    register_markdown(
        filesystem,
        tmp_path,
        external_id,
        folder_path,
        title=f"{external_id}.md",
        text=f"{external_id} fixture text",
        metadata=metadata,
    )


def test_descendant_folder_filter_treats_underscore_literally(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    _register_file(
        filesystem,
        tmp_path,
        "literal.txt",
        folder_path="/proj_1/docs",
        external_id="literal_underscore",
    )
    _register_file(
        filesystem,
        tmp_path,
        "wildcard.txt",
        folder_path="/projA1/docs",
        external_id="wildcard_neighbor",
    )

    recursive = filesystem.store.list_folder("/proj_1", recursive=True, limit=10)
    scoped_results = filesystem.store.list_files(
        scope={"folder_path": "/proj_1", "recursive": True},
        limit=10,
    )
    ranked_folders = {folder["path"]: folder for folder in filesystem.store.find_folders("/", max_depth=1, limit=10)}

    assert {folder["path"] for folder in recursive["folders"]} == {"/proj_1/docs"}
    assert {file["external_id"] for file in recursive["files"]} == {"literal_underscore"}
    assert {result["external_id"] for result in scoped_results} == {"literal_underscore"}
    assert ranked_folders["/proj_1"]["matched_files"] == 1
    assert ranked_folders["/projA1"]["matched_files"] == 1
    assert filesystem.store.count_files_in_folder("/proj_1", recursive=True) == 1


def test_metadata_contains_treats_percent_and_underscore_literally(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    filesystem.metadata.register_schema({"fields": {"status": {}}})
    _register_file(
        filesystem,
        tmp_path,
        "percent.txt",
        folder_path="/documents",
        external_id="literal_percent",
        metadata={"status": "100% done"},
    )
    _register_file(
        filesystem,
        tmp_path,
        "percent-neighbor.txt",
        folder_path="/documents",
        external_id="percent_neighbor",
        metadata={"status": "100X done"},
    )
    _register_file(
        filesystem,
        tmp_path,
        "underscore.txt",
        folder_path="/documents",
        external_id="literal_underscore",
        metadata={"status": "build_alpha"},
    )
    _register_file(
        filesystem,
        tmp_path,
        "underscore-neighbor.txt",
        folder_path="/documents",
        external_id="underscore_neighbor",
        metadata={"status": "buildXalpha"},
    )

    percent_results = filesystem.store.list_files(
        metadata_filter={"status": {"$contains": "100% done"}},
        limit=10,
    )
    underscore_results = filesystem.store.list_files(
        metadata_filter={"status": {"$contains": "build_alpha"}},
        limit=10,
    )

    assert {result["external_id"] for result in percent_results} == {"literal_percent"}
    assert {result["external_id"] for result in underscore_results} == {"literal_underscore"}
