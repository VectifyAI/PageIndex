import pytest


def test_root_virtual_file_path_resolves_without_double_slash(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    file_ref = filesystem.register_file(
        storage_uri="file:///tmp/root-source.txt",
        source_path="sources/root-source.txt",
        folder_path="/",
        external_id="doc_root_title",
        title="Root Title",
        content="root content",
    )

    assert filesystem.store.resolve_file_ref("/Root Title") == file_ref


def test_ambiguous_virtual_file_path_raises_clear_error(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    first_ref = filesystem.register_file(
        storage_uri="file:///tmp/first.txt",
        source_path="b/file.txt",
        folder_path="/a",
        external_id="doc_first",
        title="First",
        content="first content",
    )
    second_ref = filesystem.register_file(
        storage_uri="file:///tmp/second.txt",
        source_path="second-source.txt",
        folder_path="/a/b",
        external_id="doc_second",
        title="file.txt",
        content="second content",
    )

    with pytest.raises(KeyError, match="Ambiguous file target"):
        filesystem.store.resolve_file_ref("/a/b/file.txt")

    assert first_ref != second_ref
