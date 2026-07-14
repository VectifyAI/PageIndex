import json
import hashlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest


class FakeOpenAI:
    def __init__(self, **_kwargs):
        self.embeddings = self

    def create(self, *, model, input, dimensions):
        assert model == "test-embedding"
        assert dimensions == 3

        def vector(text):
            lowered = str(text).lower()
            if "alpha" in lowered:
                return [1.0, 0.0, 0.0]
            if "beta" in lowered:
                return [0.0, 1.0, 0.0]
            return [0.0, 0.0, 1.0]

        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=vector(text))
                for index, text in enumerate(input)
            ]
        )


def install_network_fakes(monkeypatch):
    import openai
    from pageindex import PageIndexClient

    def fake_index(self, file_path, mode="auto"):
        path = Path(file_path)
        text = (
            path.read_text(encoding="utf-8")
            if path.suffix.lower() in {".md", ".markdown"}
            else "pdf alpha evidence"
        )
        path_digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
        document_id = f"pageindex_{path_digest[:16]}"
        document = {
            "id": document_id,
            "type": "pdf" if path.suffix.lower() == ".pdf" else "md",
            "path": str(path.resolve()),
            "doc_name": path.name,
            "doc_description": f"Summary for {path.name}: {text}",
            "line_count": len(text.splitlines()),
            "structure": [
                {
                    "title": path.stem,
                    "node_id": "0001",
                    "line_num": 1,
                    "text": text,
                    "nodes": [],
                }
            ],
            "pages": [{"page": 1, "content": text}],
        }
        self.documents[document_id] = document
        if self.workspace:
            self._save_doc(document_id)
        return document_id

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(PageIndexClient, "index", fake_index)


def write_embedding_config(tmp_path, monkeypatch):
    config = tmp_path / "pifs.json"
    config.write_text(
        json.dumps(
            {
                "embedding_base_url": "https://EXAMPLE.invalid/v1/",
                "embedding_model": "test-embedding",
                "embedding_dimensions": 3,
                "embedding_timeout": 12,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PIFS_CONFIG_FILE", str(config))
    monkeypatch.setenv("PIFS_EMBEDDING_API_KEY", "runtime-secret")
    return config


def logical_tables(connection):
    return {
        row[1]
        for row in connection.execute("PRAGMA table_list")
        if row[2] in {"table", "virtual"} and not row[1].startswith("sqlite_")
        and not row[1].startswith("semantic_index_vec_")
    }


def workspace_state(workspace):
    workspace = Path(workspace)
    if not workspace.exists():
        return {}
    return {
        path.relative_to(workspace).as_posix(): (
            f"symlink:{path.readlink()}"
            if path.is_symlink()
            else "directory"
            if path.is_dir()
            else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in sorted(workspace.rglob("*"))
    }


def create_projection_v2(workspace):
    from pageindex.filesystem.semantic_projection import (
        SummaryEmbeddingProfile,
        SummaryProjection,
    )

    projection_dir = Path(workspace) / "artifacts" / "projection_indexes"
    SummaryProjection(
        projection_dir,
        profile=SummaryEmbeddingProfile(
            base_url="https://example.invalid/v1",
            model="test-embedding",
            dimensions=3,
            api_key="runtime-only",
        ),
        create=True,
    )
    return projection_dir


def connect_summary_database(path):
    import sqlite_vec

    connection = sqlite3.connect(path)
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    return connection


def insert_projection_document(summary_path, *, file_ref="orphan_projection_ref"):
    import sqlite_vec

    with connect_summary_database(summary_path) as connection:
        connection.execute(
            """
            INSERT INTO semantic_index_docs(
                rowid, file_ref, external_id, source_type, title,
                text_hash, text_chars, metadata_json
            ) VALUES (1, ?, 'orphan-doc', 'markdown', 'orphan.md',
                      'orphan-hash', 14, '{}')
            """,
            (file_ref,),
        )
        connection.execute(
            "INSERT INTO semantic_index_vec(rowid, source_type, embedding) "
            "VALUES (1, 'markdown', ?)",
            (sqlite_vec.serialize_float32([1.0, 0.0, 0.0]),),
        )


def delete_projection_document(summary_path, file_ref):
    with connect_summary_database(summary_path) as connection:
        row = connection.execute(
            "SELECT rowid FROM semantic_index_docs WHERE file_ref = ?",
            (file_ref,),
        ).fetchone()
        assert row is not None
        connection.execute(
            "DELETE FROM semantic_index_vec WHERE rowid = ?",
            (row[0],),
        )
        connection.execute(
            "DELETE FROM semantic_index_docs WHERE rowid = ?",
            (row[0],),
        )


def catalog_file_count(workspace):
    with sqlite3.connect(Path(workspace) / "filesystem.sqlite") as connection:
        return connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]


def registration_logical_state(workspace):
    workspace = Path(workspace)
    catalog_path = workspace / "filesystem.sqlite"
    projection_dir = workspace / "artifacts" / "projection_indexes"
    summary_path = projection_dir / "summary.sqlite"
    cache_path = projection_dir / "embedding_cache.sqlite"

    with sqlite3.connect(catalog_path) as connection:
        catalog = {
            "files": connection.execute(
                """
                SELECT file_ref, external_id, storage_uri, title, descriptor,
                       content_type, source_type, fingerprint, text_artifact_path,
                       raw_artifact_path, pageindex_doc_id, pageindex_tree_status,
                       metadata_json, metadata_status_json, deleted_at
                FROM files
                ORDER BY file_ref
                """
            ).fetchall(),
            "folders": connection.execute(
                """
                SELECT folder_id, parent_id, name, path, description, kind, metadata_json
                FROM folders
                ORDER BY path
                """
            ).fetchall(),
            "metadata_fields": connection.execute(
                """
                SELECT field_id, name, description, source
                FROM metadata_fields
                ORDER BY name
                """
            ).fetchall(),
            "metadata_values": connection.execute(
                """
                SELECT file_ref, field_id, value_text
                FROM metadata_values
                ORDER BY file_ref, field_id, value_text
                """
            ).fetchall(),
        }

    summary = {"docs": [], "vec": []}
    if summary_path.is_file():
        with connect_summary_database(summary_path) as connection:
            summary = {
                "docs": connection.execute(
                    """
                    SELECT rowid, file_ref, external_id, source_type, title,
                           text_hash, text_chars, metadata_json
                    FROM semantic_index_docs
                    ORDER BY rowid
                    """
                ).fetchall(),
                "vec": connection.execute(
                    """
                    SELECT rowid, source_type, hex(embedding)
                    FROM semantic_index_vec
                    ORDER BY rowid
                    """
                ).fetchall(),
            }

    cache = []
    if cache_path.is_file():
        with sqlite3.connect(cache_path) as connection:
            cache = connection.execute(
                """
                SELECT base_url, model, dimensions, text_hash, hex(vector_blob)
                FROM embedding_cache
                ORDER BY base_url, model, dimensions, text_hash
                """
            ).fetchall()

    artifacts = {}
    for artifact_kind in ("text", "raw"):
        artifact_dir = workspace / "artifacts" / artifact_kind
        if artifact_dir.is_dir():
            artifacts[artifact_kind] = {
                path.relative_to(artifact_dir).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(artifact_dir.rglob("*"))
                if path.is_file()
            }
        else:
            artifacts[artifact_kind] = {}

    pageindex_dir = workspace / "artifacts" / "pageindex_client"
    meta_path = pageindex_dir / "_meta.json"
    artifacts["pageindex_client"] = {
        "meta": json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.is_file()
        else {},
        "documents": {
            path.name: {
                "json": json.loads(path.read_text(encoding="utf-8")),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted(pageindex_dir.glob("*.json"))
            if path.name != "_meta.json"
        },
    }

    return {
        "catalog": catalog,
        "summary": summary,
        "cache": cache,
        "artifacts": artifacts,
    }


def open_test_filesystem(workspace):
    from pageindex.filesystem import PageIndexFileSystem

    return PageIndexFileSystem(
        workspace,
        summary_projection_embedding_base_url="https://example.invalid/v1",
        summary_projection_embedding_model="test-embedding",
        summary_projection_embedding_dimensions=3,
        summary_projection_embedding_api_key="runtime-secret",
    )


def nested_path_with_length(root, target_length):
    path = Path(root)
    while len(str(path)) < target_length:
        component_length = min(200, target_length - len(str(path)) - 1)
        path /= "x" * component_length
    assert len(str(path)) == target_length
    return path


def test_cli_rejects_removed_ls_command_with_structured_error(tmp_path, capsys):
    from pageindex.filesystem.cli import main

    status = main(["--workspace", str(tmp_path / "workspace"), "ls", "/"])

    payload = json.loads(capsys.readouterr().out)
    assert status == 2
    assert payload == {
        "success": False,
        "error": {"code": "invalid_command", "message": "Unsupported command: ls"},
        "next_steps": [],
    }


def test_fresh_cli_workspace_uses_the_five_table_catalog_schema_v2(tmp_path, capsys):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "tree", "/", "-L", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["success"] is True

    with sqlite3.connect(workspace / "filesystem.sqlite") as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }

    assert version == 2
    assert tables == {
        "files",
        "folders",
        "file_folders",
        "metadata_fields",
        "metadata_values",
    }


def test_cli_rejects_legacy_catalog_without_mutating_it(tmp_path, capsys):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = workspace / "filesystem.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE legacy_sentinel(value TEXT NOT NULL)")
        connection.execute("INSERT INTO legacy_sentinel(value) VALUES ('preserve me')")
        connection.execute("PRAGMA user_version = 1")
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    status = main(["--workspace", str(workspace), "tree", "/", "-L", "1"])

    assert status == 1
    assert "migrate_pifs_workspace.py" in capsys.readouterr().err
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "mutation",
    ["extra_column", "missing_index", "missing_primary_and_foreign_keys"],
)
def test_cli_rejects_pseudo_v2_catalog_without_mutating_workspace(
    mutation, tmp_path, capsys
):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "tree", "/", "-L", "1"]) == 0
    capsys.readouterr()
    database = workspace / "filesystem.sqlite"
    with sqlite3.connect(database) as connection:
        if mutation == "extra_column":
            connection.execute("ALTER TABLE files ADD COLUMN legacy_provider TEXT")
        elif mutation == "missing_index":
            connection.execute("DROP INDEX idx_files_external_id")
        else:
            connection.executescript(
                """
                ALTER TABLE file_folders RENAME TO legacy_file_folders;
                CREATE TABLE file_folders (
                    file_ref TEXT NOT NULL,
                    folder_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                DROP TABLE legacy_file_folders;
                CREATE INDEX idx_file_folders_folder ON file_folders(folder_id);
                """
            )
    before = workspace_state(workspace)

    status = main(["--workspace", str(workspace), "tree", "/", "-L", "1"])

    assert status == 1
    assert "migrate_pifs_workspace.py" in capsys.readouterr().err
    assert workspace_state(workspace) == before


def test_cli_rejects_partial_legacy_projection_without_creating_summary(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "tree", "/", "-L", "1"]) == 0
    capsys.readouterr()
    projection_dir = workspace / "artifacts" / "projection_indexes"
    projection_dir.mkdir(parents=True)
    cache_path = projection_dir / "embedding_cache.sqlite"
    with sqlite3.connect(cache_path) as connection:
        connection.execute(
            "CREATE TABLE embedding_cache(provider TEXT, model TEXT, text_hash TEXT)"
        )
        connection.execute("PRAGMA user_version = 1")
    before = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")

    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 1

    assert "migrate_pifs_workspace.py" in capsys.readouterr().err
    assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == before
    assert not (projection_dir / "summary.sqlite").exists()
    assert catalog_file_count(workspace) == 0


def test_cli_preflights_partial_projection_before_creating_catalog(tmp_path, capsys):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    projection_dir = workspace / "artifacts" / "projection_indexes"
    projection_dir.mkdir(parents=True)
    cache_path = projection_dir / "embedding_cache.sqlite"
    with sqlite3.connect(cache_path) as connection:
        connection.execute(
            "CREATE TABLE embedding_cache(provider TEXT, model TEXT, text_hash TEXT)"
        )
        connection.execute("PRAGMA user_version = 1")
    before = workspace_state(workspace)

    status = main(["--workspace", str(workspace), "tree", "/", "-L", "1"])

    assert status == 1
    assert "migrate_pifs_workspace.py" in capsys.readouterr().err
    assert workspace_state(workspace) == before
    assert not (workspace / "filesystem.sqlite").exists()


@pytest.mark.parametrize("catalog_state", ["missing", "zero_byte"])
def test_cli_rejects_projection_pair_without_valid_catalog_before_mutating_workspace(
    catalog_state, tmp_path, capsys
):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    create_projection_v2(workspace)
    if catalog_state == "zero_byte":
        (workspace / "filesystem.sqlite").write_bytes(b"")
    before = workspace_state(workspace)

    status = main(["--workspace", str(workspace), "tree", "/", "-L", "1"])

    assert status == 1
    assert "migrate_pifs_workspace.py" in capsys.readouterr().err
    assert workspace_state(workspace) == before
    if catalog_state == "missing":
        assert not (workspace / "filesystem.sqlite").exists()
    else:
        assert (workspace / "filesystem.sqlite").stat().st_size == 0


@pytest.mark.parametrize(
    "projection_state",
    ["both_broken", "broken_summary_only", "broken_summary_with_valid_cache"],
)
def test_cli_rejects_broken_projection_symlinks_without_mutating_workspace(
    projection_state, tmp_path, capsys
):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "tree", "/", "-L", "1"]) == 0
    capsys.readouterr()
    projection_dir = workspace / "artifacts" / "projection_indexes"
    if projection_state == "broken_summary_with_valid_cache":
        create_projection_v2(workspace)
        (projection_dir / "summary.sqlite").unlink()
    else:
        projection_dir.mkdir(parents=True)
    (projection_dir / "summary.sqlite").symlink_to("missing-summary.sqlite")
    if projection_state == "both_broken":
        (projection_dir / "embedding_cache.sqlite").symlink_to(
            "missing-cache.sqlite"
        )
    before = workspace_state(workspace)

    status = main(["--workspace", str(workspace), "tree", "/", "-L", "1"])

    assert status == 1
    assert "migrate_pifs_workspace.py" in capsys.readouterr().err
    assert workspace_state(workspace) == before


def test_cli_rejects_existing_zero_byte_catalog_without_projection_or_mutation(
    tmp_path, capsys
):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    catalog = workspace / "filesystem.sqlite"
    catalog.write_bytes(b"")
    before = workspace_state(workspace)

    status = main(["--workspace", str(workspace), "tree", "/", "-L", "1"])

    assert status == 1
    assert "migrate_pifs_workspace.py" in capsys.readouterr().err
    assert workspace_state(workspace) == before
    assert catalog.stat().st_size == 0


@pytest.mark.parametrize("mutation", ["orphan_projection", "missing_root"])
def test_cli_rejects_inconsistent_catalog_projection_relationships_without_mutation(
    mutation, tmp_path, capsys
):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "tree", "/", "-L", "1"]) == 0
    capsys.readouterr()
    projection_dir = create_projection_v2(workspace)
    insert_projection_document(projection_dir / "summary.sqlite")
    if mutation == "missing_root":
        with sqlite3.connect(workspace / "filesystem.sqlite") as connection:
            connection.execute("DELETE FROM folders WHERE path = '/'")
    before = workspace_state(workspace)

    status = main(["--workspace", str(workspace), "tree", "/", "-L", "1"])

    assert status == 1
    assert "migrate_pifs_workspace.py" in capsys.readouterr().err
    assert workspace_state(workspace) == before


@pytest.mark.parametrize(
    ("catalog_state", "projection_state", "should_fail"),
    [
        ("active", "projected", False),
        ("active", "missing", True),
        ("deleted", "missing", False),
        ("deleted", "projected", True),
    ],
)
def test_runtime_requires_complete_projection_for_every_active_catalog_file(
    catalog_state, projection_state, should_fail, tmp_path, monkeypatch
):
    install_network_fakes(monkeypatch)
    workspace = tmp_path / "workspace"
    source = tmp_path / "notes.md"
    source.write_text("alpha consistency evidence", encoding="utf-8")
    filesystem = open_test_filesystem(workspace)
    file_ref = filesystem.register_file(
        storage_uri=source.as_uri(),
        folder_path="/documents",
        title=source.name,
        content_type="text/markdown",
        content=source.read_text(encoding="utf-8"),
    )
    if catalog_state == "deleted":
        with sqlite3.connect(workspace / "filesystem.sqlite") as connection:
            connection.execute(
                "UPDATE files SET deleted_at = '2026-01-02 03:04:05' "
                "WHERE file_ref = ?",
                (file_ref,),
            )
    if projection_state == "missing":
        delete_projection_document(
            workspace / "artifacts" / "projection_indexes" / "summary.sqlite",
            file_ref,
        )
    before = workspace_state(workspace)

    if should_fail:
        with pytest.raises(RuntimeError, match="migrate_pifs_workspace.py"):
            open_test_filesystem(workspace)
    else:
        open_test_filesystem(workspace)

    assert workspace_state(workspace) == before


@pytest.mark.parametrize(("catalog_state", "should_fail"), [("active", True), ("deleted", False)])
def test_runtime_requires_projection_pair_when_catalog_only_has_active_files(
    catalog_state, should_fail, tmp_path, monkeypatch
):
    install_network_fakes(monkeypatch)
    workspace = tmp_path / "workspace"
    source = tmp_path / "notes.md"
    source.write_text("alpha catalog-only evidence", encoding="utf-8")
    filesystem = open_test_filesystem(workspace)
    file_ref = filesystem.register_file(
        storage_uri=source.as_uri(),
        folder_path="/documents",
        title=source.name,
        content_type="text/markdown",
        content=source.read_text(encoding="utf-8"),
    )
    if catalog_state == "deleted":
        with sqlite3.connect(workspace / "filesystem.sqlite") as connection:
            connection.execute(
                "UPDATE files SET deleted_at = '2026-01-02 03:04:05' "
                "WHERE file_ref = ?",
                (file_ref,),
            )
    projection_dir = workspace / "artifacts" / "projection_indexes"
    (projection_dir / "summary.sqlite").unlink()
    (projection_dir / "embedding_cache.sqlite").unlink()
    before = workspace_state(workspace)

    if should_fail:
        with pytest.raises(RuntimeError, match="migrate_pifs_workspace.py"):
            open_test_filesystem(workspace)
    else:
        open_test_filesystem(workspace)

    assert workspace_state(workspace) == before


@pytest.mark.parametrize(
    "mutation",
    ["missing_vector", "extra_vector", "source_type_mismatch", "wrong_blob_length"],
)
def test_runtime_rejects_projection_document_vector_mismatches_without_mutation(
    mutation, tmp_path, monkeypatch
):
    import sqlite_vec

    install_network_fakes(monkeypatch)
    workspace = tmp_path / "workspace"
    source = tmp_path / "notes.md"
    source.write_text("alpha vector consistency evidence", encoding="utf-8")
    filesystem = open_test_filesystem(workspace)
    filesystem.register_file(
        storage_uri=source.as_uri(),
        folder_path="/documents",
        title=source.name,
        content_type="text/markdown",
        content=source.read_text(encoding="utf-8"),
    )
    summary_path = workspace / "artifacts" / "projection_indexes" / "summary.sqlite"
    with connect_summary_database(summary_path) as connection:
        row = connection.execute(
            "SELECT rowid, source_type, embedding FROM semantic_index_vec"
        ).fetchone()
        assert row is not None
        if mutation == "missing_vector":
            connection.execute(
                "DELETE FROM semantic_index_vec WHERE rowid = ?", (row[0],)
            )
        elif mutation == "extra_vector":
            connection.execute(
                "INSERT INTO semantic_index_vec(rowid, source_type, embedding) "
                "VALUES (999, 'markdown', ?)",
                (sqlite_vec.serialize_float32([0.0, 0.0, 1.0]),),
            )
        elif mutation == "source_type_mismatch":
            connection.execute(
                "DELETE FROM semantic_index_vec WHERE rowid = ?", (row[0],)
            )
            connection.execute(
                "INSERT INTO semantic_index_vec(rowid, source_type, embedding) "
                "VALUES (?, 'pdf', ?)",
                (row[0], row[2]),
            )
        else:
            connection.execute(
                "UPDATE semantic_index_vec_vector_chunks00 SET vectors = zeroblob(4)"
            )
    before = workspace_state(workspace)

    with pytest.raises(RuntimeError, match="migrate this workspace"):
        open_test_filesystem(workspace)

    assert workspace_state(workspace) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "summary_extra_column",
        "summary_missing_primary_and_unique",
        "summary_missing_index",
        "summary_extra_config_key",
        "summary_vec_dimension_mismatch",
        "cache_extra_column",
        "cache_missing_primary_key",
    ],
)
def test_cli_rejects_pseudo_v2_projection_without_mutating_workspace(
    mutation, tmp_path, capsys
):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    projection_dir = create_projection_v2(workspace)
    summary_path = projection_dir / "summary.sqlite"
    cache_path = projection_dir / "embedding_cache.sqlite"
    if mutation.startswith("summary_"):
        with sqlite3.connect(summary_path) as connection:
            if mutation == "summary_extra_column":
                connection.execute(
                    "ALTER TABLE semantic_index_docs ADD COLUMN legacy_provider TEXT"
                )
            elif mutation == "summary_missing_primary_and_unique":
                connection.executescript(
                    """
                    DROP INDEX idx_semantic_index_docs_external_id;
                    DROP INDEX idx_semantic_index_docs_source_type;
                    ALTER TABLE semantic_index_docs RENAME TO legacy_semantic_index_docs;
                    CREATE TABLE semantic_index_docs (
                        rowid INTEGER,
                        file_ref TEXT NOT NULL,
                        external_id TEXT,
                        source_type TEXT NOT NULL DEFAULT '',
                        title TEXT NOT NULL DEFAULT '',
                        text_hash TEXT NOT NULL,
                        text_chars INTEGER NOT NULL DEFAULT 0,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                    DROP TABLE legacy_semantic_index_docs;
                    CREATE INDEX idx_semantic_index_docs_external_id
                      ON semantic_index_docs(external_id);
                    CREATE INDEX idx_semantic_index_docs_source_type
                      ON semantic_index_docs(source_type);
                    """
                )
            elif mutation == "summary_missing_index":
                connection.execute("DROP INDEX idx_semantic_index_docs_external_id")
            elif mutation == "summary_extra_config_key":
                connection.execute(
                    "INSERT INTO semantic_index_config(key, value) "
                    "VALUES ('legacy_provider', 'openai')"
                )
            else:
                connection.execute(
                    "UPDATE semantic_index_config SET value = '4' WHERE key = 'dimension'"
                )
                connection.execute(
                    "UPDATE semantic_index_config SET value = ? WHERE key = 'metadata'",
                    (
                        json.dumps(
                            {
                                "base_url": "https://example.invalid/v1",
                                "model": "test-embedding",
                                "dimensions": 4,
                            }
                        ),
                    ),
                )
    else:
        with sqlite3.connect(cache_path) as connection:
            if mutation == "cache_extra_column":
                connection.execute(
                    "ALTER TABLE embedding_cache ADD COLUMN provider TEXT"
                )
            else:
                connection.executescript(
                    """
                    ALTER TABLE embedding_cache RENAME TO legacy_embedding_cache;
                    CREATE TABLE embedding_cache (
                        base_url TEXT NOT NULL,
                        model TEXT NOT NULL,
                        dimensions INTEGER NOT NULL CHECK(dimensions > 0),
                        text_hash TEXT NOT NULL,
                        vector_blob BLOB NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                    DROP TABLE legacy_embedding_cache;
                    """
                )
    before = workspace_state(workspace)

    status = main(["--workspace", str(workspace), "tree", "/", "-L", "1"])

    assert status == 1
    assert "migrate_pifs_workspace.py" in capsys.readouterr().err
    assert workspace_state(workspace) == before
    assert not (workspace / "filesystem.sqlite").exists()


def test_cli_rejects_noncanonical_vec0_distance_without_mutating_workspace(
    tmp_path, capsys
):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "tree", "/", "-L", "1"]) == 0
    capsys.readouterr()
    projection_dir = create_projection_v2(workspace)
    with connect_summary_database(projection_dir / "summary.sqlite") as connection:
        connection.execute("DROP TABLE semantic_index_vec")
        connection.execute(
            "CREATE VIRTUAL TABLE semantic_index_vec USING "
            "vec0(source_type TEXT partition key, "
            "embedding float[3] distance_metric=cosine)"
        )
    before = workspace_state(workspace)

    status = main(["--workspace", str(workspace), "tree", "/", "-L", "1"])

    assert status == 1
    assert "migrate_pifs_workspace.py" in capsys.readouterr().err
    assert workspace_state(workspace) == before


def test_cli_accepts_canonical_vec0_declaration_with_spacing_and_case_variation(
    tmp_path, capsys
):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "tree", "/", "-L", "1"]) == 0
    capsys.readouterr()
    projection_dir = create_projection_v2(workspace)
    with connect_summary_database(projection_dir / "summary.sqlite") as connection:
        connection.execute("DROP TABLE semantic_index_vec")
        connection.execute(
            "CREATE VIRTUAL TABLE semantic_index_vec USING "
            "VEC0( source_type TEXT   PARTITION KEY , embedding FLOAT [ 3 ] )"
        )
    before = workspace_state(workspace)

    status = main(["--workspace", str(workspace), "tree", "/", "-L", "1"])

    assert status == 0
    assert json.loads(capsys.readouterr().out)["success"] is True
    assert workspace_state(workspace) == before


@pytest.mark.parametrize(
    "identity_case",
    [
        "summary_base_url",
        "summary_model",
        "cache_base_url",
        "cache_model",
    ],
)
def test_cli_rejects_noncanonical_persisted_embedding_identity_without_mutation(
    identity_case, tmp_path, capsys
):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "tree", "/", "-L", "1"]) == 0
    capsys.readouterr()
    projection_dir = create_projection_v2(workspace)
    if identity_case.startswith("summary_"):
        with sqlite3.connect(projection_dir / "summary.sqlite") as connection:
            metadata = json.loads(
                connection.execute(
                    "SELECT value FROM semantic_index_config WHERE key = 'metadata'"
                ).fetchone()[0]
            )
            if identity_case == "summary_base_url":
                metadata["base_url"] = "HTTPS://EXAMPLE.INVALID/v1/"
            else:
                metadata["model"] = " test-embedding "
            connection.execute(
                "UPDATE semantic_index_config SET value = ? WHERE key = 'metadata'",
                (json.dumps(metadata, sort_keys=True),),
            )
    else:
        base_url = (
            "HTTPS://EXAMPLE.INVALID/v1/"
            if identity_case == "cache_base_url"
            else "https://example.invalid/v1"
        )
        model = (
            " test-embedding "
            if identity_case == "cache_model"
            else "test-embedding"
        )
        with sqlite3.connect(projection_dir / "embedding_cache.sqlite") as connection:
            connection.execute(
                "INSERT INTO embedding_cache("
                "base_url, model, dimensions, text_hash, vector_blob"
                ") VALUES (?, ?, 3, 'cache-hash', ?)",
                (base_url, model, b"\0" * 12),
            )
    before = workspace_state(workspace)

    status = main(["--workspace", str(workspace), "tree", "/", "-L", "1"])

    assert status == 1
    assert "migrate_pifs_workspace.py" in capsys.readouterr().err
    assert workspace_state(workspace) == before


def test_cli_allows_fresh_listing_when_projection_is_completely_absent(tmp_path, capsys):
    from pageindex.filesystem.cli import main

    workspace = tmp_path / "workspace"

    assert main(["--workspace", str(workspace), "tree", "/", "-L", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["success"] is True
    assert (workspace / "filesystem.sqlite").is_file()
    assert not (workspace / "artifacts" / "projection_indexes").exists()

    assert main(["--workspace", str(workspace), "tree", "/", "-L", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["success"] is True
    assert not (workspace / "artifacts" / "projection_indexes").exists()


def test_pageindex_is_the_only_public_python_entry_point_for_pifs():
    import pageindex
    import pageindex.filesystem as filesystem_module

    assert pageindex.PageIndexFileSystem is filesystem_module.PageIndexFileSystem
    for internal_name in (
        "PIFSCommandExecutor",
        "OpenResult",
        "SearchResult",
        "SummaryProjectionIndexer",
        "SemanticProjectionSearchBackend",
        "SQLiteVecSemanticIndex",
        "SemanticIndexRecord",
        "SemanticSearchResult",
    ):
        assert not hasattr(filesystem_module, internal_name)
    assert not hasattr(filesystem_module, "_LAZY_EXPORTS")


def test_cli_add_creates_migration_compatible_summary_and_cache_v2(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    config = write_embedding_config(tmp_path, monkeypatch)
    config_values = json.loads(config.read_text(encoding="utf-8"))
    config_values["embedding_api_key"] = "config-secret"
    config.write_text(json.dumps(config_values), encoding="utf-8")
    monkeypatch.delenv("PIFS_EMBEDDING_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"

    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    output = capsys.readouterr().out

    projection_dir = workspace / "artifacts" / "projection_indexes"
    with sqlite3.connect(projection_dir / "summary.sqlite") as summary:
        summary_version = summary.execute("PRAGMA user_version").fetchone()[0]
        summary_tables = logical_tables(summary)
        metadata = json.loads(
            summary.execute(
                "SELECT value FROM semantic_index_config WHERE key = 'metadata'"
            ).fetchone()[0]
        )
    with sqlite3.connect(projection_dir / "embedding_cache.sqlite") as cache:
        cache_version = cache.execute("PRAGMA user_version").fetchone()[0]
        cache_tables = logical_tables(cache)
        cache_columns = {
            row[1] for row in cache.execute("PRAGMA table_info(embedding_cache)")
        }

    assert summary_version == cache_version == 2
    assert summary_tables == {
        "semantic_index_config",
        "semantic_index_docs",
        "semantic_index_vec",
    }
    assert cache_tables == {"embedding_cache"}
    assert cache_columns == {
        "base_url",
        "model",
        "dimensions",
        "text_hash",
        "vector_blob",
        "created_at",
    }
    assert metadata == {
        "base_url": "https://example.invalid/v1",
        "model": "test-embedding",
        "dimensions": 3,
    }
    for database in (
        workspace / "filesystem.sqlite",
        projection_dir / "summary.sqlite",
        projection_dir / "embedding_cache.sqlite",
    ):
        assert b"config-secret" not in database.read_bytes()
    assert "config-secret" not in output


def test_cli_add_reports_path_without_persistence_identity(tmp_path, monkeypatch, capsys):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")

    assert (
        main(
            [
                "--workspace",
                str(tmp_path / "workspace"),
                "add",
                str(source),
                "/documents",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert output == "added: /documents/notes.md\n"


def test_cli_browse_reopens_owned_file_with_canonical_result_fields(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()
    source.unlink()

    assert (
        main(
            [
                "--workspace",
                str(workspace),
                "browse",
                "/documents",
                "alpha",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    document = payload["data"]["documents"][0]
    assert set(document) == {
        "path",
        "document_id",
        "title",
        "status",
        "rank",
        "similarity",
        "summary",
        "metadata",
        "folder_path",
        "folder_paths",
    }
    assert document["path"] == "/documents/notes.md"
    assert document["title"] == "notes.md"
    assert document["status"] == "built"
    assert document["rank"] == 1
    assert document["summary"].startswith("Summary for notes.md")
    assert document["folder_path"] == "/documents"


def test_cli_stat_translates_persistence_identity_once(tmp_path, monkeypatch, capsys):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()

    assert main(["--workspace", str(workspace), "stat", "/documents/notes.md"]) == 0

    document = json.loads(capsys.readouterr().out)["data"]["document"]
    assert set(document) == {
        "path",
        "document_id",
        "title",
        "status",
        "content_type",
        "metadata",
        "metadata_status",
        "folder_paths",
    }
    assert document["path"] == "/documents/notes.md"
    assert document["title"] == "notes.md"
    assert document["status"] == "built"
    assert document["folder_paths"] == ["/documents"]


def test_cli_cat_reads_structure_without_internal_identity(tmp_path, monkeypatch, capsys):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()
    source.unlink()

    assert (
        main(
            [
                "--workspace",
                str(workspace),
                "cat",
                "/documents/notes.md",
                "--structure",
            ]
        )
        == 0
    )

    data = json.loads(capsys.readouterr().out)["data"]
    assert set(data["document"]) == {
        "path",
        "document_id",
        "title",
        "status",
        "content_type",
        "metadata",
        "metadata_status",
        "folder_paths",
        "available",
    }
    assert data["structure"] == [
        {"title": "notes", "node_id": "0001", "line_num": 1, "nodes": []}
    ]


def create_metadata_scope_cli_fixture(tmp_path, monkeypatch, capsys):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    records = [
        (
            "current.md",
            "alpha current evidence",
            "/documents",
            {"year": 2024, "ticker": "AAPL", "sector": "finance/tech"},
        ),
        (
            "filing.md",
            "alpha filing evidence",
            "/documents/sec-filings",
            {"year": 2024, "ticker": "AAPL", "doc_type": "10-K"},
        ),
        (
            "prior.md",
            "alpha prior evidence",
            "/documents",
            {"year": 2023, "ticker": "MSFT"},
        ),
        ("archive.md", "archive evidence", "/archive", {"region": "EMEA"}),
    ]
    for filename, content, folder, metadata in records:
        source = tmp_path / filename
        source.write_text(content, encoding="utf-8")
        assert main(["--workspace", str(workspace), "add", str(source), folder]) == 0
        capsys.readouterr()
        assert main(
            [
                "--workspace",
                str(workspace),
                "setmeta",
                f"{folder}/{filename}",
                json.dumps(metadata),
            ]
        ) == 0
        capsys.readouterr()
    return workspace


def test_tree_folder_trailing_slash_lists_scope_local_metadata_axes(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    workspace = create_metadata_scope_cli_fixture(tmp_path, monkeypatch, capsys)

    payloads = []
    for folder_path in ("/documents", "/documents/"):
        assert main(
            ["--workspace", str(workspace), "tree", folder_path, "-L", "1"]
        ) == 0
        payloads.append(json.loads(capsys.readouterr().out))

    assert payloads[1]["data"] == payloads[0]["data"]
    tree = payloads[0]["data"]["tree"]
    assert tree["path"] == "/documents"
    axes = {
        row["name"]: row
        for row in tree["folders"]
        if row["type"] == "metadata_axis"
    }

    assert set(axes) == {"@doc_type", "@sector", "@ticker", "@year"}
    assert {name: row["path"] for name, row in axes.items()} == {
        "@doc_type": "/documents/@doc_type",
        "@sector": "/documents/@sector",
        "@ticker": "/documents/@ticker",
        "@year": "/documents/@year",
    }
    assert {row["type"] for row in axes.values()} == {"metadata_axis"}
    physical = [row for row in tree["folders"] if row["type"] == "folder"]
    assert [(row["name"], row["path"]) for row in physical] == [
        ("sec-filings", "/documents/sec-filings")
    ]
    assert all(row["type"] != "file" for row in tree["folders"])
    assert "@region" not in axes


def test_tree_metadata_axis_trailing_slash_lists_actionable_paginated_values(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    workspace = create_metadata_scope_cli_fixture(tmp_path, monkeypatch, capsys)

    payloads = []
    for axis_path in ("/documents/@year", "/documents/@year/"):
        assert main(
            ["--workspace", str(workspace), "tree", axis_path, "-L", "1"]
        ) == 0
        payloads.append(json.loads(capsys.readouterr().out))

    values = payloads[0]["data"]["tree"]["folders"]
    assert payloads[1]["data"] == payloads[0]["data"]
    assert [(row["name"], row["type"], row["path"]) for row in values] == [
        ("2024", "metadata_value", "/documents/@year/2024"),
        ("2023", "metadata_value", "/documents/@year/2023"),
    ]
    assert payloads[0]["data"]["pagination"] == {
        "page": 1,
        "page_size": 50,
        "has_more": False,
        "next_page": None,
    }

    selected = values[0]["path"]
    scoped_payloads = []
    for value_path in (selected, f"{selected}/"):
        assert main(
            ["--workspace", str(workspace), "tree", value_path, "-L", "1"]
        ) == 0
        scoped_payloads.append(json.loads(capsys.readouterr().out))
    assert scoped_payloads[1]["data"] == scoped_payloads[0]["data"]
    selected_tree = scoped_payloads[0]["data"]["tree"]
    assert selected_tree["path"] == selected
    assert {row["title"] for row in selected_tree["files"]} == {
        "current.md",
        "filing.md",
    }
    selected_folders = {
        (row["type"], row["name"]): row["path"]
        for row in selected_tree["folders"]
    }
    assert selected_folders == {
        ("folder", "sec-filings"): "/documents/sec-filings/@year/2024",
        ("metadata_axis", "@doc_type"): "/documents/@year/2024/@doc_type",
        ("metadata_axis", "@sector"): "/documents/@year/2024/@sector",
        ("metadata_axis", "@ticker"): "/documents/@year/2024/@ticker",
    }
    assert main(
        ["--workspace", str(workspace), "browse", selected, "alpha"]
    ) == 0
    documents = json.loads(capsys.readouterr().out)["data"]["documents"]
    assert {row["title"] for row in documents} == {"current.md", "filing.md"}

    locator = selected_tree["files"][0]["path"]
    for command in (
        ["stat", locator],
        ["cat", locator, "--structure"],
        ["grep", "alpha", locator],
    ):
        assert main(["--workspace", str(workspace), *command]) == 0
        capsys.readouterr()

    child_scope = selected_folders[("folder", "sec-filings")]
    assert main(["--workspace", str(workspace), "tree", child_scope, "-L", "1"]) == 0
    child_tree = json.loads(capsys.readouterr().out)["data"]["tree"]
    assert [row["title"] for row in child_tree["files"]] == ["filing.md"]

    assert main(
        ["--workspace", str(workspace), "tree", "/documents/@sector/", "-L", "1"]
    ) == 0
    encoded = json.loads(capsys.readouterr().out)["data"]["tree"]["folders"][0]
    assert encoded["path"] == "/documents/@sector/finance%2Ftech"
    assert main(
        ["--workspace", str(workspace), "tree", encoded["path"], "-L", "1"]
    ) == 0
    capsys.readouterr()


def test_tree_metadata_values_keep_fixed_fifty_item_pagination(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "tickers.md"
    source.write_text("alpha ticker evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()
    tickers = [f"T{index:02d}" for index in range(55)]
    assert main(
        [
            "--workspace",
            str(workspace),
            "setmeta",
            "/documents/tickers.md",
            json.dumps({"ticker": tickers}),
        ]
    ) == 0
    capsys.readouterr()

    assert main(
        ["--workspace", str(workspace), "tree", "/documents/@ticker/", "-L", "1"]
    ) == 0
    first = json.loads(capsys.readouterr().out)["data"]
    assert [row["value"] for row in first["tree"]["folders"]] == tickers[:50]
    assert first["pagination"] == {
        "page": 1,
        "page_size": 50,
        "has_more": True,
        "next_page": 2,
    }

    assert main(
        [
            "--workspace",
            str(workspace),
            "tree",
            "/documents/@ticker",
            "--page",
            "2",
        ]
    ) == 0
    second = json.loads(capsys.readouterr().out)["data"]
    assert [row["value"] for row in second["tree"]["folders"]] == tickers[50:]
    assert second["pagination"] == {
        "page": 2,
        "page_size": 50,
        "has_more": False,
        "next_page": None,
    }


def test_tree_metadata_dot_values_return_actionable_encoded_scope_paths(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("alpha dot value evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()
    assert main(
        [
            "--workspace",
            str(workspace),
            "setmeta",
            "/documents/notes.md",
            json.dumps({"tag": [".", ".."]}),
        ]
    ) == 0
    capsys.readouterr()

    assert main(["--workspace", str(workspace), "tree", "/documents/@tag"]) == 0
    values = json.loads(capsys.readouterr().out)["data"]["tree"]["folders"]
    assert {row["value"]: row["path"] for row in values} == {
        ".": "/documents/@tag/%2E",
        "..": "/documents/@tag/%2E%2E",
    }

    for scope_path in (row["path"] for row in values):
        assert main(
            ["--workspace", str(workspace), "tree", scope_path, "-L", "1"]
        ) == 0
        file_path = json.loads(capsys.readouterr().out)["data"]["tree"]["files"][0][
            "path"
        ]
        assert main(
            ["--workspace", str(workspace), "browse", scope_path, "alpha"]
        ) == 0
        document = json.loads(capsys.readouterr().out)["data"]["documents"][0]
        assert document["path"] == file_path
        assert main(["--workspace", str(workspace), "stat", file_path]) == 0
        capsys.readouterr()


def test_metadata_virtual_paths_use_alternating_encoded_field_value_segments(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "--workspace",
                str(workspace),
                "setmeta",
                "/documents/notes.md",
                '{"year": 2024, "sector": "finance/tech"}',
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["--workspace", str(workspace), "tree", "/documents/@year"]) == 0
    year_payload = json.loads(capsys.readouterr().out)
    values = year_payload["data"]["tree"]["folders"]
    assert values[0]["path"] == "/documents/@year/2024"
    assert year_payload["next_steps"] == [
        'browse /documents/@year/<value> "<query>"'
    ]

    assert (
        main(
            [
                "--workspace",
                str(workspace),
                "tree",
                "/documents/@year/2024",
                "-L",
                "1",
            ]
        )
        == 0
    )
    scoped_tree = json.loads(capsys.readouterr().out)["data"]["tree"]
    assert scoped_tree["files"][0]["path"] == "/documents/@year/2024/notes.md"
    assert [folder["path"] for folder in scoped_tree["folders"]] == [
        "/documents/@year/2024/@sector"
    ]

    assert main(
        [
            "--workspace",
            str(workspace),
            "tree",
            "/documents/@year/2024/@sector",
        ]
    ) == 0
    sector = json.loads(capsys.readouterr().out)["data"]["tree"]["folders"][0]
    assert sector["path"] == "/documents/@year/2024/@sector/finance%2Ftech"

    locator = f"{sector['path']}/notes.md"
    for command in (
        ["stat", locator],
        ["cat", locator, "--structure"],
        ["grep", "alpha", locator],
    ):
        assert main(["--workspace", str(workspace), *command]) == 0
        capsys.readouterr()

    assert main(
        ["--workspace", str(workspace), "tree", "/documents/@year=2024"]
    ) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert "@field/value" in error["message"]


def test_setmeta_translates_identity_at_the_cli_boundary(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()

    assert main(
        [
            "--workspace",
            str(workspace),
            "setmeta",
            "/documents/notes.md",
            '{"year": 2024}',
        ]
    ) == 0
    document = json.loads(capsys.readouterr().out)

    assert set(document) == {
        "path",
        "document_id",
        "title",
        "status",
        "metadata",
        "metadata_status",
    }
    assert document["path"] == "/documents/notes.md"
    assert document["metadata"]["year"] == 2024


def test_setmeta_by_file_ref_returns_an_actionable_path_without_leaking_identity(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()
    with sqlite3.connect(workspace / "filesystem.sqlite") as connection:
        file_ref = connection.execute("SELECT file_ref FROM files").fetchone()[0]

    assert main(
        [
            "--workspace",
            str(workspace),
            "setmeta",
            file_ref,
            '{"year": 2024}',
        ]
    ) == 0
    output = capsys.readouterr().out
    document = json.loads(output)

    assert document["path"] == "/documents/notes.md"
    assert file_ref not in output
    assert main(["--workspace", str(workspace), "stat", document["path"]]) == 0
    stat = json.loads(capsys.readouterr().out)["data"]["document"]
    assert stat["path"] == document["path"]


@pytest.mark.parametrize("operation", ["replace", "clear"])
def test_metadata_scoped_setmeta_returns_the_post_update_actionable_path(
    operation, tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()
    assert main(
        [
            "--workspace",
            str(workspace),
            "setmeta",
            "/documents/notes.md",
            '{"year": 2024}',
        ]
    ) == 0
    capsys.readouterr()
    old_path = "/documents/@year/2024/notes.md"
    command = ["--workspace", str(workspace), "setmeta"]
    if operation == "clear":
        command.extend(["--clear", old_path])
    else:
        command.extend([old_path, '{"year": 2025}'])

    assert main(command) == 0
    document = json.loads(capsys.readouterr().out)

    assert document["path"] == "/documents/notes.md"
    assert document["path"] != old_path
    if operation == "clear":
        assert "year" not in document["metadata"]
    else:
        assert document["metadata"]["year"] == 2025
    assert main(["--workspace", str(workspace), "stat", document["path"]]) == 0
    stat = json.loads(capsys.readouterr().out)["data"]["document"]
    assert stat["path"] == document["path"]


def test_tree_file_records_use_only_command_identity_names(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()

    assert main(["--workspace", str(workspace), "tree", "/documents", "-L", "1"]) == 0
    file_record = json.loads(capsys.readouterr().out)["data"]["tree"]["files"][0]

    assert set(file_record) == {
        "path",
        "document_id",
        "title",
        "status",
        "type",
        "metadata",
    }


def test_duplicate_virtual_leaves_use_actionable_paths_without_internal_ids(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    for folder_name in ("a", "b"):
        source_dir = tmp_path / folder_name
        source_dir.mkdir()
        source = source_dir / "notes.md"
        source.write_text(f"alpha evidence {folder_name}", encoding="utf-8")
        folder = f"/documents/{folder_name}"
        assert main(["--workspace", str(workspace), "add", str(source), folder]) == 0
        capsys.readouterr()
        assert main(
            [
                "--workspace",
                str(workspace),
                "setmeta",
                f"{folder}/notes.md",
                '{"year": 2024}',
            ]
        ) == 0
        capsys.readouterr()

    assert main(
        ["--workspace", str(workspace), "tree", "/documents/@year/2024", "-L", "1"]
    ) == 0
    files = json.loads(capsys.readouterr().out)["data"]["tree"]["files"]
    paths = [row["path"] for row in files]

    assert len(paths) == len(set(paths)) == 2
    assert all("file_" not in path for path in paths)
    for path in paths:
        assert main(["--workspace", str(workspace), "stat", path]) == 0
        assert json.loads(capsys.readouterr().out)["data"]["document"]["path"] == path


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("notes.md", b"markdown alpha evidence"),
        ("report.pdf", b"%PDF-1.4 fake fixture"),
    ],
)
def test_cli_add_owns_supported_documents_and_keeps_evidence_readable(
    filename, content, tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / filename
    source.write_bytes(content)
    workspace = tmp_path / "workspace"

    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()
    source.unlink()

    owned = list((workspace / "artifacts" / "uploads").glob(f"*/{filename}"))
    assert len(owned) == 1
    assert owned[0].read_bytes() == content
    assert main(
        ["--workspace", str(workspace), "grep", "alpha", f"/documents/{filename}"]
    ) == 0
    matches = json.loads(capsys.readouterr().out)["data"]["matches"]
    assert matches[0]["text"].endswith("alpha evidence")


def test_cli_add_rejects_unsupported_type_before_registration(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.txt"
    source.write_text("unsupported", encoding="utf-8")
    workspace = tmp_path / "workspace"

    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 1

    assert "Unsupported file type" in capsys.readouterr().err
    assert catalog_file_count(workspace) == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))


def test_cli_add_rolls_back_when_pageindex_fails(
    tmp_path, monkeypatch, capsys
):
    from pageindex import PageIndexClient
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)

    def fail_index(self, file_path, mode="auto"):
        raise RuntimeError("pageindex unavailable")

    monkeypatch.setattr(PageIndexClient, "index", fail_index)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"

    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 1

    assert "pageindex unavailable" in capsys.readouterr().err
    assert catalog_file_count(workspace) == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    pageindex_cache = workspace / "artifacts" / "pageindex_client"
    assert json.loads((pageindex_cache / "_meta.json").read_text(encoding="utf-8")) == {}
    assert not [path for path in pageindex_cache.glob("*.json") if path.name != "_meta.json"]


def test_cli_add_rolls_back_when_summary_projection_fails(
    tmp_path, monkeypatch, capsys
):
    import openai
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)

    class FailingOpenAI(FakeOpenAI):
        def create(self, *, model, input, dimensions):
            raise RuntimeError("embedding unavailable")

    monkeypatch.setattr(openai, "OpenAI", FailingOpenAI)
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"

    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 1

    assert "summary projection" in capsys.readouterr().err.lower()
    assert catalog_file_count(workspace) == 0
    assert not list((workspace / "artifacts" / "uploads").glob("**/*"))
    with sqlite3.connect(
        workspace / "artifacts" / "projection_indexes" / "summary.sqlite"
    ) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM semantic_index_docs"
        ).fetchone()[0] == 0


@pytest.mark.parametrize("cache_state", ["fresh", "preexisting_shared_key"])
def test_add_restores_embedding_cache_when_vec_upsert_fails(
    cache_state, tmp_path, monkeypatch
):
    from pageindex.filesystem.semantic_index import SQLiteVecSemanticIndex

    install_network_fakes(monkeypatch)
    workspace = tmp_path / "workspace"
    filesystem = open_test_filesystem(workspace)
    source_dir = tmp_path / "add"
    source_dir.mkdir()
    source = source_dir / "notes.md"
    source.write_text("alpha shared cache evidence", encoding="utf-8")

    if cache_state == "preexisting_shared_key":
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        existing = existing_dir / "notes.md"
        existing.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        filesystem.register_file(
            storage_uri=existing.as_uri(),
            folder_path="/documents/existing",
            title=existing.name,
            content_type="text/markdown",
            content=existing.read_text(encoding="utf-8"),
        )
    else:
        create_projection_v2(workspace)
        filesystem = open_test_filesystem(workspace)

    baseline = registration_logical_state(workspace)
    baseline_uploads = sorted(
        path.relative_to(workspace).as_posix()
        for path in workspace.glob("artifacts/uploads/**/*")
    )

    def fail_vec_upsert(self, records):
        raise RuntimeError("vec upsert unavailable after cache write")

    monkeypatch.setattr(SQLiteVecSemanticIndex, "upsert_many", fail_vec_upsert)

    with pytest.raises(RuntimeError, match="vec upsert unavailable after cache write"):
        filesystem.add_file(source, "/documents/new")

    assert registration_logical_state(workspace) == baseline
    assert sorted(
        path.relative_to(workspace).as_posix()
        for path in workspace.glob("artifacts/uploads/**/*")
    ) == baseline_uploads
    reopened = open_test_filesystem(workspace)
    assert registration_logical_state(workspace) == baseline
    if cache_state == "preexisting_shared_key":
        assert reopened.store.file_refs_for_scope() == filesystem.store.file_refs_for_scope()
    else:
        assert reopened.store.file_refs_for_scope() == []


def test_register_file_completes_summary_projection_for_immediate_and_reopen_browse(
    tmp_path, monkeypatch
):
    install_network_fakes(monkeypatch)
    workspace = tmp_path / "workspace"
    source = tmp_path / "notes.md"
    source.write_text("alpha registration evidence", encoding="utf-8")
    filesystem = open_test_filesystem(workspace)
    caller_metadata = {"year": 2024, "labels": {"team": "alpha"}}
    caller_metadata_baseline = json.loads(json.dumps(caller_metadata))

    file_ref = filesystem.register_file(
        storage_uri=source.as_uri(),
        folder_path="/documents",
        title=source.name,
        content_type="text/markdown",
        content=source.read_text(encoding="utf-8"),
        metadata=caller_metadata,
    )

    projection_dir = workspace / "artifacts" / "projection_indexes"
    assert (projection_dir / "summary.sqlite").is_file()
    assert (projection_dir / "embedding_cache.sqlite").is_file()
    stored = filesystem.store.get_file(file_ref)
    assert caller_metadata == caller_metadata_baseline
    assert "summary" not in caller_metadata
    assert stored.metadata["summary"].startswith("Summary for notes.md:")
    assert stored.metadata_status["summary_projection"]["status"] == "ready"
    assert file_ref in {
        row["file_ref"]
        for row in filesystem.browse_semantic_files("/documents", "alpha")["data"]
    }

    reopened = open_test_filesystem(workspace)
    assert file_ref in {
        row["file_ref"]
        for row in reopened.browse_semantic_files("/documents", "alpha")["data"]
    }


def test_register_file_opens_existing_projection_and_upserts_second_document(
    tmp_path, monkeypatch
):
    install_network_fakes(monkeypatch)
    workspace = tmp_path / "workspace"
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("alpha registration evidence", encoding="utf-8")
    second.write_text("beta registration evidence", encoding="utf-8")
    filesystem = open_test_filesystem(workspace)
    first_ref = filesystem.register_file(
        storage_uri=first.as_uri(),
        folder_path="/documents",
        title=first.name,
        content_type="text/markdown",
        content=first.read_text(encoding="utf-8"),
    )

    reopened = open_test_filesystem(workspace)
    second_ref = reopened.register_file(
        storage_uri=second.as_uri(),
        folder_path="/documents",
        title=second.name,
        content_type="text/markdown",
        content=second.read_text(encoding="utf-8"),
    )

    assert reopened.store.get_file(second_ref).metadata_status["summary_projection"]["status"] == "ready"
    assert second_ref in {
        row["file_ref"]
        for row in reopened.browse_semantic_files("/documents", "beta")["data"]
    }
    with connect_summary_database(
        workspace / "artifacts" / "projection_indexes" / "summary.sqlite"
    ) as connection:
        assert connection.execute("SELECT COUNT(*) FROM semantic_index_docs").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM semantic_index_vec").fetchone()[0] == 2
    assert set(reopened.store.file_refs_for_scope()) == {first_ref, second_ref}


def test_register_files_rolls_back_new_batch_when_second_projection_upsert_fails(
    tmp_path, monkeypatch
):
    import openai

    install_network_fakes(monkeypatch)

    class FailingSecondOpenAI(FakeOpenAI):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def create(self, *, model, input, dimensions):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("embedding unavailable for second registration")
            return super().create(model=model, input=input, dimensions=dimensions)

    monkeypatch.setattr(openai, "OpenAI", FailingSecondOpenAI)
    workspace = tmp_path / "workspace"
    filesystem = open_test_filesystem(workspace)
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("alpha registration evidence", encoding="utf-8")
    second.write_text("beta registration evidence", encoding="utf-8")
    baseline = registration_logical_state(workspace)

    with pytest.raises(RuntimeError, match="summary projection.*second registration"):
        filesystem.register_files(
            [
                {
                    "storage_uri": source.as_uri(),
                    "folder_path": "/documents/new",
                    "title": source.name,
                    "content_type": "text/markdown",
                    "content": source.read_text(encoding="utf-8"),
                    "metadata": {"year": 2024},
                }
                for source in (first, second)
            ]
        )

    assert registration_logical_state(workspace) == baseline

    reopened = open_test_filesystem(workspace)
    assert reopened.store.folder_info("/")["path"] == "/"
    assert reopened.store.file_refs_for_scope() == []


def test_register_files_restores_existing_document_when_later_projection_upsert_fails(
    tmp_path, monkeypatch, capsys
):
    import openai
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    original = tmp_path / "original.md"
    replacement = tmp_path / "replacement.md"
    failing = tmp_path / "failing.md"
    original.write_text("alpha original evidence", encoding="utf-8")
    replacement.write_text("alpha replacement evidence", encoding="utf-8")
    failing.write_text("beta second evidence", encoding="utf-8")

    filesystem = open_test_filesystem(workspace)
    existing_ref = filesystem.register_file(
        storage_uri=original.as_uri(),
        external_id="doc-existing",
        folder_path="/documents",
        title=original.name,
        content_type="text/markdown",
        content=original.read_text(encoding="utf-8"),
        metadata={"year": 2023},
    )
    baseline = registration_logical_state(workspace)

    class FailingBetaOpenAI(FakeOpenAI):
        def create(self, *, model, input, dimensions):
            if any("beta" in str(text).lower() for text in input):
                raise RuntimeError("embedding unavailable for second registration")
            return super().create(model=model, input=input, dimensions=dimensions)

    monkeypatch.setattr(openai, "OpenAI", FailingBetaOpenAI)
    reopened = open_test_filesystem(workspace)
    with pytest.raises(RuntimeError, match="summary projection.*second registration"):
        reopened.register_files(
            [
                {
                    "storage_uri": replacement.as_uri(),
                    "external_id": "doc-existing",
                    "folder_path": "/documents/replacement",
                    "title": replacement.name,
                    "content_type": "text/markdown",
                    "content": replacement.read_text(encoding="utf-8"),
                    "metadata": {"year": 2024},
                },
                {
                    "storage_uri": failing.as_uri(),
                    "external_id": "doc-failing",
                    "folder_path": "/documents/new",
                    "title": failing.name,
                    "content_type": "text/markdown",
                    "content": failing.read_text(encoding="utf-8"),
                    "metadata": {"year": 2025},
                },
            ]
        )

    assert registration_logical_state(workspace) == baseline
    strict_reopen = open_test_filesystem(workspace)
    assert strict_reopen.store.file_refs_for_scope() == [existing_ref]
    entry = strict_reopen.store.get_file(existing_ref)
    assert entry.metadata["year"] == 2023
    assert entry.storage_uri == original.as_uri()
    assert strict_reopen.pageindex_structure(existing_ref)["available"] is True
    for command in (
        ["stat", "/documents/original.md"],
        ["cat", "/documents/original.md", "--structure"],
        ["grep", "original", "/documents/original.md"],
        ["browse", "/documents", "original"],
    ):
        assert main(["--workspace", str(workspace), *command]) == 0
        output = json.loads(capsys.readouterr().out)
        assert output["success"] is True


def test_register_files_rollback_preserves_preexisting_catalog_projection_and_cache(
    tmp_path, monkeypatch
):
    import openai

    install_network_fakes(monkeypatch)
    workspace = tmp_path / "workspace"
    existing_dir = tmp_path / "existing"
    batch_dir = tmp_path / "batch"
    existing_dir.mkdir()
    batch_dir.mkdir()
    existing = existing_dir / "shared.md"
    shared = batch_dir / "shared.md"
    second = batch_dir / "second.md"
    existing.write_text("alpha registration evidence", encoding="utf-8")
    shared.write_text(existing.read_text(encoding="utf-8"), encoding="utf-8")
    second.write_text("beta registration evidence", encoding="utf-8")

    filesystem = open_test_filesystem(workspace)
    existing_ref = filesystem.register_file(
        storage_uri=existing.as_uri(),
        folder_path="/documents",
        title=existing.name,
        content_type="text/markdown",
        content=existing.read_text(encoding="utf-8"),
        metadata={"year": 2023},
    )
    filesystem.metadata.register_schema(
        {"fields": {"year": {"description": "preexisting definition"}}},
        source="manual",
    )
    baseline = registration_logical_state(workspace)

    class FailingBetaOpenAI(FakeOpenAI):
        def create(self, *, model, input, dimensions):
            if any("beta" in str(text).lower() for text in input):
                raise RuntimeError("embedding unavailable for second registration")
            return super().create(model=model, input=input, dimensions=dimensions)

    monkeypatch.setattr(openai, "OpenAI", FailingBetaOpenAI)
    reopened = open_test_filesystem(workspace)
    with pytest.raises(RuntimeError, match="summary projection.*second registration"):
        reopened.register_files(
            [
                {
                    "storage_uri": source.as_uri(),
                    "folder_path": "/documents/new",
                    "title": source.name,
                    "content_type": "text/markdown",
                    "content": source.read_text(encoding="utf-8"),
                    "metadata": {"year": 2024, "batch_only": "new"},
                }
                for source in (shared, second)
            ]
        )

    assert registration_logical_state(workspace) == baseline
    strict_reopen = open_test_filesystem(workspace)
    assert strict_reopen.store.get_file(existing_ref).metadata["year"] == 2023
    structure = strict_reopen.pageindex_structure(existing_ref)
    assert structure["available"] is True
    assert structure["structure"][0]["title"] == existing.stem


def test_register_files_rolls_back_partial_pageindex_preparation(
    tmp_path, monkeypatch
):
    from pageindex import PageIndexClient

    install_network_fakes(monkeypatch)
    successful_index = PageIndexClient.index
    calls = 0

    def fail_after_second_pageindex_write(self, file_path, mode="auto"):
        nonlocal calls
        calls += 1
        document_id = successful_index(self, file_path, mode=mode)
        if calls == 2:
            raise RuntimeError("PageIndex extraction failed for second registration")
        return document_id

    monkeypatch.setattr(PageIndexClient, "index", fail_after_second_pageindex_write)
    workspace = tmp_path / "workspace"
    filesystem = open_test_filesystem(workspace)
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("alpha registration evidence", encoding="utf-8")
    second.write_text("beta registration evidence", encoding="utf-8")
    baseline = registration_logical_state(workspace)

    with pytest.raises(
        RuntimeError,
        match="PageIndex extraction failed for second registration",
    ):
        filesystem.register_files(
            [
                {
                    "storage_uri": source.as_uri(),
                    "folder_path": "/documents/new",
                    "title": source.name,
                    "content_type": "text/markdown",
                    "content": source.read_text(encoding="utf-8"),
                    "metadata": {"year": 2024},
                }
                for source in (first, second)
            ]
        )

    assert registration_logical_state(workspace) == baseline
    reopened = open_test_filesystem(workspace)
    assert reopened.store.folder_info("/")["path"] == "/"
    assert reopened.store.file_refs_for_scope() == []


def test_register_files_prepare_failure_restores_existing_owned_artifacts(
    tmp_path, monkeypatch
):
    from pageindex.filesystem.commands import PIFSCommandExecutor

    install_network_fakes(monkeypatch)
    workspace = tmp_path / "workspace"
    existing = tmp_path / "existing.md"
    replacement = tmp_path / "replacement.md"
    existing.write_text("alpha original registration evidence", encoding="utf-8")
    replacement.write_text("beta replacement registration evidence", encoding="utf-8")

    filesystem = open_test_filesystem(workspace)
    existing_ref = filesystem.register_file(
        storage_uri=existing.as_uri(),
        external_id="doc-existing",
        folder_path="/documents",
        title=existing.name,
        content_type="text/markdown",
        content=existing.read_text(encoding="utf-8"),
        metadata={"year": 2023},
    )
    existing_entry = filesystem.store.get_file(existing_ref)
    text_path = Path(existing_entry.text_artifact_path)
    raw_path = Path(existing_entry.raw_artifact_path)
    original_text = text_path.read_bytes()
    original_raw = raw_path.read_bytes()
    original_pageindex_doc_id = existing_entry.pageindex_doc_id
    baseline = registration_logical_state(workspace)

    with pytest.raises(RuntimeError, match="requires PageIndex extraction"):
        filesystem.register_files(
            [
                {
                    "storage_uri": replacement.as_uri(),
                    "external_id": "doc-existing",
                    "folder_path": "/documents/replacement",
                    "title": replacement.name,
                    "content_type": "text/markdown",
                    "content": replacement.read_text(encoding="utf-8"),
                    "metadata": {"year": 2024},
                },
                {
                    "storage_uri": "https://example.invalid/unresolvable.md",
                    "external_id": "doc-failing",
                    "folder_path": "/documents/new",
                    "title": "unresolvable.md",
                    "content_type": "text/markdown",
                    "content": "must not be registered",
                    "metadata": {"year": 2025},
                },
            ]
        )

    assert registration_logical_state(workspace) == baseline
    assert text_path.read_bytes() == original_text
    assert raw_path.read_bytes() == original_raw

    reopened = open_test_filesystem(workspace)
    reopened_entry = reopened.store.get_file(existing_ref)
    assert reopened_entry.pageindex_doc_id == original_pageindex_doc_id
    executor = PIFSCommandExecutor(reopened)
    locator = "/documents/existing.md"
    assert json.loads(executor.execute(f"stat {locator}"))["success"] is True
    cat = json.loads(executor.execute(f"cat {locator} --structure"))
    assert cat["success"] is True
    assert cat["data"]["pagination"]["available"] is True
    grep = json.loads(executor.execute(f"grep alpha {locator}"))
    assert grep["success"] is True
    assert grep["data"]["matches"]
    structure = reopened.pageindex_structure(existing_ref)
    assert structure["available"] is True
    assert structure["structure"][0]["title"] == existing.stem


@pytest.mark.parametrize("raw_path_kind", ["canonical", "custom"])
def test_register_files_rollback_follows_explicit_raw_artifact_management_policy(
    raw_path_kind, tmp_path, monkeypatch
):
    import openai
    from pageindex.filesystem.store import make_file_ref

    install_network_fakes(monkeypatch)

    class FailingSecondOpenAI(FakeOpenAI):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def create(self, *, model, input, dimensions):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("embedding unavailable for second registration")
            return super().create(model=model, input=input, dimensions=dimensions)

    monkeypatch.setattr(openai, "OpenAI", FailingSecondOpenAI)
    workspace = tmp_path / "workspace"
    filesystem = open_test_filesystem(workspace)
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("alpha registration evidence", encoding="utf-8")
    second.write_text("beta registration evidence", encoding="utf-8")
    first_external_id = f"doc-{raw_path_kind}-raw"
    first_file_ref = make_file_ref(first_external_id)
    raw_dir = workspace / "artifacts" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / (
        f"{first_file_ref}.json"
        if raw_path_kind == "canonical"
        else "custom-sentinel.json"
    )
    sentinel = f"{raw_path_kind} raw sentinel".encode()
    raw_path.write_bytes(sentinel)
    baseline = registration_logical_state(workspace)

    with pytest.raises(RuntimeError, match="summary projection.*second registration"):
        filesystem.register_files(
            [
                {
                    "storage_uri": first.as_uri(),
                    "external_id": first_external_id,
                    "folder_path": "/documents/new",
                    "title": first.name,
                    "content_type": "text/markdown",
                    "content": first.read_text(encoding="utf-8"),
                    "metadata": {"year": 2024},
                    "raw_artifact_path": str(raw_path),
                },
                {
                    "storage_uri": second.as_uri(),
                    "external_id": "doc-second-raw",
                    "folder_path": "/documents/new",
                    "title": second.name,
                    "content_type": "text/markdown",
                    "content": second.read_text(encoding="utf-8"),
                    "metadata": {"year": 2025},
                },
            ]
        )

    assert raw_path.read_bytes() == sentinel
    assert registration_logical_state(workspace) == baseline
    reopened = open_test_filesystem(workspace)
    assert reopened.store.file_refs_for_scope() == []


def test_register_file_rejects_non_json_metadata_before_side_effects(
    tmp_path, monkeypatch
):
    from pageindex import PageIndexClient

    install_network_fakes(monkeypatch)
    successful_index = PageIndexClient.index
    indexed_paths = []

    def record_index(self, file_path, mode="auto"):
        indexed_paths.append(Path(file_path).name)
        return successful_index(self, file_path, mode=mode)

    monkeypatch.setattr(PageIndexClient, "index", record_index)
    workspace = tmp_path / "workspace"
    filesystem = open_test_filesystem(workspace)
    source = tmp_path / "notes.md"
    source.write_text("alpha registration evidence", encoding="utf-8")
    baseline = registration_logical_state(workspace)

    with pytest.raises(ValueError, match="metadata must be JSON serializable"):
        filesystem.register_file(
            storage_uri=source.as_uri(),
            folder_path="/documents/new",
            title=source.name,
            content_type="text/markdown",
            content=source.read_text(encoding="utf-8"),
            metadata={"not_json": object()},
        )

    assert registration_logical_state(workspace) == baseline
    assert indexed_paths == []
    reopened = open_test_filesystem(workspace)
    assert reopened.store.folder_info("/")["path"] == "/"
    assert reopened.store.file_refs_for_scope() == []


def test_register_files_preflights_all_metadata_before_preparing_batch(
    tmp_path, monkeypatch
):
    from pageindex import PageIndexClient

    install_network_fakes(monkeypatch)
    successful_index = PageIndexClient.index
    indexed_paths = []

    def record_index(self, file_path, mode="auto"):
        indexed_paths.append(Path(file_path).name)
        return successful_index(self, file_path, mode=mode)

    monkeypatch.setattr(PageIndexClient, "index", record_index)
    workspace = tmp_path / "workspace"
    filesystem = open_test_filesystem(workspace)
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("alpha registration evidence", encoding="utf-8")
    second.write_text("beta registration evidence", encoding="utf-8")
    baseline = registration_logical_state(workspace)

    with pytest.raises(ValueError, match="metadata must be JSON serializable"):
        filesystem.register_files(
            [
                {
                    "storage_uri": first.as_uri(),
                    "folder_path": "/documents/new",
                    "title": first.name,
                    "content_type": "text/markdown",
                    "content": first.read_text(encoding="utf-8"),
                    "metadata": {"year": 2024},
                },
                {
                    "storage_uri": second.as_uri(),
                    "folder_path": "/documents/new",
                    "title": second.name,
                    "content_type": "text/markdown",
                    "content": second.read_text(encoding="utf-8"),
                    "metadata": {"not_json": object()},
                },
            ]
        )

    assert registration_logical_state(workspace) == baseline
    assert indexed_paths == []
    reopened = open_test_filesystem(workspace)
    assert reopened.store.folder_info("/")["path"] == "/"
    assert reopened.store.file_refs_for_scope() == []


def test_register_file_rolls_back_partial_projection_creation_and_reopens(
    tmp_path, monkeypatch
):
    install_network_fakes(monkeypatch)
    # SQLite's Unix VFS accepts summary.sqlite at this path length but rejects
    # the eight-character-longer embedding_cache.sqlite path.
    workspace = nested_path_with_length(tmp_path / "long-workspace", 453)
    filesystem = open_test_filesystem(workspace)
    source = tmp_path / "notes.md"
    source.write_text("alpha registration evidence", encoding="utf-8")

    with pytest.raises(sqlite3.OperationalError, match="unable to open database file"):
        filesystem.register_file(
            storage_uri=source.as_uri(),
            folder_path="/documents",
            title=source.name,
            content_type="text/markdown",
            content=source.read_text(encoding="utf-8"),
        )

    projection_dir = workspace / "artifacts" / "projection_indexes"
    assert not (projection_dir / "summary.sqlite").exists()
    assert not (projection_dir / "embedding_cache.sqlite").exists()
    assert catalog_file_count(workspace) == 0
    assert not list((workspace / "artifacts" / "text").glob("*"))
    assert not list((workspace / "artifacts" / "raw").glob("*"))

    reopened = open_test_filesystem(workspace)
    assert reopened.store.folder_info("/")["path"] == "/"


def test_cli_add_rejects_duplicate_target_without_changing_owned_document(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "notes.md"
    second = second_dir / "notes.md"
    first.write_text("first alpha evidence", encoding="utf-8")
    second.write_text("second beta evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    command = ["--workspace", str(workspace), "add"]

    assert main([*command, str(first), "/documents"]) == 0
    capsys.readouterr()
    assert main([*command, str(second), "/documents"]) == 1

    assert "already exists" in capsys.readouterr().err
    assert catalog_file_count(workspace) == 1
    owned = list((workspace / "artifacts" / "uploads").glob("*/notes.md"))
    assert len(owned) == 1
    assert owned[0].read_text(encoding="utf-8") == "first alpha evidence"


def test_physical_browse_preserves_collision_aware_tree_locators(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    for filename, content in (
        ("alpha.md", "alpha evidence"),
        ("beta.md", "beta evidence"),
    ):
        source = tmp_path / filename
        source.write_text(content, encoding="utf-8")
        assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
        capsys.readouterr()
    with sqlite3.connect(workspace / "filesystem.sqlite") as connection:
        connection.execute(
            "UPDATE file_folders SET metadata_json = ?",
            (json.dumps({"display_name": "same.md"}),),
        )

    assert main(["--workspace", str(workspace), "tree", "/documents", "-L", "1"]) == 0
    tree_files = json.loads(capsys.readouterr().out)["data"]["tree"]["files"]
    tree_paths = {row["path"] for row in tree_files}
    assert tree_paths == {
        "/documents/same.md~1",
        "/documents/same.md~2",
    }

    assert main(["--workspace", str(workspace), "browse", "/documents", "alpha"]) == 0
    browse_documents = json.loads(capsys.readouterr().out)["data"]["documents"]
    browse_paths = [row["path"] for row in browse_documents]

    assert len(browse_documents) == 2
    assert len(set(browse_paths)) == 2
    assert set(browse_paths) == tree_paths
    for path in browse_paths:
        assert main(["--workspace", str(workspace), "stat", path]) == 0
        stat = json.loads(capsys.readouterr().out)["data"]["document"]
        assert stat["path"] == path


def test_browse_recursively_returns_one_global_ranked_page_of_ten(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    for index in range(12):
        source = tmp_path / f"doc-{index:02d}.md"
        relevance = "alpha" if index in {1, 10} else "beta"
        source.write_text(f"{relevance} evidence {index}", encoding="utf-8")
        folder = "/documents/a" if index % 2 == 0 else "/documents/b"
        assert main(["--workspace", str(workspace), "add", str(source), folder]) == 0
        capsys.readouterr()

    command = [
        "--workspace",
        str(workspace),
        "browse",
        "/documents",
        "alpha",
        "--recursive",
    ]
    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)["data"]
    assert len(first["documents"]) == 10
    assert first["pagination"] == {
        "page": 1,
        "page_size": 10,
        "has_more": True,
        "next_page": 2,
    }
    assert {row["title"] for row in first["documents"][:2]} == {
        "doc-01.md",
        "doc-10.md",
    }
    assert {row["folder_path"] for row in first["documents"]} == {
        "/documents/a",
        "/documents/b",
    }

    assert main([*command, "--page", "2"]) == 0
    second = json.loads(capsys.readouterr().out)["data"]
    assert len(second["documents"]) == 2
    assert second["pagination"] == {
        "page": 2,
        "page_size": 10,
        "has_more": False,
        "next_page": None,
    }
    assert not {
        row["path"] for row in first["documents"]
    }.intersection(row["path"] for row in second["documents"])


def test_browse_respects_physical_and_virtual_scope(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    records = [
        ("current.md", "alpha current", "/documents/current", 2024),
        ("old.md", "alpha old", "/documents/archive", 2023),
    ]
    for filename, content, folder, year in records:
        source = tmp_path / filename
        source.write_text(content, encoding="utf-8")
        assert main(["--workspace", str(workspace), "add", str(source), folder]) == 0
        capsys.readouterr()
        assert main(
            [
                "--workspace",
                str(workspace),
                "setmeta",
                f"{folder}/{filename}",
                json.dumps({"year": year}),
            ]
        ) == 0
        capsys.readouterr()

    assert main(
        [
            "--workspace",
            str(workspace),
            "browse",
            "/documents/current",
            "alpha",
        ]
    ) == 0
    physical = json.loads(capsys.readouterr().out)["data"]["documents"]
    assert [row["title"] for row in physical] == ["current.md"]

    assert main(
        [
            "--workspace",
            str(workspace),
            "browse",
            "/documents/@year/2024",
            "alpha",
        ]
    ) == 0
    virtual = json.loads(capsys.readouterr().out)["data"]["documents"]
    assert [row["title"] for row in virtual] == ["current.md"]
    assert virtual[0]["path"] == "/documents/@year/2024/current.md"


def test_browse_requires_query_and_an_existing_summary_projection(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    write_embedding_config(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "tree", "/", "-L", "1"]) == 0
    capsys.readouterr()

    assert main(["--workspace", str(workspace), "browse", "/documents"]) == 2
    missing_query = json.loads(capsys.readouterr().out)
    assert "requires a query" in missing_query["error"]["message"]

    assert main(["--workspace", str(workspace), "browse", "/", "alpha"]) == 2
    missing_projection = json.loads(capsys.readouterr().out)
    assert "Summary Projection is not available" in missing_projection["error"]["message"]


def test_browse_rejects_incompatible_embedding_identity_without_mutation(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    config = write_embedding_config(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    source = tmp_path / "notes.md"
    source.write_text("alpha evidence", encoding="utf-8")
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()
    databases = [
        workspace / "filesystem.sqlite",
        workspace / "artifacts" / "projection_indexes" / "summary.sqlite",
        workspace / "artifacts" / "projection_indexes" / "embedding_cache.sqlite",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in databases}
    config.write_text(
        json.dumps(
            {
                "embedding_base_url": "https://example.invalid/v1",
                "embedding_model": "different-model",
                "embedding_dimensions": 3,
            }
        ),
        encoding="utf-8",
    )

    assert main(["--workspace", str(workspace), "browse", "/documents", "alpha"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert "Incompatible PIFS Summary Embedding Profile" in payload["error"]["message"]
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in databases
    } == before


def test_agent_prompts_describe_only_the_retained_command_surface():
    from pageindex.filesystem.agent import (
        AGENT_SYSTEM_PROMPT,
        AGENT_TOOL_POLICY,
        BASH_TOOL_DESCRIPTION,
    )

    prompts = "\n".join(
        [AGENT_SYSTEM_PROMPT, BASH_TOOL_DESCRIPTION, AGENT_TOOL_POLICY]
    )
    for command in ("tree", "browse", "stat", "cat", "grep"):
        assert command in prompts
    for retired_contract in (
        "ls as an alias",
        "file_ref",
        "--space",
        "Do not use find",
        "recursive grep",
    ):
        assert retired_contract not in prompts


def test_cat_page_reads_are_bounded_and_grep_is_single_document(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    install_network_fakes(monkeypatch)
    write_embedding_config(tmp_path, monkeypatch)
    source = tmp_path / "notes.md"
    source.write_text("line one\nalpha evidence", encoding="utf-8")
    workspace = tmp_path / "workspace"
    assert main(["--workspace", str(workspace), "add", str(source), "/documents"]) == 0
    capsys.readouterr()

    assert main(
        [
            "--workspace",
            str(workspace),
            "cat",
            "/documents/notes.md",
            "--page",
            "1",
        ]
    ) == 0
    page = json.loads(capsys.readouterr().out)["data"]
    assert page["requested_pages"] == "1"
    assert page["content"]["text"] == "line one\nalpha evidence"

    assert main(
        [
            "--workspace",
            str(workspace),
            "cat",
            "/documents/notes.md",
            "--page",
            "1-6",
        ]
    ) == 2
    too_wide = json.loads(capsys.readouterr().out)
    assert "at most 5 pages" in too_wide["error"]["message"]

    assert main(
        ["--workspace", str(workspace), "grep", "alpha", "/documents"]
    ) == 2
    folder_grep = json.loads(capsys.readouterr().out)
    assert "resolved file locator" in folder_grep["error"]["message"]


@pytest.mark.parametrize(
    "command",
    [
        ["ls", "/"],
        ["find", "/"],
        ["browse", "/", "alpha", "--space", "entity"],
        ["browse", "/", "alpha", "--limit", "2"],
        ["stat", "/", "--schema"],
        ["stat", "/", "--field", "year"],
        ["stat", "/one", "/two"],
        ["cat", "/document"],
        ["cat", "/document", "--all"],
        ["cat", "/document", "--range", "1-2"],
        ["grep", "alpha", "/document", "--recursive"],
    ],
)
def test_retired_command_forms_return_structured_errors(command, tmp_path, capsys):
    from pageindex.filesystem.cli import main

    status = main(["--workspace", str(tmp_path / "workspace"), *command])

    payload = json.loads(capsys.readouterr().out)
    assert status == 2
    assert payload["success"] is False
    assert payload["error"]["code"] == "invalid_command"


def test_set_workspace_preserves_embedding_api_key_without_exposing_or_persisting_it_elsewhere(
    tmp_path, monkeypatch, capsys
):
    from pageindex.filesystem.cli import main

    config = tmp_path / "pifs.json"
    config.write_text(
        json.dumps(
            {
                "embedding_base_url": "https://example.invalid/v1",
                "embedding_model": "test-embedding",
                "embedding_dimensions": 3,
                "embedding_timeout": 12,
                "embedding_api_key": "config-secret",
                "embedding_provider": "legacy-provider",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PIFS_CONFIG_FILE", str(config))
    workspace = tmp_path / "workspace"

    assert main(["set", "workspace", str(workspace)]) == 0
    output = capsys.readouterr().out

    assert json.loads(config.read_text(encoding="utf-8")) == {
        "embedding_api_key": "config-secret",
        "embedding_base_url": "https://example.invalid/v1",
        "embedding_dimensions": "3",
        "embedding_model": "test-embedding",
        "embedding_timeout": "12",
        "workspace": str(workspace),
    }
    assert "config-secret" not in output


def test_public_example_is_a_short_supported_cli_walkthrough():
    example = Path(__file__).parents[1] / "examples" / "pifs_demo.py"
    source = example.read_text(encoding="utf-8")

    assert len(source.splitlines()) <= 60
    for command in ("add", "tree", "browse", "stat", "cat", "grep"):
        assert f'"{command}"' in source
    for retired_surface in (
        "embedding_provider",
        "metadata_provider",
        "file_ref",
        "SummaryProjectionIndexer",
        "shutil.rmtree",
        '"ls"',
    ):
        assert retired_surface not in source
