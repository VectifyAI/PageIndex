from __future__ import annotations

import sqlite3
from pathlib import Path


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _inconsistent_workspace_error(detail: str) -> RuntimeError:
    return RuntimeError(
        f"Inconsistent PIFS workspace: {detail}; migrate this workspace with "
        "pifs-data/scripts/migrate_pifs_workspace.py before opening it."
    )


def validate_catalog_root(catalog_path: str | Path) -> None:
    try:
        with _readonly_connection(Path(catalog_path)) as catalog:
            root = catalog.execute(
                "SELECT folder_id, parent_id, name FROM folders WHERE path = '/'"
            ).fetchone()
    except sqlite3.Error as exc:
        raise _inconsistent_workspace_error(
            "the catalog canonical root folder could not be read"
        ) from exc
    if (
        root is None
        or root["folder_id"] != "folder_root"
        or root["parent_id"] is not None
        or root["name"] != "/"
    ):
        raise _inconsistent_workspace_error(
            "the catalog is missing its canonical root folder"
        )


def validate_workspace_consistency(
    catalog_path: str | Path,
    summary_path: str | Path,
) -> None:
    catalog_path = Path(catalog_path)
    validate_catalog_root(catalog_path)
    try:
        with _readonly_connection(catalog_path) as catalog:
            active_file_refs = {
                str(row[0])
                for row in catalog.execute(
                    "SELECT file_ref FROM files WHERE deleted_at IS NULL"
                )
            }
        with _readonly_connection(Path(summary_path)) as summary:
            projected_file_refs = {
                str(row[0])
                for row in summary.execute(
                    "SELECT file_ref FROM semantic_index_docs"
                )
            }
    except sqlite3.Error as exc:
        raise _inconsistent_workspace_error(
            "catalog and Summary Projection references could not be read"
        ) from exc
    orphaned = sorted(projected_file_refs - active_file_refs)
    if orphaned:
        shown = ", ".join(orphaned[:5])
        raise _inconsistent_workspace_error(
            "Summary Projection file_ref values do not reference active catalog files: "
            f"{shown}"
        )
    missing = sorted(active_file_refs - projected_file_refs)
    if missing:
        shown = ", ".join(missing[:5])
        raise _inconsistent_workspace_error(
            "active catalog file_ref values are missing from the Summary Projection: "
            f"{shown}"
        )


def validate_catalog_without_projection(catalog_path: str | Path) -> None:
    catalog_path = Path(catalog_path)
    validate_catalog_root(catalog_path)
    try:
        with _readonly_connection(catalog_path) as catalog:
            active_file_refs = sorted(
                str(row[0])
                for row in catalog.execute(
                    "SELECT file_ref FROM files WHERE deleted_at IS NULL"
                )
            )
    except sqlite3.Error as exc:
        raise _inconsistent_workspace_error(
            "active catalog references could not be read"
        ) from exc
    if active_file_refs:
        shown = ", ".join(active_file_refs[:5])
        raise _inconsistent_workspace_error(
            "active catalog files require a complete Summary Projection: "
            f"{shown}"
        )
