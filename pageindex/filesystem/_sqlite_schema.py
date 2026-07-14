from __future__ import annotations

import sqlite3
from typing import Any


def regular_table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def sqlite_schema_signature(
    connection: sqlite3.Connection,
    tables: set[str],
) -> dict[str, Any]:
    columns = {
        table: tuple(
            tuple(row)
            for row in connection.execute(f'PRAGMA table_xinfo("{table}")')
        )
        for table in sorted(tables)
    }
    indexes: dict[str, tuple[tuple[Any, ...], ...]] = {}
    foreign_keys: dict[str, tuple[tuple[Any, ...], ...]] = {}
    for table in sorted(tables):
        table_indexes = []
        for row in connection.execute(f'PRAGMA index_list("{table}")'):
            name = str(row[1])
            origin = str(row[3])
            index_columns = tuple(
                str(column[2])
                for column in connection.execute(f'PRAGMA index_info("{name}")')
            )
            table_indexes.append(
                (
                    name if origin == "c" else None,
                    int(row[2]),
                    origin,
                    int(row[4]),
                    index_columns,
                )
            )
        indexes[table] = tuple(sorted(table_indexes, key=repr))
        foreign_keys[table] = tuple(
            sorted(
                (
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                    str(row[5]),
                    str(row[6]),
                    str(row[7]),
                )
                for row in connection.execute(f'PRAGMA foreign_key_list("{table}")')
            )
        )
    return {
        "tables": tables,
        "columns": columns,
        "indexes": indexes,
        "foreign_keys": foreign_keys,
    }


def normalized_table_sql(connection: sqlite3.Connection, table: str) -> str:
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return " ".join(str(row[0] if row else "").lower().split())
