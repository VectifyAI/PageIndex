from __future__ import annotations

import json
import re
import shlex
from typing import Any

from .core import PageIndexFileSystem


class PIFSCommandError(ValueError):
    pass


class PIFSCommandExecutor:
    COMMAND_NAMES = {"tree", "browse", "stat", "cat", "grep"}
    FORBIDDEN_SUBSTRINGS = (";", "`", "$(", "||", "&&", "\n", "\r")
    FORBIDDEN_TOKENS = {"|", ">", "<", ">>", "<<", "&"}
    BROWSE_PAGE_SIZE = 10
    TREE_VALUE_PAGE_SIZE = 50
    MAX_TREE_FOLDERS = 200
    MAX_GREP_MATCHES = 20
    MAX_PAGE_SPAN = 5

    def __init__(self, filesystem: PageIndexFileSystem):
        self.filesystem = filesystem

    def allowed_commands(self) -> set[str]:
        return set(self.COMMAND_NAMES)

    def describe_available_command_surfaces(self) -> str:
        return "\n".join(
            [
                "Available PIFS BashLike commands:",
                "- tree <scope> [-L depth] [--page N]: folder and metadata-scope orientation",
                '- browse <scope> "<query>" [--page N] [--where JSON] [-R]: summary-ranked document discovery',
                "- stat <file>: single document identity, status, and metadata",
                "- cat <file> --structure | --page N[-M]: structure-first document reads",
                "- grep <query> <file>: single-document lexical evidence fallback",
            ]
        )

    def execute(self, command: str) -> str:
        try:
            data, next_steps = self._execute(command)
            return self._success(data, next_steps=next_steps)
        except PIFSCommandError as exc:
            return self._error(str(exc))
        except (KeyError, ValueError) as exc:
            return self._error(self._clean_error_message(exc))
        except Exception as exc:
            return self._error(self._clean_error_message(exc))

    def _execute(self, command: str) -> tuple[dict[str, Any], list[str]]:
        self._validate_raw_command(command)
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise PIFSCommandError(f"Invalid command syntax: {exc}") from exc
        if not tokens:
            raise PIFSCommandError("Empty command")
        self._validate_tokens(tokens)
        if "--json" in tokens:
            raise PIFSCommandError("--json is removed; command output is always JSON")
        name, args = tokens[0], tokens[1:]
        if name not in self.allowed_commands():
            raise PIFSCommandError(f"Unsupported command: {name}")
        return getattr(self, f"_cmd_{name}")(args)

    def _cmd_tree(self, args: list[str]) -> tuple[dict[str, Any], list[str]]:
        path = "/"
        depth = 10
        page = 1
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-L":
                i, value = self._option_value(args, i, "tree -L")
                depth = self._parse_positive_int(value, "tree -L")
            elif arg == "--page":
                i, value = self._option_value(args, i, "tree --page")
                page = self._parse_positive_int(value, "tree --page")
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported tree option: {arg}")
            else:
                path = arg
            i += 1
        scope = self.filesystem.resolve_query_scope(path)
        if scope.metadata_axis is not None:
            rows, has_more = self.filesystem.scope_metadata_values(
                scope,
                page=page,
                page_size=self.TREE_VALUE_PAGE_SIZE,
            )
            root = self._scope_root(scope, file_count=self.filesystem.scope_file_count(scope))
            root["folders"] = [
                {
                    "path": self._metadata_value_path(
                        scope.path.rsplit("/", 1)[0] or "/",
                        scope.metadata_axis,
                        row["value"],
                    ),
                    "name": str(row["value"]),
                    "type": "metadata_value",
                    "value": row["value"],
                    "file_count": row["file_count"],
                    "folders": [],
                }
                for row in rows
            ]
            root["children_count"] = len(root["folders"])
            data = {
                "tree": root,
                "total_folders": len(root["folders"]),
                "depth": depth,
                "truncated": has_more,
                "pagination": {
                    "page": page,
                    "page_size": self.TREE_VALUE_PAGE_SIZE,
                    "has_more": has_more,
                    "next_page": page + 1 if has_more else None,
                },
            }
            next_steps = [
                f'browse {shlex.quote(scope.path)}/<value> "<query>"'
            ]
            return data, next_steps
        page_size = self.TREE_VALUE_PAGE_SIZE
        needed = page * page_size + 1
        folders = self.filesystem.scope_folders(
            scope,
            max_depth=depth,
            limit=max(self.MAX_TREE_FOLDERS, needed),
        )
        files = [
            self._tree_file(row)
            for row in self.filesystem.scope_files(scope, limit=needed)
        ]
        axes = self.filesystem.scope_metadata_axes(scope)
        tree, has_more = self._folder_tree(
            scope,
            folders,
            files,
            axes,
            page=page,
            page_size=page_size,
        )
        data = {
            "tree": tree,
            "total_folders": len(folders) + len(axes),
            "depth": depth,
            "truncated": has_more or len(folders) >= self.MAX_TREE_FOLDERS,
        }
        if has_more or page > 1:
            data["pagination"] = {
                "page": page,
                "page_size": page_size,
                "has_more": has_more,
                "next_page": page + 1 if has_more else None,
            }
        next_steps = [f'browse {shlex.quote(scope.path)} "<query>"']
        return data, next_steps

    def _cmd_browse(self, args: list[str]) -> tuple[dict[str, Any], list[str]]:
        recursive = False
        where = None
        page = 1
        positionals: list[str] = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg in {"-R", "--recursive"}:
                recursive = True
            elif arg == "--where":
                i, where = self._option_value(args, i, "browse --where")
            elif arg == "--page":
                i, value = self._option_value(args, i, "browse --page")
                page = self._parse_positive_int(value, "browse --page")
            elif arg == "--space":
                raise PIFSCommandError("browse --space is removed; browse uses summary retrieval only")
            elif arg in {"--limit", "--offset", "--query"}:
                raise PIFSCommandError(f"browse does not support {arg}; use --page N")
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported browse option: {arg}")
            else:
                positionals.append(arg)
            i += 1
        if len(positionals) < 2 or not str(positionals[1]).strip():
            raise PIFSCommandError('browse requires a query: browse <folder> "<query>"')
        if len(positionals) > 2:
            raise PIFSCommandError('browse accepts a folder and one quoted query')
        path, query = positionals
        if not str(path).startswith("/"):
            raise PIFSCommandError("browse target must be a PIFS folder path like /documents")
        scope = self.filesystem.resolve_query_scope(path)
        if scope.metadata_axis is not None:
            raise PIFSCommandError(
                "Metadata axis paths require @field/value; run tree <scope>/@field to inspect values."
            )
        merged_filter = self.filesystem.merge_scope_filter(scope, where)
        effective_recursive = recursive or bool(scope.metadata_filter)
        payload = self.filesystem.browse_semantic_files(
            scope.path,
            query,
            recursive=effective_recursive,
            page=page,
            metadata_filter=where,
        )
        documents = [self._document_hit(row) for row in payload.get("data", [])]
        next_steps = []
        if payload.get("has_more"):
            next_steps.append(self._browse_command(path, query, recursive=recursive, where=where, page=page + 1))
        scope_payload = {
            "folder": scope.folder_path,
            "recursive": effective_recursive,
            "query": query,
            "where": self._json_filter(where),
            "retrieval": "summary",
        }
        if scope.path != scope.folder_path or merged_filter is not None:
            scope_payload.update(
                {
                    "path": scope.path,
                    "folder_path": scope.folder_path,
                    "metadata_filter": merged_filter,
                }
            )
        return {
            "documents": documents,
            "pagination": {
                "page": page,
                "page_size": self.BROWSE_PAGE_SIZE,
                "has_more": bool(payload.get("has_more")),
                "next_page": page + 1 if payload.get("has_more") else None,
            },
            "scope": scope_payload,
        }, next_steps

    def _cmd_stat(self, args: list[str]) -> tuple[dict[str, Any], list[str]]:
        if any(arg == "--schema" for arg in args):
            raise PIFSCommandError("stat --schema is removed")
        if any(arg == "--field" for arg in args):
            raise PIFSCommandError("stat --field is removed")
        if any(arg.startswith("-") for arg in args):
            raise PIFSCommandError("stat accepts only one document target")
        if len(args) != 1:
            raise PIFSCommandError("stat accepts exactly one document target")
        target = args[0]
        return {"document": self._document_stat(target)}, []

    def _cmd_cat(self, args: list[str]) -> tuple[dict[str, Any], list[str]]:
        if not args:
            raise PIFSCommandError("cat requires a document target")
        target = args[0]
        if target.startswith("-"):
            raise PIFSCommandError("cat syntax is target-first: cat <file> --structure or cat <file> --page N[-M]")
        if "--all" in args or "--range" in args:
            raise PIFSCommandError("cat --all and cat --range are removed; use --structure or --page")
        if len(args) == 2 and args[1] == "--structure":
            payload = self.filesystem.pageindex_structure(target)
            return {
                "document": self._document_from_structural_payload(payload, target),
                "structure": payload.get("structure") if payload.get("available", True) else None,
                "pagination": {"available": bool(payload.get("available", True))},
            }, self._page_next_steps(target, payload)
        if len(args) == 3 and args[1] == "--page":
            pages = args[2]
            if not re.fullmatch(r"\d+(?:-\d+)?", pages):
                raise PIFSCommandError("cat --page requires one page selector like 31 or 31-33")
            start, end = self._parse_numeric_range(pages, "cat --page")
            if end - start + 1 > self.MAX_PAGE_SPAN:
                raise PIFSCommandError(f"cat --page supports at most {self.MAX_PAGE_SPAN} pages")
            payload = self.filesystem.pageindex_pages(target, pages)
            return {
                "document": self._document_from_structural_payload(payload, target),
                "requested_pages": pages,
                "returned_pages": payload.get("data", []),
                "content": {
                    "text": payload.get("text", ""),
                    "available": bool(payload.get("available", True)),
                },
            }, []
        raise PIFSCommandError("cat requires either --structure or --page N[-M]")

    def _cmd_grep(self, args: list[str]) -> tuple[dict[str, Any], list[str]]:
        if any(arg in {"-R", "-r", "--recursive", "--where"} for arg in args):
            raise PIFSCommandError("grep is single-document only: grep <query> <file>")
        if any(arg.startswith("-") for arg in args):
            raise PIFSCommandError("grep accepts no options")
        if len(args) != 2:
            raise PIFSCommandError("grep requires a query and one document target")
        query, target = args
        if self._is_folder(target):
            raise PIFSCommandError("grep requires a resolved file locator, not a folder")
        return {
            "document": self._document_stat(target),
            "matches": self._grep_file_matches(target, query),
        }, []

    def _folder_tree(
        self,
        scope: Any,
        folders: list[dict[str, Any]],
        files: list[dict[str, Any]],
        axes: list[dict[str, Any]],
        *,
        page: int,
        page_size: int,
    ) -> tuple[dict[str, Any], bool]:
        root_key = self._normalize_folder_path(scope.folder_path)
        root = self._scope_root(scope, file_count=self.filesystem.scope_file_count(scope))
        nodes = {
            root_key: root
        }
        for folder in sorted(folders, key=lambda item: item["path"]):
            physical_path = self._normalize_folder_path(folder["path"])
            nodes[physical_path] = {
                "path": self._scoped_folder_path(scope, physical_path),
                "name": folder.get("name") or physical_path.rsplit("/", 1)[-1],
                "type": "folder",
                "file_count": folder.get("matched_files", 0) or folder.get("file_count", 0),
                "children_count": folder.get("children_count", 0),
                "folders": [],
            }
        for folder_path, node in sorted(nodes.items(), key=lambda item: item[0].count("/")):
            if folder_path == root_key:
                continue
            parent = folder_path.rsplit("/", 1)[0] or "/"
            nodes.get(parent, nodes[root_key])["folders"].append(node)
        axis_nodes = [
            {
                "path": self._join_scope_path(
                    scope.path,
                    f"@{self.filesystem.encode_scope_segment(axis['name'])}",
                ),
                "name": f"@{axis['name']}",
                "type": "metadata_axis",
                "value_count": axis["value_count"],
                "folders": [],
            }
            for axis in axes
        ]
        child_items = (
            [("folder", node) for node in nodes[root_key]["folders"]]
            + [("file", file) for file in files]
            + [("folder", node) for node in axis_nodes]
        )
        offset = (page - 1) * page_size
        page_items = child_items[offset : offset + page_size]
        nodes[root_key]["folders"] = [node for kind, node in page_items if kind == "folder"]
        nodes[root_key]["files"] = [node for kind, node in page_items if kind == "file"]
        nodes[root_key]["children_count"] = len(child_items)
        return nodes[root_key], len(child_items) > offset + page_size

    def _document_hit(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": row.get("path"),
            "document_id": row.get("external_id"),
            "title": row.get("title"),
            "status": row.get("pageindex_tree_status"),
            "rank": row.get("rank"),
            "similarity": row.get("similarity"),
            "summary": row.get("summary", ""),
            "metadata": row.get("metadata", {}),
            "folder_path": row.get("folder_path"),
            "folder_paths": row.get("folder_paths", []),
        }

    @staticmethod
    def _tree_file(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": row["path"],
            "document_id": row.get("external_id"),
            "title": row["title"],
            "status": row["pageindex_tree_status"],
            "type": "file",
            "metadata": row.get("metadata", {}),
        }

    def _document_stat(self, target: str) -> dict[str, Any]:
        file_ref = self.filesystem._resolve_target(target)
        entry = self.filesystem.store.get_file(file_ref)
        folder_paths = [
            folder["path"]
            for folder in self.filesystem.store.folder_memberships(file_ref)
        ]
        folder_path = self.filesystem._preferred_folder_path(
            folder_paths,
            entry.folder_path,
            entry.folder_path,
        )
        path = (
            target
            if str(target).startswith("/")
            else self.filesystem._stable_file_locator(
                file_ref,
                entry,
                folder_path=folder_path,
            )
        )
        return {
            "path": path,
            "document_id": entry.external_id,
            "title": entry.title,
            "status": entry.pageindex_tree_status,
            "content_type": entry.content_type,
            "metadata": entry.metadata,
            "metadata_status": entry.metadata_status,
            "folder_paths": folder_paths,
        }

    def _document_from_structural_payload(self, payload: dict[str, Any], target: str) -> dict[str, Any]:
        document = self._document_stat(target)
        document["available"] = bool(payload.get("available", True))
        if payload.get("message"):
            document["message"] = payload.get("message")
        return document

    def _page_next_steps(self, target: str, payload: dict[str, Any]) -> list[str]:
        if not payload.get("available", True):
            return []
        return [f"cat {shlex.quote(target)} --page <N[-M]>"]

    def _grep_file_matches(self, target: str, query: str) -> list[dict[str, Any]]:
        file_ref = self.filesystem._resolve_target(target)
        matches = []
        for line_number, line in enumerate(self.filesystem.store.read_text(file_ref).splitlines(), 1):
            if self._line_matches(line, query):
                matches.append({"line": line_number, "text": self._compact_text(line, max_chars=220)})
                if len(matches) >= self.MAX_GREP_MATCHES:
                    break
        return matches

    def _is_folder(self, path: str) -> bool:
        try:
            self.filesystem.folder_info(path)
            return True
        except KeyError:
            return False

    def _browse_command(
        self,
        path: str,
        query: str,
        *,
        recursive: bool,
        where: str | None,
        page: int,
    ) -> str:
        parts = ["browse"]
        if recursive:
            parts.append("-R")
        parts.extend([shlex.quote(self._normalize_folder_path(path)), shlex.quote(query)])
        if where is not None:
            parts.extend(["--where", shlex.quote(where)])
        parts.extend(["--page", str(page)])
        return " ".join(parts)

    @staticmethod
    def _json_filter(value: str | None) -> Any:
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _line_matches(line: str, query: str) -> bool:
        haystack = line.lower()
        needle = query.lower().strip()
        if needle and needle in haystack:
            return True
        terms = [term for term in re.findall(r"[A-Za-z0-9_]+", needle) if term]
        return bool(terms) and all(term in haystack for term in terms)

    @staticmethod
    def _compact_text(text: str, *, max_chars: int) -> str:
        collapsed = re.sub(r"\s+", " ", text or "").strip()
        if len(collapsed) <= max_chars:
            return collapsed
        return collapsed[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _parse_numeric_range(value: str, label: str) -> tuple[int, int]:
        try:
            if "-" in value:
                left, right = value.split("-", 1)
                start, end = int(left), int(right)
            else:
                start = end = int(value)
        except ValueError as exc:
            raise PIFSCommandError(f"{label} requires a numeric range") from exc
        if start < 1 or end < start:
            raise PIFSCommandError(f"Invalid {label} range: {value}")
        return start, end

    @staticmethod
    def _option_value(args: list[str], index: int, label: str) -> tuple[int, str]:
        value_index = index + 1
        if value_index >= len(args):
            raise PIFSCommandError(f"{label} requires a value")
        return value_index, args[value_index]

    @staticmethod
    def _parse_positive_int(value: str, label: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise PIFSCommandError(f"{label} must be an integer") from exc
        if parsed < 1:
            raise PIFSCommandError(f"{label} must be at least 1")
        return parsed

    @staticmethod
    def _normalize_folder_path(path: str) -> str:
        value = str(path or "/").strip()
        if not value or value == "/":
            return "/"
        return "/" + value.strip("/")

    def _scope_root(self, scope: Any, *, file_count: int) -> dict[str, Any]:
        if getattr(scope, "metadata_axis", None):
            name = f"@{scope.metadata_axis}"
            node_type = "metadata_axis"
        elif getattr(scope, "metadata_filter", None):
            name = str(next(reversed(scope.metadata_filter.values())))
            node_type = "metadata_value"
        else:
            info = self.filesystem.folder_info(scope.folder_path)
            name = info.get("name") or ("/" if scope.folder_path == "/" else scope.folder_path.rsplit("/", 1)[-1])
            node_type = "folder"
        return {
            "path": scope.path,
            "name": name,
            "type": node_type,
            "file_count": file_count,
            "children_count": 0,
            "folders": [],
        }

    def _scoped_folder_path(self, scope: Any, folder_path: str) -> str:
        path = self._normalize_folder_path(folder_path)
        for field, value in getattr(scope, "metadata_filter", {}).items():
            path = self._metadata_value_path(path, field, value)
        return path

    def _metadata_value_path(self, base: str, field: object, value: object) -> str:
        axis_path = self._join_scope_path(
            base,
            f"@{self.filesystem.encode_scope_segment(field)}",
        )
        return self._join_scope_path(
            axis_path,
            self.filesystem.encode_scope_segment(value),
        )

    @staticmethod
    def _join_scope_path(base: str, segment: str) -> str:
        base_path = str(base or "/").rstrip("/")
        if not base_path:
            base_path = "/"
        if base_path == "/":
            return f"/{segment}"
        return f"{base_path}/{segment}"

    @staticmethod
    def _success(data: dict[str, Any], *, next_steps: list[str] | None = None) -> str:
        return json.dumps(
            {"success": True, "data": data, "next_steps": next_steps or []},
            ensure_ascii=False,
        )

    @staticmethod
    def _error(message: str) -> str:
        return json.dumps(
            {
                "success": False,
                "error": {"code": "invalid_command", "message": message},
                "next_steps": [],
            },
            ensure_ascii=False,
        )

    @classmethod
    def _validate_raw_command(cls, command: str) -> None:
        if not command.strip():
            raise PIFSCommandError("Empty command")
        if any(token in command for token in cls.FORBIDDEN_SUBSTRINGS):
            raise PIFSCommandError("Only PageIndex FileSystem commands are allowed")

    @classmethod
    def _validate_tokens(cls, tokens: list[str]) -> None:
        if any(token in cls.FORBIDDEN_TOKENS for token in tokens):
            raise PIFSCommandError("Only PageIndex FileSystem commands are allowed")

    @staticmethod
    def _clean_error_message(exc: BaseException) -> str:
        message = str(exc)
        if isinstance(exc, KeyError) and len(exc.args) == 1:
            message = str(exc.args[0])
        return message or exc.__class__.__name__
