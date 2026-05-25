from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .core import SEMANTIC_GREP_CHANNELS, SEMANTIC_RETRIEVAL_CHANNELS, PageIndexFileSystem


class PIFSCommandError(ValueError):
    pass


class PIFSCommandExecutor:
    FORBIDDEN_SUBSTRINGS = (";", "`", "$(", "||", "\n", "\r")
    FORBIDDEN_TOKENS = {"|", ">", "<", ">>", "<<", "&"}
    BASE_ALLOWED_COMMANDS = {
        "ls",
        "tree",
        "find",
        "grep",
        "cat",
        "stat",
        "head",
        "tail",
        "sed",
    }
    SEMANTIC_CHANNEL_COMMANDS = {
        "summary": "search-summary",
        "entity": "search-entity",
        "relation": "search-relation",
    }
    ALLOWED_COMMANDS = (
        BASE_ALLOWED_COMMANDS
        | {"semantic-grep"}
        | set(SEMANTIC_CHANNEL_COMMANDS.values())
    )
    ALLOWED_PIPE_FILTERS = {"head", "tail", "grep", "sed"}
    COMMAND_METHODS = {
        "search-summary": "_cmd_search_summary",
        "search-entity": "_cmd_search_entity",
        "search-relation": "_cmd_search_relation",
        "semantic-grep": "_cmd_semantic_grep",
    }
    MAX_TREE_DEPTH = 4
    MAX_LS_RENDER_FILES = 25
    MAX_STAT_METADATA_FIELDS = 8
    SEMANTIC_GREP_VECTOR_CANDIDATE_LIMIT = 20
    GREP_RECURSIVE_FOLDER_DEPTH_LIMIT = 2
    GREP_RECURSIVE_FOLDER_FILE_LIMIT = 10

    def __init__(
        self,
        filesystem: PageIndexFileSystem,
        *,
        json_output: bool = False,
        query_context: str | None = None,
    ):
        self.filesystem = filesystem
        self.json_output = json_output
        self.query_context = query_context

    def allowed_commands(self) -> set[str]:
        commands = set(self.BASE_ALLOWED_COMMANDS)
        semantic_channels = set(self.filesystem.semantic_retrieval_channels())
        for channel in SEMANTIC_RETRIEVAL_CHANNELS:
            if channel in semantic_channels:
                commands.add(self.SEMANTIC_CHANNEL_COMMANDS[channel])
        if any(channel in semantic_channels for channel in SEMANTIC_GREP_CHANNELS):
            commands.add("semantic-grep")
        return commands

    def command_capabilities(self) -> dict[str, Any]:
        return {
            "allowed_commands": sorted(self.allowed_commands()),
            "retrieval": self.filesystem.retrieval_capabilities(),
        }

    def describe_available_command_surfaces(self) -> str:
        capabilities = self.filesystem.retrieval_capabilities()
        semantic = capabilities["semantic"]
        semantic_channels = set(semantic["channels"])
        lines = [
            "Available command surfaces for this workspace:",
            "- mode: read-only inspection",
            "- ls/tree: folder browsing",
            "- find --where: exact/canonical metadata DSL filtering",
            "- grep -R: recursive lexical/FTS search only; semantic vector prefilter is disabled",
            "- cat --structure/--node/--page: cached PageIndex reads for PDF/Markdown files",
            "- cat --all: full text artifact reads for txt/text files",
        ]
        if "entity" in semantic_channels:
            lines.append("- find --name: entity semantic candidate discovery alias")
        if "relation" in semantic_channels:
            lines.append("- find --relation: relation semantic candidate discovery alias")
        for channel in SEMANTIC_RETRIEVAL_CHANNELS:
            if channel not in semantic_channels:
                continue
            lines.append(
                f"- {self.SEMANTIC_CHANNEL_COMMANDS[channel]}: "
                f"{channel} semantic vector candidate discovery"
            )
        semantic_grep_channels = semantic.get("semantic_grep_channels") or []
        if semantic_grep_channels:
            lines.append(
                "- semantic-grep -R: semantic candidates from "
                + ", ".join(semantic_grep_channels)
                + " indexes followed by real line matching"
            )
        if not semantic.get("commands"):
            lines.append("- semantic vector commands: none available in this workspace")
        lines.append("- grep <query> <ref>, cat, stat: evidence inspection")
        return "\n".join(lines)

    def execute(self, command: str) -> str:
        try:
            if not command.strip():
                raise PIFSCommandError("Empty command")
            commands = self._split_chained_commands(command)
            if len(commands) > 1:
                return "\n".join(self._execute_pipeline(part) for part in commands)
            return self._execute_pipeline(commands[0])
        except PIFSCommandError:
            raise
        except (KeyError, ValueError) as exc:
            raise PIFSCommandError(self._clean_error_message(exc)) from exc

    def _execute_pipeline(self, command: str) -> str:
        commands = self._split_piped_commands(command)
        output = self._execute_single(commands[0])
        for pipe_command in commands[1:]:
            output = self._execute_pipe_filter(output, pipe_command)
        return output

    def _execute_single(self, command: str) -> str:
        self._validate_raw_command(command)
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise PIFSCommandError(f"Invalid command syntax: {exc}") from exc
        if not tokens:
            raise PIFSCommandError("Empty command")
        self._validate_tokens(tokens)
        if "--json" in tokens:
            tokens = [token for token in tokens if token != "--json"]
            json_output = True
        else:
            json_output = self.json_output
        name = tokens[0]
        if name not in self.allowed_commands():
            raise PIFSCommandError(f"Unsupported command: {name}")
        method_name = self.COMMAND_METHODS.get(name, f"_cmd_{name}")
        data = getattr(self, method_name)(tokens[1:])
        return self._render(data, json_output=json_output, command_name=name)

    def _execute_pipe_filter(self, input_text: str, command: str) -> str:
        self._validate_raw_command(command)
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            raise PIFSCommandError(f"Invalid command syntax: {exc}") from exc
        if not tokens:
            raise PIFSCommandError("Empty pipe command")
        self._validate_tokens(tokens)
        name = tokens[0]
        if name not in self.ALLOWED_PIPE_FILTERS:
            raise PIFSCommandError(f"Unsupported pipe command: {name}")
        if name == "head":
            return self._pipe_head_tail(input_text, tokens[1:], from_tail=False)
        if name == "tail":
            return self._pipe_head_tail(input_text, tokens[1:], from_tail=True)
        if name == "grep":
            return self._pipe_grep(input_text, tokens[1:])
        if name == "sed":
            return self._pipe_sed(input_text, tokens[1:])
        raise PIFSCommandError(f"Unsupported pipe command: {name}")

    def _cmd_ls(self, args: list[str]) -> Any:
        recursive = False
        limit = 100
        path = "/"
        i = 0
        while i < len(args):
            arg = args[i]
            if arg in {"-R", "-r", "--recursive"}:
                recursive = True
            elif arg == "--limit":
                i += 1
                limit = int(args[i])
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported ls option: {arg}")
            else:
                path = arg
            i += 1
        return self.filesystem.browse(path, recursive=recursive, limit=limit)

    def _cmd_tree(self, args: list[str]) -> Any:
        path = "/"
        limit = 1000
        depth = 2
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--limit":
                i += 1
                limit = int(args[i])
            elif arg in {"--depth", "-L"}:
                i += 1
                depth = int(args[i])
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported tree option: {arg}")
            else:
                path = arg
            i += 1
        if depth < 1:
            raise PIFSCommandError("tree --depth must be at least 1")
        if depth > self.MAX_TREE_DEPTH:
            depth = self.MAX_TREE_DEPTH
        listing = self.filesystem.browse(path, recursive=True, limit=limit)
        return {"path": path, "depth": depth, "limit": limit, **listing}

    def _cmd_find(self, args: list[str]) -> Any:
        path = "/"
        where = None
        name = None
        relation = None
        limit = 10
        file_type = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--where":
                i += 1
                where = args[i]
            elif arg == "--name":
                i += 1
                name = args[i]
            elif arg == "--relation":
                i += 1
                relation = args[i]
            elif arg == "--limit":
                i += 1
                limit = int(args[i])
            elif arg == "-type":
                i += 1
                file_type = args[i]
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported find option: {arg}")
            else:
                path = arg
            i += 1
        if file_type and file_type not in {"f", "d"}:
            raise PIFSCommandError("find -type supports only f or d")
        if name and relation:
            raise PIFSCommandError("find supports only one of --name or --relation")
        if file_type == "d":
            if where:
                return self.filesystem.find_folders(path, metadata_filter=where, limit=limit)
            return self.filesystem.browse(path, recursive=True, limit=limit)["folders"]
        if relation:
            if not self.filesystem.has_semantic_channel("relation"):
                raise PIFSCommandError(
                    "find --relation requires a relation semantic index in this workspace"
                )
            return self.filesystem.search_semantic_channel(
                "relation",
                self._semantic_retrieval_query(relation),
                scope={"folder_path": path, "recursive": True},
                metadata_filter=where,
                limit=limit,
            )
        if name and self.filesystem.has_semantic_channel("entity"):
            return self.filesystem.search_semantic_channel(
                "entity",
                self._semantic_retrieval_query(name),
                scope={"folder_path": path, "recursive": True},
                metadata_filter=where,
                limit=limit,
            )
        return self.filesystem.search(
            query=name,
            scope={"folder_path": path, "recursive": True},
            metadata_filter=where,
            limit=limit,
            semantic=False,
        )

    def _cmd_grep(self, args: list[str]) -> Any:
        recursive = False
        where = None
        limit = 10
        positionals = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg in {"-R", "-r", "--recursive"}:
                recursive = True
            elif self._is_combined_grep_flag(arg):
                recursive = recursive or "R" in arg or "r" in arg
            elif arg in {"-n", "--line-number", "-i", "--ignore-case"}:
                pass
            elif arg == "--where":
                i += 1
                where = args[i]
            elif arg == "--limit":
                i += 1
                limit = int(args[i])
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported grep option: {arg}")
            else:
                positionals.append(arg)
            i += 1
        if not positionals:
            raise PIFSCommandError("grep requires a query")
        query = positionals[0]
        path = positionals[1] if len(positionals) > 1 else "/"
        if self._is_folder(path):
            normalized = self._normalize_folder_path(path)
            if recursive:
                limit_notice = self._recursive_grep_limit_notice(normalized, query)
                if limit_notice:
                    return limit_notice
                children = self.filesystem.browse(normalized, recursive=False, limit=1000)["folders"]
                if children:
                    direct_results = self.filesystem.search(
                        query=query,
                        scope={"folder_path": normalized, "recursive": False},
                        metadata_filter=where,
                        limit=limit,
                        semantic=False,
                    )
                    if direct_results:
                        return {
                            "mode": "files",
                            "query": query,
                            "scope": normalized,
                            "data": self._grep_file_hits_from_results(direct_results, query),
                        }
                    if where is None:
                        direct_source_hits = self._grep_source_file_hits(
                            normalized,
                            query,
                            limit=limit,
                            direct_only=True,
                        )
                        if direct_source_hits:
                            return {
                                "mode": "files",
                                "query": query,
                                "scope": normalized,
                                "data": direct_source_hits,
                            }
                    ranked = self._rank_child_folders(
                        query=query,
                        children=children,
                        metadata_filter=where,
                        limit=limit,
                    )
                    if not ranked and where is None:
                        ranked = self._rank_child_folders_from_source(
                            query=query,
                            parent_path=normalized,
                            children=children,
                            limit=limit,
                        )
                    return {
                        "mode": "folders",
                        "query": query,
                        "scope": normalized,
                        "data": ranked,
                        "hint": "narrow into one directory, then run grep -R again",
                    }
            results = self.filesystem.search(
                query=query,
                scope={"folder_path": normalized, "recursive": recursive},
                metadata_filter=where,
                limit=limit,
                semantic=False,
            )
            if not results and where is None:
                source_hits = self._grep_source_file_hits(normalized, query, limit=limit)
                return {
                    "mode": "files",
                    "query": query,
                    "scope": normalized,
                    "data": source_hits,
                }
            return {
                "mode": "files",
                "query": query,
                "scope": normalized,
                "data": self._grep_file_hits_from_results(results, query),
            }
        return {
            "mode": "matches",
            "query": query,
            "target": path,
            "data": self._grep_file_matches(path, query, limit=limit),
        }

    def _cmd_cat(self, args: list[str]) -> Any:
        if not args:
            raise PIFSCommandError("cat requires a file target")
        target = None
        location = "all"
        structural_mode: str | None = None
        node_id: str | None = None
        page_range: str | None = None
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--range":
                i += 1
                if i >= len(args):
                    raise PIFSCommandError("cat --range requires a range")
                location = args[i]
            elif arg == "--all":
                location = "all"
            elif arg == "--structure":
                structural_mode = "structure"
            elif arg == "--node":
                i += 1
                if i >= len(args):
                    raise PIFSCommandError("cat --node requires a node id")
                structural_mode = "node"
                node_id = args[i]
            elif arg == "--page":
                i += 1
                if i >= len(args):
                    raise PIFSCommandError("cat --page requires a page range")
                structural_mode = "page"
                page_range = args[i]
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported cat option: {arg}")
            else:
                target = arg
            i += 1
        if not target:
            raise PIFSCommandError("cat requires a file target")
        if structural_mode == "structure":
            return self.filesystem.pageindex_structure(target)
        if structural_mode == "node":
            return self.filesystem.pageindex_node(target, str(node_id))
        if structural_mode == "page":
            return self.filesystem.pageindex_pages(target, str(page_range))
        return self.filesystem.cat_text_artifact(target, location)

    def _cmd_stat(self, args: list[str]) -> Any:
        if args and args[0] == "--schema":
            return self.filesystem._metadata_schema()
        if not args:
            raise PIFSCommandError("stat requires a file target or --schema")
        return {"target": args[0], **self.filesystem._stat(args[0])}

    def _cmd_head(self, args: list[str]) -> Any:
        count, target = self._parse_standalone_head_tail(args, default_count=10)
        opened = self.filesystem.cat_text_artifact(target, "all")
        lines = opened.text.splitlines()
        text = "\n".join(lines[:count])
        return {**self._jsonable(opened), "text": text, "end_line": min(count, len(lines))}

    def _cmd_tail(self, args: list[str]) -> Any:
        count, target = self._parse_standalone_head_tail(args, default_count=10)
        opened = self.filesystem.cat_text_artifact(target, "all")
        lines = opened.text.splitlines()
        selected = lines[-count:] if count else []
        start_line = max(1, len(lines) - len(selected) + 1)
        return {
            **self._jsonable(opened),
            "text": "\n".join(selected),
            "start_line": start_line,
            "end_line": len(lines),
        }

    def _cmd_sed(self, args: list[str]) -> Any:
        if len(args) < 3 or args[0] != "-n":
            raise PIFSCommandError("sed supports only: sed -n '<start>,<end>p' <target>")
        match = re.fullmatch(r"(\d+),(\d+)p", args[1])
        if not match:
            raise PIFSCommandError("sed supports only: sed -n '<start>,<end>p' <target>")
        return self.filesystem.cat_text_artifact(
            args[2],
            f"{match.group(1)}-{match.group(2)}",
        )

    def _cmd_search_summary(self, args: list[str]) -> Any:
        return self._cmd_semantic_channel("summary", args)

    def _cmd_search_entity(self, args: list[str]) -> Any:
        return self._cmd_semantic_channel("entity", args)

    def _cmd_search_relation(self, args: list[str]) -> Any:
        return self._cmd_semantic_channel("relation", args)

    def _cmd_semantic_grep(self, args: list[str]) -> Any:
        recursive = False
        where = None
        limit = 10
        positionals = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg in {"-R", "-r", "--recursive"}:
                recursive = True
            elif self._is_combined_grep_flag(arg):
                recursive = recursive or "R" in arg or "r" in arg
            elif arg in {"-n", "--line-number", "-i", "--ignore-case"}:
                pass
            elif arg == "--where":
                i += 1
                where = args[i]
            elif arg == "--limit":
                i += 1
                limit = int(args[i])
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported semantic-grep option: {arg}")
            else:
                positionals.append(arg)
            i += 1
        if not recursive:
            raise PIFSCommandError("semantic-grep requires -R/--recursive")
        channels = self._semantic_grep_channels()
        if not channels:
            raise PIFSCommandError(
                "semantic-grep is not available; entity/relation semantic indexes are not configured"
            )
        if not positionals:
            raise PIFSCommandError("semantic-grep requires a query")
        query = positionals[0]
        path = positionals[1] if len(positionals) > 1 else "/"
        if not self._is_folder(path):
            raise PIFSCommandError("semantic-grep target must be a folder")
        return self._semantic_recursive_grep(
            self._normalize_folder_path(path),
            query,
            metadata_filter=where,
            limit=limit,
            channels=channels,
        )

    def _cmd_semantic_channel(self, channel: str, args: list[str]) -> Any:
        if not self.filesystem.has_semantic_channel(channel):
            raise PIFSCommandError(
                f"search-{channel} is not available; {channel} semantic index is not configured"
            )
        where = None
        limit = 10
        positionals = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--where":
                i += 1
                where = args[i]
            elif arg == "--limit":
                i += 1
                limit = int(args[i])
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported search-{channel} option: {arg}")
            else:
                positionals.append(arg)
            i += 1
        if not positionals:
            raise PIFSCommandError(f"search-{channel} requires a query")
        query = positionals[0]
        path = positionals[1] if len(positionals) > 1 else "/"
        normalized = self._normalize_folder_path(path)
        results = self.filesystem.search_semantic_channel(
            channel,
            self._semantic_retrieval_query(query),
            scope={"folder_path": normalized, "recursive": True},
            metadata_filter=where,
            limit=limit,
        )
        return {
            "mode": "files",
            "query": query,
            "scope": normalized,
            "retrieval": f"{channel}_vector",
            "data": self._grep_file_hits_from_results(results, query),
        }

    def _semantic_recursive_grep(
        self,
        folder_path: str,
        query: str,
        *,
        metadata_filter: str | None,
        limit: int,
        channels: tuple[str, ...],
    ) -> dict[str, Any]:
        vector_query = str(query or "").strip()
        candidate_debug: dict[str, Any] = {}
        for channel in channels:
            channel_results = self.filesystem.search_semantic_channel(
                channel,
                vector_query,
                scope={"folder_path": folder_path, "recursive": True},
                metadata_filter=metadata_filter,
                limit=self.SEMANTIC_GREP_VECTOR_CANDIDATE_LIMIT,
            )
            matches = self._grep_file_hits_from_results(
                channel_results,
                query,
                require_match=True,
                limit=limit,
            )
            candidate_debug[channel] = {
                "candidates": len(channel_results),
                "line_matches": len(matches),
                "candidate_doc_ids": [
                    getattr(result, "external_id", None)
                    for result in channel_results[:5]
                ],
            }
            if matches:
                return {
                    "mode": "files",
                    "query": query,
                    "scope": folder_path,
                    "retrieval": "semantic_grep_" + "_then_".join(channels),
                    "candidate_limit_per_channel": self.SEMANTIC_GREP_VECTOR_CANDIDATE_LIMIT,
                    "matched_channel": channel,
                    "candidate_debug": candidate_debug,
                    "data": matches,
                }
        return {
            "mode": "files",
            "query": query,
            "scope": folder_path,
            "retrieval": "semantic_grep_" + "_then_".join(channels),
            "candidate_limit_per_channel": self.SEMANTIC_GREP_VECTOR_CANDIDATE_LIMIT,
            "matched_channel": "",
            "candidate_debug": candidate_debug,
            "data": [],
        }

    def _semantic_grep_channels(self) -> tuple[str, ...]:
        available = set(self.filesystem.semantic_retrieval_channels())
        return tuple(channel for channel in SEMANTIC_GREP_CHANNELS if channel in available)

    def _render(self, data: Any, *, json_output: bool, command_name: str) -> str:
        jsonable = self._jsonable(data)
        if json_output:
            return json.dumps({"ok": True, "data": jsonable}, ensure_ascii=False)
        return self._render_shell(command_name, jsonable)

    def _render_shell(self, command_name: str, data: Any) -> str:
        if command_name == "cat":
            return self._render_cat(data)
        if command_name == "ls":
            return self._render_listing(data)
        if command_name == "tree":
            return self._render_tree(data)
        if command_name in {"grep", "semantic-grep"}:
            return self._render_grep(data)
        if command_name in {"search-summary", "search-entity", "search-relation"}:
            return self._render_grep(data)
        if command_name == "find":
            return self._render_find(data)
        if command_name == "stat":
            return self._render_stat(data)
        if command_name in {"head", "tail", "sed"}:
            return str(data.get("text", "")) if isinstance(data, dict) else str(data)
        if isinstance(data, dict):
            return "\n".join(f"{key}: {value}" for key, value in data.items())
        if isinstance(data, list):
            return "\n".join(str(item) for item in data)
        return str(data)

    def _render_cat(self, data: Any) -> str:
        if not isinstance(data, dict):
            return str(data)
        if data.get("available") is False:
            return f"# {data.get('message', 'PageIndex structural content is unavailable')}"
        if data.get("mode") == "structure":
            return json.dumps(data.get("structure", {}), ensure_ascii=False, indent=2)
        return str(data.get("text", ""))

    def _render_listing(self, data: Any) -> str:
        if not isinstance(data, dict):
            return str(data)
        lines: list[str] = []
        for folder in data.get("folders", []):
            name = folder["path"] if folder.get("path", "").startswith("/") else folder["name"]
            if not name.endswith("/"):
                name = f"{name}/"
            lines.append(
                f"{name} folders={folder.get('children_count', 0)} files={folder.get('file_count', 0)}"
            )
        files = data.get("files", [])
        for file in files[: self.MAX_LS_RENDER_FILES]:
            lines.append(self._file_row_text(file))
        if len(files) > self.MAX_LS_RENDER_FILES:
            remaining = len(files) - self.MAX_LS_RENDER_FILES
            lines.append(
                f"# ... {remaining} more files omitted from ls output; use grep/find to search this folder"
            )
        return "\n".join(lines)

    def _render_tree(self, data: Any) -> str:
        if not isinstance(data, dict):
            return str(data)
        root = self._normalize_folder_path(data.get("path", "/"))
        max_depth = int(data.get("depth", 2))
        lines = [root]
        folders = [
            folder
            for folder in data.get("folders", [])
            if self._relative_depth(root, folder["path"]) <= max_depth
        ]
        for folder in folders:
            depth = self._relative_depth(root, folder["path"])
            indent = "  " * max(depth - 1, 0)
            lines.append(
                f"{indent}{folder['name']}/ folders={folder.get('children_count', 0)} "
                f"files={folder.get('file_count', 0)}"
            )
        if len(folders) < len(data.get("folders", [])):
            lines.append(f"# truncated at depth={max_depth}")
        return "\n".join(lines)

    def _render_grep(self, data: Any) -> str:
        if not isinstance(data, dict):
            return str(data)
        mode = data.get("mode")
        if mode == "folders":
            lines = [f"# folder matches for: {data.get('query', '')}"]
            for folder in data.get("data", []):
                path = folder["path"]
                if not path.endswith("/"):
                    path = f"{path}/"
                lines.append(
                    f"{path} matched_files={folder.get('matched_files', 0)} "
                    f"files={folder.get('files', 0)}"
                )
            lines.append(f"# {data.get('hint', 'narrow into one directory, then run grep -R again')}")
            return "\n".join(lines)
        if mode == "limited":
            query = str(data.get("query") or "")
            scope = str(data.get("scope") or "/")
            suggested_commands = list(data.get("suggested_commands") or [])
            lines = [
                f"# grep -R skipped for broad folder: {scope}",
                (
                    "# reason: recursive lexical grep is limited when a folder is deeper "
                    f"than {data.get('folder_depth_limit', self.GREP_RECURSIVE_FOLDER_DEPTH_LIMIT)} "
                    f"levels or has more than {data.get('file_count_limit', self.GREP_RECURSIVE_FOLDER_FILE_LIMIT)} files"
                ),
            ]
            if suggested_commands:
                lines.extend(f"# suggested: {command}" for command in suggested_commands)
                lines.append("# also try: narrow with ls/tree/find --where")
            else:
                lines.append("# suggested: narrow with ls/tree/find --where")
            if data.get("sample_deep_folder_path"):
                lines.append(f"# deep descendant example: {data['sample_deep_folder_path']}/")
            return "\n".join(lines)
        if mode == "files":
            if not data.get("data", []):
                return f"# no matches for: {data.get('query', '')}"
            return "\n".join(
                self._grep_file_hit_text(item)
                for item in data.get("data", [])
            )
        if mode == "matches":
            return "\n".join(
                f"{item['reference_id']}:{item['line']}: "
                f"{self._compact_text(item['text'], max_chars=220)}"
                for item in data.get("data", [])
            )
        return str(data)

    def _render_find(self, data: Any) -> str:
        if not isinstance(data, list):
            return str(data)
        if data and isinstance(data[0], dict) and "path" in data[0] and "file_ref" not in data[0]:
            return "\n".join(
                (
                    f"{item['path']}/ matched_files={item['matched_files']} "
                    f"files={item.get('file_count', 0)}"
                    if item.get("matched_files")
                    else f"{item['path']}/ folders={item.get('children_count', 0)} "
                    f"files={item.get('file_count', 0)}"
                )
                for item in data
            )
        return "\n".join(self._file_row_text(item) for item in data)

    def _render_stat(self, data: Any) -> str:
        if not isinstance(data, dict):
            return str(data)
        if "fields" in data:
            lines = ["metadata schema:"]
            for name, field in sorted(data["fields"].items()):
                lines.append(f"{name}: {field.get('type', 'string')}")
            return "\n".join(lines)
        lines = [
            f"ref: {data.get('target') or data.get('file_ref')}",
            f"file_ref: {data.get('file_ref')}",
            f"document_id: {data.get('external_id') or data.get('document_id') or '-'}",
            f"source_path: {data.get('source_path') or '-'}",
            f"storage_uri: {data.get('storage_uri') or '-'}",
        ]
        folders = data.get("folders") or []
        if folders:
            lines.append("folders:")
            lines.extend(f"  {folder['path']}" for folder in folders)
        metadata = data.get("metadata") or {}
        if metadata:
            lines.append("metadata:")
            metadata_items = sorted(metadata.items())[: self.MAX_STAT_METADATA_FIELDS]
            for key, value in metadata_items:
                lines.append(f"  {key}: {self._compact_value(value)}")
            if len(metadata) > self.MAX_STAT_METADATA_FIELDS:
                lines.append(f"  ... {len(metadata) - self.MAX_STAT_METADATA_FIELDS} more fields")
        return "\n".join(lines)

    def _file_row_text(self, item: dict[str, Any]) -> str:
        file_ref = item.get("file_ref")
        ref = item.get("reference_id") or (self.filesystem._reference_for(file_ref) if file_ref else "-")
        doc_id = item.get("external_id") or item.get("document_id") or "-"
        title = self._compact_text(item.get("title") or item.get("name") or "", max_chars=80)
        source_path = item.get("source_path") or "-"
        folder_paths = item.get("folder_paths") or self._folder_paths_for_file(file_ref)
        folders = f" folders={','.join(folder_paths)}" if folder_paths else ""
        return f"{ref} {doc_id} {title} {source_path}{folders}".strip()

    def _grep_file_hit_text(self, item: dict[str, Any]) -> str:
        doc_id = item.get("external_id") or "-"
        source_path = item.get("source_path") or "-"
        line = item.get("line") or 1
        return (
            f"{item['reference_id']} {doc_id} {source_path}:{line}: "
            f"{self._compact_text(item.get('text') or '', max_chars=180)}"
        )

    def _semantic_retrieval_query(self, query: str) -> str:
        query = str(query or "").strip()
        context = str(self.query_context or "").strip()
        if context and query and query.lower() not in context.lower():
            return f"{context}\nSearch phrase: {query}"
        return context or query

    def _recursive_grep_limit_notice(self, folder_path: str, query: str) -> dict[str, Any] | None:
        stats = self.filesystem.store.folder_subtree_thresholds(
            folder_path,
            depth_limit=self.GREP_RECURSIVE_FOLDER_DEPTH_LIMIT,
            file_limit=self.GREP_RECURSIVE_FOLDER_FILE_LIMIT,
        )
        if not (
            stats["folder_depth_exceeds_limit"]
            or stats["file_count_exceeds_limit"]
        ):
            return None
        suggested_commands = self._semantic_alternative_commands(query, folder_path)
        semantic_hint = (
            "Use " + "; ".join(suggested_commands) + " to discover candidates. "
            if suggested_commands
            else ""
        )
        return {
            "mode": "limited",
            "query": query,
            "scope": folder_path,
            "folder_depth_limit": stats["depth_limit"],
            "file_count_limit": stats["file_limit"],
            "folder_depth_exceeds_limit": stats["folder_depth_exceeds_limit"],
            "file_count_exceeds_limit": stats["file_count_exceeds_limit"],
            "sampled_file_count": stats["sampled_file_count"],
            "sample_deep_folder_path": stats["sample_deep_folder_path"],
            "suggested_commands": suggested_commands,
            "hint": (
                "Default grep -R remains lexical and is intentionally limited for broad deep folders "
                "because the SQLite FTS path cannot guarantee fast recursive search at this scope. "
                f"{semantic_hint}Use ls/tree or find --where to narrow first."
            ),
        }

    def _semantic_alternative_commands(self, query: str, folder_path: str) -> list[str]:
        commands = []
        quoted_query = shlex.quote(query)
        quoted_folder = shlex.quote(folder_path)
        if self._semantic_grep_channels():
            commands.append(f"semantic-grep -R {quoted_query} {quoted_folder}")
        for channel in SEMANTIC_RETRIEVAL_CHANNELS:
            if self.filesystem.has_semantic_channel(channel):
                command = self.SEMANTIC_CHANNEL_COMMANDS[channel]
                commands.append(f"{command} {quoted_query} {quoted_folder}")
        return commands

    def _rank_child_folders(
        self,
        *,
        query: str,
        children: list[dict[str, Any]],
        metadata_filter: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for child in children:
            results = self.filesystem.search(
                query=query,
                scope={"folder_path": child["path"], "recursive": True},
                metadata_filter=metadata_filter,
                limit=max(limit, 50),
                semantic=False,
            )
            if not results:
                continue
            ranked.append(
                {
                    "path": child["path"],
                    "name": child["name"],
                    "matched_files": len(results),
                    "files": self.filesystem.store.count_files_in_folder(child["path"], recursive=True),
                    "children_count": child.get("children_count", 0),
                }
            )
        ranked.sort(key=lambda item: (-item["matched_files"], item["path"]))
        return ranked[:limit]

    def _grep_file_hits_from_results(
        self,
        results: list[Any],
        query: str,
        *,
        require_match: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        hits = []
        for result in results:
            line, text = self._first_matching_line(result.file_ref, query)
            if require_match and not text:
                continue
            hits.append(
                {
                    "reference_id": result.reference_id,
                    "file_ref": result.file_ref,
                    "external_id": result.external_id,
                    "title": result.title,
                    "source_path": result.source_path,
                    "folder_paths": result.folder_paths,
                    "line": line,
                    "text": text or result.snippet,
                }
            )
            if limit is not None and len(hits) >= limit:
                break
        return hits

    def _rank_child_folders_from_source(
        self,
        *,
        query: str,
        parent_path: str,
        children: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        source_dir = self._source_dir_for_folder(parent_path)
        source_root = self._source_root()
        if source_dir is None or source_root is None:
            return []
        child_paths = {child["path"]: child for child in children}
        counts: dict[str, int] = {}
        for path in self._rg_candidate_files(query, source_dir, max_files=5000):
            source_path = self._source_path_from_storage(path, source_root)
            folder_path = "/" + str(Path(source_path).parent).strip("/")
            child_path = self._matching_child_path(parent_path, folder_path, child_paths)
            if child_path:
                counts[child_path] = counts.get(child_path, 0) + 1
        ranked = [
            {
                "path": path,
                "name": child_paths[path]["name"],
                "matched_files": matched,
                "files": self.filesystem.store.count_files_in_folder(path, recursive=True),
                "children_count": child_paths[path].get("children_count", 0),
            }
            for path, matched in counts.items()
        ]
        ranked.sort(key=lambda item: (-item["matched_files"], item["path"]))
        return ranked[:limit]

    def _grep_source_file_hits(
        self,
        folder_path: str,
        query: str,
        *,
        limit: int,
        direct_only: bool = False,
    ) -> list[dict[str, Any]]:
        source_dir = self._source_dir_for_folder(folder_path)
        source_root = self._source_root()
        if source_dir is None or source_root is None:
            return []
        hits = []
        for path in self._rg_candidate_files(query, source_dir, max_files=max(limit * 10, 50)):
            file_row = self._file_row_for_storage(path)
            if not file_row:
                continue
            if direct_only and self._folder_path_for_source_path(file_row["source_path"]) != folder_path:
                continue
            reference_id = self.filesystem._reference_for(file_row["file_ref"])
            line_number, text = self._first_matching_source_line(path, query)
            hits.append(
                {
                    "reference_id": reference_id,
                    "file_ref": file_row["file_ref"],
                    "external_id": file_row["external_id"],
                    "title": file_row["title"],
                    "source_path": file_row["source_path"],
                    "folder_paths": self._folder_paths_for_file(file_row["file_ref"]),
                    "line": line_number,
                    "text": text or file_row["title"],
                }
            )
            if len(hits) >= limit:
                break
        return hits

    def _grep_file_matches(self, target: str, query: str, *, limit: int) -> list[dict[str, Any]]:
        file_ref = self.filesystem._resolve_reference(target)
        reference_id = self.filesystem._reference_for(file_ref)
        entry = self.filesystem.store.get_file(file_ref)
        matches = []
        for line_number, line in enumerate(self.filesystem.store.read_text(file_ref).splitlines(), 1):
            if self._line_matches(line, query):
                matches.append(
                    {
                        "reference_id": reference_id,
                        "file_ref": file_ref,
                        "external_id": entry.external_id,
                        "source_path": entry.source_path,
                        "line": line_number,
                        "text": self._compact_text(line, max_chars=220),
                    }
                )
                if len(matches) >= limit:
                    break
        return matches

    def _first_matching_line(self, file_ref: str, query: str) -> tuple[int, str]:
        for line_number, line in enumerate(self.filesystem.store.read_text(file_ref).splitlines(), 1):
            if self._line_matches(line, query):
                return line_number, self._compact_text(line, max_chars=220)
        return 1, ""

    def _line_matches(self, line: str, query: str) -> bool:
        haystack = line.lower()
        needle = query.lower().strip()
        if needle and needle in haystack:
            return True
        terms = [term for term in re.findall(r"[A-Za-z0-9_]+", needle) if term]
        return bool(terms) and all(term in haystack for term in terms)

    @staticmethod
    def _is_combined_grep_flag(arg: str) -> bool:
        return bool(re.fullmatch(r"-[Rrni]+", arg)) and len(arg) > 2

    def _rg_candidate_files(self, query: str, directory: Path, *, max_files: int) -> list[Path]:
        if not directory.exists():
            return []
        terms = [term.lower() for term in re.findall(r"[A-Za-z0-9_]{3,}", query)]
        if not terms:
            return []
        primary = max(terms, key=len)
        try:
            completed = subprocess.run(
                [
                    "rg",
                    "-l",
                    "-i",
                    "-F",
                    primary,
                    str(directory),
                    "--glob",
                    "*.json",
                    "--no-messages",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []
        candidates = [Path(line) for line in completed.stdout.splitlines() if line.strip()]
        filtered = []
        for path in candidates[: max(max_files * 20, max_files)]:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            if all(term in text for term in terms):
                filtered.append(path)
                if len(filtered) >= max_files:
                    break
        return filtered

    def _first_matching_source_line(self, path: Path, query: str) -> tuple[int, str]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return 1, ""
        for line_number, line in enumerate(lines, 1):
            if self._line_matches(line, query):
                return line_number, self._compact_text(line, max_chars=220)
        return 1, self._compact_text(lines[0], max_chars=220) if lines else ""

    def _source_root(self) -> Path | None:
        with self.filesystem.store.connect() as conn:
            row = conn.execute(
                """
                SELECT storage_uri, source_path
                FROM files
                WHERE deleted_at IS NULL
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        storage_path = Path(row["storage_uri"])
        source_path = Path(row["source_path"])
        root = storage_path
        for _part in source_path.parts:
            root = root.parent
        return root

    def _source_dir_for_folder(self, folder_path: str) -> Path | None:
        source_root = self._source_root()
        if source_root is None:
            return None
        stripped = folder_path.strip("/")
        return source_root / stripped if stripped else source_root

    @staticmethod
    def _source_path_from_storage(path: Path, source_root: Path) -> str:
        try:
            return path.relative_to(source_root).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def _matching_child_path(
        parent_path: str,
        folder_path: str,
        child_paths: dict[str, dict[str, Any]],
    ) -> str | None:
        normalized_parent = parent_path.rstrip("/")
        if normalized_parent == "":
            normalized_parent = "/"
        if normalized_parent == "/":
            parts = [part for part in folder_path.strip("/").split("/") if part]
            candidate = "/" + parts[0] if parts else "/"
            return candidate if candidate in child_paths else None
        prefix = normalized_parent + "/"
        if not folder_path.startswith(prefix):
            return None
        remainder = folder_path[len(prefix):]
        first = remainder.split("/", 1)[0]
        candidate = prefix + first
        return candidate if candidate in child_paths else None

    def _file_row_for_storage(self, path: Path) -> dict[str, Any] | None:
        storage_uri = str(path)
        with self.filesystem.store.connect() as conn:
            row = conn.execute(
                """
                SELECT file_ref, external_id, title, source_path
                FROM files
                WHERE storage_uri = ? AND deleted_at IS NULL
                LIMIT 1
                """,
                (storage_uri,),
            ).fetchone()
        if row is None:
            return None
        return {
            "file_ref": row["file_ref"],
            "external_id": row["external_id"],
            "title": row["title"],
            "source_path": row["source_path"],
        }

    @staticmethod
    def _folder_path_for_source_path(source_path: str) -> str:
        parent = str(Path(source_path).parent).strip(".")
        return "/" + parent.strip("/") if parent and parent != "." else "/"

    def _folder_paths_for_file(self, file_ref: str | None) -> list[str]:
        if not file_ref:
            return []
        try:
            return [folder["path"] for folder in self.filesystem.store.folder_memberships(file_ref)]
        except KeyError:
            return []

    def _is_folder(self, path: str) -> bool:
        try:
            self.filesystem.browse(path, recursive=False, limit=1)
            return True
        except KeyError:
            return False

    @staticmethod
    def _normalize_folder_path(path: str) -> str:
        value = str(path or "/").strip()
        if not value or value == "/":
            return "/"
        return "/" + value.strip("/")

    @classmethod
    def _relative_depth(cls, root: str, path: str) -> int:
        root = cls._normalize_folder_path(root).rstrip("/")
        path = cls._normalize_folder_path(path).rstrip("/")
        if root == "":
            root = "/"
        if root == "/":
            rel = path.strip("/")
        else:
            rel = path[len(root):].strip("/")
        return 0 if not rel else len(rel.split("/"))

    @classmethod
    def _compact_value(cls, value: Any) -> str:
        if isinstance(value, list):
            rendered = ", ".join(cls._compact_text(str(item), max_chars=40) for item in value[:3])
            if len(value) > 3:
                rendered += f", ... {len(value) - 3} more"
            return rendered
        if isinstance(value, dict):
            return cls._compact_text(json.dumps(value, ensure_ascii=False, sort_keys=True), max_chars=120)
        return cls._compact_text(str(value), max_chars=120)

    @staticmethod
    def _compact_text(text: str, *, max_chars: int) -> str:
        collapsed = re.sub(r"\s+", " ", text or "").strip()
        if len(collapsed) <= max_chars:
            return collapsed
        return collapsed[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _clean_error_message(exc: BaseException) -> str:
        message = str(exc)
        if isinstance(exc, KeyError) and len(exc.args) == 1:
            message = str(exc.args[0])
        return message or exc.__class__.__name__

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, list):
            return [cls._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._jsonable(item) for key, item in value.items()}
        return value

    @classmethod
    def _validate_raw_command(cls, command: str) -> None:
        if any(token in command for token in cls.FORBIDDEN_SUBSTRINGS):
            raise PIFSCommandError("Only PageIndex FileSystem commands are allowed")

    @classmethod
    def _validate_tokens(cls, tokens: list[str]) -> None:
        if any(token in cls.FORBIDDEN_TOKENS for token in tokens):
            raise PIFSCommandError("Only PageIndex FileSystem commands are allowed")

    @classmethod
    def _split_chained_commands(cls, command: str) -> list[str]:
        return cls._split_unquoted_operator(command, "&&", reject_single_amp=True)

    @classmethod
    def _split_piped_commands(cls, command: str) -> list[str]:
        return cls._split_unquoted_operator(command, "|")

    @classmethod
    def _split_unquoted_operator(
        cls,
        command: str,
        operator: str,
        *,
        reject_single_amp: bool = False,
    ) -> list[str]:
        cls._validate_raw_command(command)
        parts: list[str] = []
        current: list[str] = []
        quote: str | None = None
        escaped = False
        i = 0
        while i < len(command):
            char = command[i]
            if escaped:
                current.append(char)
                escaped = False
                i += 1
                continue
            if char == "\\" and quote != "'":
                current.append(char)
                escaped = True
                i += 1
                continue
            if quote:
                current.append(char)
                if char == quote:
                    quote = None
                i += 1
                continue
            if char in {"'", '"'}:
                quote = char
                current.append(char)
                i += 1
                continue
            if command.startswith(operator, i):
                part = "".join(current).strip()
                if not part:
                    raise PIFSCommandError("Invalid command syntax")
                parts.append(part)
                current = []
                i += len(operator)
                continue
            if reject_single_amp and char == "&":
                raise PIFSCommandError("Only PageIndex FileSystem commands are allowed")
            current.append(char)
            i += 1
        part = "".join(current).strip()
        if quote:
            raise PIFSCommandError("Invalid command syntax: No closing quotation")
        if not part:
            raise PIFSCommandError("Invalid command syntax")
        parts.append(part)
        return parts

    def _pipe_head_tail(self, input_text: str, args: list[str], *, from_tail: bool) -> str:
        count = self._parse_head_tail_count(args)
        payload = self._try_json_loads(input_text)
        if payload is not None:
            return self._render_json_payload(self._slice_payload(payload, count, from_tail=from_tail))
        lines = input_text.splitlines()
        selected = [] if count == 0 else lines[-count:] if from_tail else lines[:count]
        return "\n".join(selected)

    def _pipe_grep(self, input_text: str, args: list[str]) -> str:
        ignore_case = False
        invert = False
        regex = False
        patterns: list[str] = []
        for arg in args:
            if arg in {"-i", "--ignore-case"}:
                ignore_case = True
            elif arg in {"-v", "--invert-match"}:
                invert = True
            elif arg in {"-E", "--extended-regexp"}:
                regex = True
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported pipe grep option: {arg}")
            else:
                patterns.append(arg)
        if len(patterns) != 1:
            raise PIFSCommandError("pipe grep requires exactly one pattern")
        pattern = patterns[0]
        payload = self._try_json_loads(input_text)
        if payload is not None:
            return self._render_json_payload(
                self._filter_payload(
                    payload,
                    pattern,
                    ignore_case=ignore_case,
                    invert=invert,
                    regex=regex,
                )
            )
        filtered = [
            line
            for line in input_text.splitlines()
            if self._text_matches(line, pattern, ignore_case=ignore_case, invert=invert, regex=regex)
        ]
        return "\n".join(filtered)

    def _pipe_sed(self, input_text: str, args: list[str]) -> str:
        if not args:
            raise PIFSCommandError("pipe sed requires an expression")
        if args[0] == "-n":
            args = args[1:]
        if len(args) != 1:
            raise PIFSCommandError("pipe sed supports only -n '<start>,<end>p'")
        match = re.fullmatch(r"(\d+)(?:,(\d+))?p", args[0])
        if not match:
            raise PIFSCommandError("pipe sed supports only -n '<start>,<end>p'")
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        if start < 1 or end < start:
            raise PIFSCommandError("Invalid sed line range")
        payload = self._try_json_loads(input_text)
        if payload is not None:
            return self._render_json_payload(self._slice_text_payload(payload, start, end))
        lines = input_text.splitlines()
        return "\n".join(lines[start - 1 : end])

    @staticmethod
    def _parse_head_tail_count(args: list[str]) -> int:
        count = 10
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-n":
                i += 1
                if i >= len(args):
                    raise PIFSCommandError("head/tail -n requires a count")
                count = PIFSCommandExecutor._parse_non_negative_int(args[i], "head/tail count")
            elif re.fullmatch(r"-\d+", arg):
                count = PIFSCommandExecutor._parse_non_negative_int(arg[1:], "head/tail count")
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported head/tail option: {arg}")
            else:
                count = PIFSCommandExecutor._parse_non_negative_int(arg, "head/tail count")
            i += 1
        return count

    @staticmethod
    def _parse_standalone_head_tail(args: list[str], *, default_count: int) -> tuple[int, str]:
        count = default_count
        target = ""
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-n":
                i += 1
                if i >= len(args):
                    raise PIFSCommandError("head/tail -n requires a count")
                count = PIFSCommandExecutor._parse_non_negative_int(args[i], "head/tail count")
            elif re.fullmatch(r"-\d+", arg):
                count = PIFSCommandExecutor._parse_non_negative_int(arg[1:], "head/tail count")
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported head/tail option: {arg}")
            else:
                target = arg
            i += 1
        if not target:
            raise PIFSCommandError("head/tail requires a file target")
        return count, target

    @staticmethod
    def _parse_non_negative_int(value: str, label: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise PIFSCommandError(f"{label} must be an integer") from exc
        if parsed < 0:
            raise PIFSCommandError(f"{label} must be non-negative")
        return parsed

    @staticmethod
    def _try_json_loads(input_text: str) -> Any | None:
        try:
            return json.loads(input_text)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _render_json_payload(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def _slice_payload(cls, payload: Any, count: int, *, from_tail: bool) -> Any:
        if isinstance(payload, list):
            return payload[-count:] if from_tail and count else payload[:count]
        if not isinstance(payload, dict):
            return payload
        sliced = dict(payload)
        if "data" in sliced:
            sliced["data"] = cls._slice_data(sliced["data"], count, from_tail=from_tail)
        else:
            sliced = cls._slice_mapping_lists(sliced, count, from_tail=from_tail)
        return sliced

    @classmethod
    def _slice_data(cls, data: Any, count: int, *, from_tail: bool) -> Any:
        if isinstance(data, list):
            return data[-count:] if from_tail and count else data[:count]
        if isinstance(data, dict):
            if isinstance(data.get("text"), str):
                copied = dict(data)
                lines = copied["text"].splitlines()
                copied["text"] = "\n".join(lines[-count:] if from_tail and count else lines[:count])
                return copied
            return cls._slice_mapping_lists(data, count, from_tail=from_tail)
        return data

    @classmethod
    def _slice_mapping_lists(cls, data: dict[str, Any], count: int, *, from_tail: bool) -> dict[str, Any]:
        copied = dict(data)
        for key, value in copied.items():
            if isinstance(value, list):
                copied[key] = value[-count:] if from_tail and count else value[:count]
        return copied

    @classmethod
    def _filter_payload(
        cls,
        payload: Any,
        pattern: str,
        *,
        ignore_case: bool,
        invert: bool,
        regex: bool,
    ) -> Any:
        if isinstance(payload, list):
            return [
                item
                for item in payload
                if cls._json_matches(item, pattern, ignore_case=ignore_case, invert=invert, regex=regex)
            ]
        if not isinstance(payload, dict):
            return payload
        filtered = dict(payload)
        if "data" in filtered:
            filtered["data"] = cls._filter_data(
                filtered["data"],
                pattern,
                ignore_case=ignore_case,
                invert=invert,
                regex=regex,
            )
        else:
            filtered = cls._filter_mapping_lists(
                filtered,
                pattern,
                ignore_case=ignore_case,
                invert=invert,
                regex=regex,
            )
        return filtered

    @classmethod
    def _filter_data(
        cls,
        data: Any,
        pattern: str,
        *,
        ignore_case: bool,
        invert: bool,
        regex: bool,
    ) -> Any:
        if isinstance(data, list):
            return [
                item
                for item in data
                if cls._json_matches(item, pattern, ignore_case=ignore_case, invert=invert, regex=regex)
            ]
        if isinstance(data, dict):
            return cls._filter_mapping_lists(
                data,
                pattern,
                ignore_case=ignore_case,
                invert=invert,
                regex=regex,
            )
        if isinstance(data, str):
            return "\n".join(
                line
                for line in data.splitlines()
                if cls._text_matches(line, pattern, ignore_case=ignore_case, invert=invert, regex=regex)
            )
        return data

    @classmethod
    def _filter_mapping_lists(
        cls,
        data: dict[str, Any],
        pattern: str,
        *,
        ignore_case: bool,
        invert: bool,
        regex: bool,
    ) -> dict[str, Any]:
        filtered = dict(data)
        for key, value in filtered.items():
            if isinstance(value, list):
                filtered[key] = [
                    item
                    for item in value
                    if cls._json_matches(item, pattern, ignore_case=ignore_case, invert=invert, regex=regex)
                ]
        return filtered

    @classmethod
    def _json_matches(
        cls,
        value: Any,
        pattern: str,
        *,
        ignore_case: bool,
        invert: bool,
        regex: bool,
    ) -> bool:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return cls._text_matches(text, pattern, ignore_case=ignore_case, invert=invert, regex=regex)

    @staticmethod
    def _text_matches(
        text: str,
        pattern: str,
        *,
        ignore_case: bool,
        invert: bool,
        regex: bool,
    ) -> bool:
        flags = re.IGNORECASE if ignore_case else 0
        if regex:
            try:
                matched = re.search(pattern, text, flags) is not None
            except re.error as exc:
                raise PIFSCommandError(f"Invalid grep regex: {exc}") from exc
        elif ignore_case:
            matched = pattern.lower() in text.lower()
        else:
            matched = pattern in text
        return not matched if invert else matched

    @classmethod
    def _slice_text_payload(cls, payload: Any, start: int, end: int) -> Any:
        if not isinstance(payload, dict):
            return payload
        sliced = dict(payload)
        data = sliced.get("data")
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            copied_data = dict(data)
            lines = copied_data["text"].splitlines()
            copied_data["text"] = "\n".join(lines[start - 1 : end])
            copied_data["start_line"] = start
            copied_data["end_line"] = min(end, len(lines))
            sliced["data"] = copied_data
        return sliced
