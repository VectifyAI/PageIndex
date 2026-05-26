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
    MAX_CHAINED_COMMANDS = 3
    MAX_PIPE_COMMANDS = 3
    MAX_LS_LIMIT = 100
    MAX_TREE_LIMIT = 200
    MAX_FIND_LIMIT = 50
    MAX_GREP_LIMIT = 20
    MAX_SEMANTIC_LIMIT = 20
    MAX_TEXT_LINES = 100
    MAX_PAGE_SPAN = 3
    MAX_STRUCTURE_NODES = 25
    MAX_NODE_IDS = 5
    MAX_NODE_TEXT_LINES = 100
    MAX_NODE_TEXT_CHARS = 12_000
    MAX_STAT_FIELD_TARGETS = 20
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
            "- find <folder>: folder path is positional; do not put paths in --where",
            "- find --where: exact/canonical metadata DSL filtering using stat --schema fields only",
            "- find <folder> -maxdepth N -type f|d: bounded folder traversal for find",
            "- grep -R: recursive lexical/FTS search only; semantic vector prefilter is disabled",
            "- cat <path|file_ref|document_id> --structure: cached PageIndex node list, paginated at 25 nodes",
            "- cat <path|file_ref|document_id> --page: cached PageIndex page reads, limited to 3 pages",
            "- cat <path|file_ref|document_id> --node: cached PageIndex node reads, limited to 5 node ids",
            "- cat <path|file_ref|document_id> --all: text artifact reads for txt/text files, paginated at 100 lines",
            "- stat --field <metadata_field> <target...>: one metadata field across up to 20 documents",
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
        lines.append("- grep <query> <path|file_ref|document_id>, cat, stat: evidence inspection")
        return "\n".join(lines)

    def execute(self, command: str) -> str:
        try:
            if not command.strip():
                raise PIFSCommandError("Empty command")
            commands = self._split_chained_commands(command)
            if len(commands) > self.MAX_CHAINED_COMMANDS:
                raise PIFSCommandError(
                    f"Command chain supports at most {self.MAX_CHAINED_COMMANDS} commands. "
                    "Run fewer commands or narrow the request first; if you are unsure where "
                    "to inspect, use cat <target> --structure."
                )
            if len(commands) > 1:
                return "\n".join(self._execute_pipeline(part) for part in commands)
            return self._execute_pipeline(commands[0])
        except PIFSCommandError:
            raise
        except (KeyError, ValueError) as exc:
            raise PIFSCommandError(self._clean_error_message(exc)) from exc

    def _execute_pipeline(self, command: str) -> str:
        commands = self._split_piped_commands(command)
        if len(commands) > self.MAX_PIPE_COMMANDS:
            raise PIFSCommandError(
                f"Pipeline supports at most {self.MAX_PIPE_COMMANDS} commands. "
                "Use a smaller command and explicit limits; if you are unsure where "
                "to inspect, use cat <target> --structure."
            )
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
            raise PIFSCommandError(
                f"Unsupported pipe command: {name}. Supported pipes are: "
                f"{', '.join(sorted(self.ALLOWED_PIPE_FILTERS))}. "
                "If you meant regex alternation such as a|b, PIFS grep/search "
                "does not support it; run multiple grep or search-summary "
                "commands with one phrase each."
            )
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
        limit = self.MAX_LS_LIMIT
        path = "/"
        i = 0
        while i < len(args):
            arg = args[i]
            if arg in {"-R", "-r", "--recursive"}:
                recursive = True
            elif arg == "--limit":
                i += 1
                limit = self._parse_bounded_int(
                    args[i], "ls --limit", max_value=self.MAX_LS_LIMIT
                )
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported ls option: {arg}")
            else:
                path = arg
            i += 1
        return self.filesystem.browse(path, recursive=recursive, limit=limit)

    def _cmd_tree(self, args: list[str]) -> Any:
        path = "/"
        limit = self.MAX_TREE_LIMIT
        depth = 2
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--limit":
                i += 1
                limit = self._parse_bounded_int(
                    args[i], "tree --limit", max_value=self.MAX_TREE_LIMIT
                )
            elif arg in {"--depth", "-L"}:
                i += 1
                depth = self._parse_non_negative_int(args[i], "tree --depth")
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
        max_depth = None
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
                limit = self._parse_bounded_int(
                    args[i], "find --limit", max_value=self.MAX_FIND_LIMIT
                )
            elif arg == "-type":
                i += 1
                file_type = args[i]
            elif arg == "-maxdepth":
                i += 1
                max_depth = self._parse_find_maxdepth(args[i] if i < len(args) else None)
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
                return self.filesystem.find_folders(
                    path,
                    metadata_filter=where,
                    limit=limit,
                    max_depth=max_depth,
                )
            folders = self.filesystem.browse(
                path,
                recursive=True,
                limit=limit,
                max_depth=max_depth,
            )["folders"]
            if max_depth is not None and limit != 0:
                return [self.filesystem.folder_info(path), *folders][:limit]
            return folders
        scope = {"folder_path": path, "recursive": True}
        if max_depth is not None:
            if max_depth == 0:
                return []
            scope["max_depth"] = max_depth
        if relation:
            if not self.filesystem.has_semantic_channel("relation"):
                raise PIFSCommandError(
                    "find --relation requires a relation semantic index in this workspace"
                )
            return self.filesystem.search_semantic_channel(
                "relation",
                self._semantic_retrieval_query(relation),
                scope=scope,
                metadata_filter=where,
                limit=limit,
            )
        if name and self.filesystem.has_semantic_channel("entity"):
            return self.filesystem.search_semantic_channel(
                "entity",
                self._semantic_retrieval_query(name),
                scope=scope,
                metadata_filter=where,
                limit=limit,
            )
        return self.filesystem.search(
            query=name,
            scope=scope,
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
                limit = self._parse_bounded_int(
                    args[i], "grep --limit", max_value=self.MAX_GREP_LIMIT
                )
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported grep option: {arg}")
            else:
                positionals.append(arg)
            i += 1
        if not positionals:
            raise PIFSCommandError("grep requires a query")
        query = positionals[0]
        self._reject_regex_alternation_query(query, "grep")
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
        target = args[0]
        if target.startswith("-"):
            raise PIFSCommandError(
                "cat syntax is target-first: cat <path|file_ref|document_id> --structure, "
                "cat <path|file_ref|document_id> --page 31-59, or "
                "cat <path|file_ref|document_id> --node 0009"
            )
        location = "all"
        structural_mode: str | None = None
        node_ids: list[str] = []
        page_range: str | None = None
        structure_offset = 0
        structure_limit = self.MAX_STRUCTURE_NODES
        i = 1
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
            elif arg == "--offset":
                i += 1
                if i >= len(args):
                    raise PIFSCommandError("cat --structure --offset requires a value")
                structure_offset = self._parse_non_negative_int(args[i], "cat --structure --offset")
            elif arg == "--limit":
                i += 1
                if i >= len(args):
                    raise PIFSCommandError("cat --structure --limit requires a value")
                structure_limit = self._parse_bounded_int(
                    args[i],
                    "cat --structure --limit",
                    max_value=self.MAX_STRUCTURE_NODES,
                )
            elif arg == "--node":
                i += 1
                if i >= len(args):
                    raise PIFSCommandError("cat --node requires a node id")
                structural_mode = "node"
                node_ids.extend(self._parse_node_ids(args[i]))
            elif arg == "--page":
                i += 1
                if i >= len(args):
                    raise PIFSCommandError("cat --page requires a page range")
                structural_mode = "page"
                page_range = args[i]
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported cat option: {arg}")
            else:
                raise PIFSCommandError(
                    "cat accepts one file target. Use: cat <path|file_ref|document_id> --page <page-or-range>, "
                    "for example: cat /documents/report.pdf --page 31-59"
                )
            i += 1
        if structural_mode == "structure":
            if structure_limit < 1:
                raise PIFSCommandError(
                    "cat --structure --limit must be at least 1 and at most "
                    f"{self.MAX_STRUCTURE_NODES}."
                )
            data = self.filesystem.pageindex_structure(
                target,
                offset=structure_offset,
                limit=structure_limit,
            )
            self._attach_structure_next_command(data, target)
            return data
        if structural_mode == "node":
            self._require_at_most(
                len(node_ids),
                "cat --node node count",
                self.MAX_NODE_IDS,
            )
            if not node_ids:
                raise PIFSCommandError("cat --node requires a node id")
            node_results = [
                self._bounded_node_result(
                    self.filesystem.pageindex_node(target, node_id),
                    target=target,
                    node_id=node_id,
                )
                for node_id in node_ids
            ]
            if len(node_results) == 1:
                return node_results[0]
            return {
                "mode": "nodes",
                "target": target,
                "available": all(result.get("available") is not False for result in node_results),
                "node_ids": node_ids,
                "nodes": node_results,
                "text": "\n\n".join(
                    f"[node {result.get('node_id') or node_id}]\n{result.get('text', '')}"
                    for node_id, result in zip(node_ids, node_results)
                ),
            }
        if structural_mode == "page":
            if not page_range or not re.fullmatch(r"\d+(?:-\d+)?", page_range):
                raise PIFSCommandError(
                    "cat --page requires one page selector like 31 or 31-59. "
                    "Use: cat <path|file_ref|document_id> --page <page-or-range>"
                )
            start, end = self._parse_numeric_range(page_range, "cat --page")
            self._require_at_most(
                end - start + 1,
                "cat --page page count",
                self.MAX_PAGE_SPAN,
            )
            data = self.filesystem.pageindex_pages(target, page_range)
            self._attach_page_next_command(data, target, start=start, end=end)
            return data
        return self._bounded_text_artifact(target, location)

    def _cmd_stat(self, args: list[str]) -> Any:
        schema = False
        field: str | None = None
        targets: list[str] = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--schema":
                schema = True
            elif arg == "--field":
                i += 1
                if i >= len(args):
                    raise PIFSCommandError("stat --field requires a metadata field name")
                field = args[i]
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported stat option: {arg}")
            else:
                targets.append(arg)
            i += 1
        if schema:
            if field or targets:
                raise PIFSCommandError("stat --schema cannot be combined with file targets or --field")
            return self.filesystem._metadata_schema()
        if field:
            if not targets:
                raise PIFSCommandError("stat --field requires at least one file target")
            self._require_at_most(
                len(targets),
                "stat --field target count",
                self.MAX_STAT_FIELD_TARGETS,
            )
            self._validate_metadata_field_for_stat(field)
            return {
                "mode": "field_values",
                "field": field,
                "target_count": len(targets),
                "max_targets": self.MAX_STAT_FIELD_TARGETS,
                "data": [self._stat_field_row(field, target) for target in targets],
            }
        if not targets:
            raise PIFSCommandError("stat requires a file target or --schema")
        self._require_at_most(
            len(targets),
            "stat target count",
            self.MAX_STAT_FIELD_TARGETS,
        )
        if len(targets) == 1:
            return {"target": targets[0], **self.filesystem._stat(targets[0])}
        return {
            "mode": "files",
            "target_count": len(targets),
            "data": [{"target": target, **self.filesystem._stat(target)} for target in targets],
        }

    def _cmd_head(self, args: list[str]) -> Any:
        count, target = self._parse_standalone_head_tail(args, default_count=10)
        count = self._require_at_most(count, "head line count", self.MAX_TEXT_LINES)
        opened = self.filesystem.cat_text_artifact(target, "all")
        lines = opened.text.splitlines()
        text = "\n".join(lines[:count])
        return {**self._jsonable(opened), "text": text, "end_line": min(count, len(lines))}

    def _cmd_tail(self, args: list[str]) -> Any:
        count, target = self._parse_standalone_head_tail(args, default_count=10)
        count = self._require_at_most(count, "tail line count", self.MAX_TEXT_LINES)
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
        start, end = int(match.group(1)), int(match.group(2))
        if start < 1 or end < start:
            raise PIFSCommandError("Invalid sed line range")
        self._require_at_most(end - start + 1, "sed line count", self.MAX_TEXT_LINES)
        return self.filesystem.cat_text_artifact(
            args[2],
            f"{start}-{end}",
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
                limit = self._parse_bounded_int(
                    args[i], "semantic-grep --limit", max_value=self.MAX_SEMANTIC_LIMIT
                )
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
        self._validate_search_positionals("semantic-grep", positionals)
        query = positionals[0]
        self._reject_regex_alternation_query(query, "semantic-grep")
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
                limit = self._parse_bounded_int(
                    args[i],
                    f"search-{channel} --limit",
                    max_value=self.MAX_SEMANTIC_LIMIT,
                )
            elif arg.startswith("-"):
                raise PIFSCommandError(f"Unsupported search-{channel} option: {arg}")
            else:
                positionals.append(arg)
            i += 1
        if not positionals:
            raise PIFSCommandError(f"search-{channel} requires a query")
        self._validate_search_positionals(f"search-{channel}", positionals)
        query = positionals[0]
        self._reject_regex_alternation_query(query, f"search-{channel}")
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

    def _bounded_text_artifact(self, target: str, location: str) -> dict[str, Any]:
        if str(location).strip().lower() in {"all", "full", "*"}:
            start, end = 1, self.MAX_TEXT_LINES
        else:
            start, end = self._parse_numeric_range(location, "cat --range")
            self._require_at_most(
                end - start + 1,
                "cat --range line count",
                self.MAX_TEXT_LINES,
            )
        opened = self.filesystem.cat_text_artifact(target, f"{start}-{end}")
        data = self._jsonable(opened)
        total_lines = len(self.filesystem.store.read_text(opened.file_ref).splitlines())
        has_more = int(data.get("end_line") or end) < total_lines
        pagination = {
            "offset_line": start,
            "limit": self.MAX_TEXT_LINES,
            "returned_lines": max(0, int(data.get("end_line") or end) - start + 1),
            "total_lines": total_lines,
            "has_more": has_more,
            "next_range": None,
            "next_command": None,
        }
        if has_more:
            next_start = int(data.get("end_line") or end) + 1
            next_end = min(total_lines, next_start + self.MAX_TEXT_LINES - 1)
            next_range = f"{next_start}-{next_end}"
            pagination["next_range"] = next_range
            pagination["next_command"] = (
                f"cat {shlex.quote(target)} --range {shlex.quote(next_range)}"
            )
            data["text"] = (
                str(data.get("text") or "").rstrip()
                + "\n"
                + self._pagination_footer(
                    "cat --all",
                    f"showing lines {start}-{data.get('end_line')} of {total_lines}",
                    str(pagination["next_command"]),
                )
            ).strip()
        data["pagination"] = pagination
        return data

    def _bounded_node_result(
        self,
        data: dict[str, Any],
        *,
        target: str,
        node_id: str,
    ) -> dict[str, Any]:
        if not isinstance(data, dict) or data.get("available") is False:
            return data
        text = str(data.get("text") or "")
        lines = text.splitlines()
        truncated_by_lines = len(lines) > self.MAX_NODE_TEXT_LINES
        truncated_by_chars = len(text) > self.MAX_NODE_TEXT_CHARS
        if not truncated_by_lines and not truncated_by_chars:
            data["node_pagination"] = {
                "limit_nodes": self.MAX_NODE_IDS,
                "text_truncated": False,
            }
            return data

        selected = "\n".join(lines[: self.MAX_NODE_TEXT_LINES])
        if len(selected) > self.MAX_NODE_TEXT_CHARS:
            selected = selected[: self.MAX_NODE_TEXT_CHARS].rstrip()
        data["text"] = (
            selected.rstrip()
            + "\n"
            + self._pagination_footer(
                "cat --node",
                (
                    f"node text limited to {self.MAX_NODE_TEXT_LINES} lines/"
                    f"{self.MAX_NODE_TEXT_CHARS} chars"
                ),
                f"cat {shlex.quote(target)} --structure",
            )
        ).strip()
        data["node_pagination"] = {
            "limit_nodes": self.MAX_NODE_IDS,
            "line_limit": self.MAX_NODE_TEXT_LINES,
            "char_limit": self.MAX_NODE_TEXT_CHARS,
            "original_lines": len(lines),
            "original_chars": len(text),
            "text_truncated": True,
            "suggested_command": f"cat {shlex.quote(target)} --structure",
            "node_id": node_id,
        }
        return data

    def _attach_structure_next_command(self, data: dict[str, Any], target: str) -> None:
        pagination = data.get("structure_pagination")
        if not isinstance(pagination, dict):
            return
        if pagination.get("has_more") and pagination.get("next_offset") is not None:
            next_command = (
                f"cat {shlex.quote(target)} --structure "
                f"--offset {pagination['next_offset']} --limit {pagination['limit']}"
            )
            pagination["next_command"] = next_command
        else:
            pagination["next_command"] = None

    def _attach_page_next_command(
        self,
        data: dict[str, Any],
        target: str,
        *,
        start: int,
        end: int,
    ) -> None:
        page_count = end - start + 1
        next_command = None
        if page_count == self.MAX_PAGE_SPAN:
            next_start = end + 1
            next_end = next_start + self.MAX_PAGE_SPAN - 1
            next_command = f"cat {shlex.quote(target)} --page {next_start}-{next_end}"
        data["page_pagination"] = {
            "start": start,
            "end": end,
            "returned_pages": page_count,
            "limit": self.MAX_PAGE_SPAN,
            "next_command": next_command,
        }

    @staticmethod
    def _pagination_footer(command: str, reason: str, next_command: str) -> str:
        return (
            f"# output limited by {command}: {reason}. "
            f"Next: {next_command}. If unsure, use cat <target> --structure."
        )

    @staticmethod
    def _parse_node_ids(value: str) -> list[str]:
        return [part.strip() for part in value.split(",") if part.strip()]

    @staticmethod
    def _reject_regex_alternation_query(query: str, command_name: str) -> None:
        if "|" not in str(query):
            return
        raise PIFSCommandError(
            f"{command_name} does not support regex alternation '|'. "
            "Run multiple grep commands or multiple search-summary commands "
            "with one phrase each."
        )

    @staticmethod
    def _validate_search_positionals(command_name: str, positionals: list[str]) -> None:
        if len(positionals) > 2:
            raise PIFSCommandError(
                f"{command_name} accepts one query and an optional folder path. "
                f"Quote multi-word queries, for example: {command_name} "
                '"Federal Reserve" /documents'
            )
        if len(positionals) == 2 and not positionals[1].startswith("/"):
            raise PIFSCommandError(
                f"{command_name} target must be a PIFS folder path like /documents. "
                f"If your query has spaces, quote it, for example: {command_name} "
                '"Federal Reserve" /documents'
            )

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

    def _validate_metadata_field_for_stat(self, field: str) -> None:
        schema = self.filesystem._metadata_schema()
        fields = schema.get("fields", {})
        if field not in fields:
            available = ", ".join(sorted(fields)[:20]) or "(none)"
            raise PIFSCommandError(
                f"Unknown metadata field: {field}. Use stat --schema to inspect fields. "
                f"Available fields include: {available}"
            )

    def _stat_field_row(self, field: str, target: str) -> dict[str, Any]:
        info = self.filesystem._stat(target)
        folder_paths = [
            folder.get("path", "")
            for folder in info.get("folders", [])
            if folder.get("path")
        ]
        row = dict(info)
        row["target"] = target
        row["folder_paths"] = folder_paths
        metadata = info.get("metadata") or {}
        raw_value = metadata.get(field)
        value_text = "" if raw_value is None else str(raw_value)
        row.update(
            {
                "field": field,
                "present": field in metadata,
                "value": raw_value if field in metadata else None,
                "display_target": self._file_target_path(row),
            }
        )
        return row

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
            return json.dumps(
                {
                    "structure": data.get("structure", []),
                    "pagination": data.get("structure_pagination", {}),
                },
                ensure_ascii=False,
                indent=2,
            )
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
                f"{self._file_target_path(item)}:{item['line']}: "
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
                    f"{self._folder_row_path(item['path'])} matched_files={item['matched_files']} "
                    f"files={item.get('file_count', 0)}"
                    if item.get("matched_files")
                    else f"{self._folder_row_path(item['path'])} folders={item.get('children_count', 0)} "
                    f"files={item.get('file_count', 0)}"
                )
                for item in data
            )
        return "\n".join(self._file_row_text(item) for item in data)

    def _folder_row_path(self, path: str) -> str:
        normalized = self._normalize_folder_path(path)
        return "/" if normalized == "/" else f"{normalized}/"

    def _render_stat(self, data: Any) -> str:
        if not isinstance(data, dict):
            return str(data)
        if "fields" in data:
            lines = ["metadata schema:"]
            for name, field in sorted(data["fields"].items()):
                lines.append(f"{name}: {field.get('type', 'string')}")
            return "\n".join(lines)
        if data.get("mode") == "field_values":
            field = data.get("field", "")
            lines = []
            for item in data.get("data", []):
                lines.append(f"{item.get('display_target') or item.get('target')}:")
                value = item.get("value")
                if value is None:
                    lines.append(f"{field}: -")
                else:
                    lines.append(f"{field}: {self._one_line_value(value)}")
            return "\n\n".join(lines)
        if data.get("mode") == "files":
            return "\n\n".join(self._render_stat(item) for item in data.get("data", []))
        lines = [
            f"target: {data.get('target') or data.get('file_ref')}",
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
        metadata_status = data.get("metadata_status") or {}
        if metadata_status:
            lines.append(f"metadata_status: {metadata_status.get('status', '-')}")
            summary_projection = (
                metadata_status.get("projection_indexes", {}).get("summary", {})
            )
            if summary_projection:
                lines.append(
                    f"summary_projection_status: {summary_projection.get('status', '-')}"
                )
        return "\n".join(lines)

    def _file_row_text(self, item: dict[str, Any]) -> str:
        file_ref = item.get("file_ref")
        doc_id = item.get("external_id") or item.get("document_id") or "-"
        title = self._compact_text(item.get("title") or item.get("name") or "", max_chars=80)
        source_path = item.get("source_path") or "-"
        folder_paths = item.get("folder_paths") or self._folder_paths_for_file(file_ref)
        folders = f" folders={','.join(folder_paths)}" if folder_paths else ""
        target = self._file_target_path(item)
        return f"{target} id={doc_id} file_ref={file_ref or '-'} title={title} source={source_path}{folders}".strip()

    def _grep_file_hit_text(self, item: dict[str, Any]) -> str:
        doc_id = item.get("external_id") or "-"
        line = item.get("line") or 1
        target = self._file_target_path(item)
        return (
            f"{target}:{line}: id={doc_id} "
            f"{self._compact_text(item.get('text') or '', max_chars=180)}"
        )

    def _file_target_path(self, item: dict[str, Any]) -> str:
        file_ref = item.get("file_ref")
        title = str(item.get("title") or item.get("name") or "").strip()
        folder_paths = item.get("folder_paths") or []
        folder_path = item.get("folder_path")
        if not folder_paths and folder_path:
            folder_paths = [folder_path]
        if not folder_paths:
            folder_paths = self._folder_paths_for_file(file_ref)
        if folder_paths and title:
            folder = str(folder_paths[0] or "/").rstrip("/")
            return f"{folder}/{title}" if folder else f"/{title}"
        return str(item.get("source_path") or item.get("external_id") or file_ref or "-")

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
            line_number, text = self._first_matching_source_line(path, query)
            hits.append(
                {
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
        file_ref = self.filesystem._resolve_target(target)
        entry = self.filesystem.store.get_file(file_ref)
        matches = []
        for line_number, line in enumerate(self.filesystem.store.read_text(file_ref).splitlines(), 1):
            if self._line_matches(line, query):
                matches.append(
                    {
                        "file_ref": file_ref,
                        "external_id": entry.external_id,
                        "title": entry.title,
                        "source_path": entry.source_path,
                        "folder_paths": self._folder_paths_for_file(file_ref),
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
    def _one_line_value(value: Any) -> str:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return re.sub(r"\s+", " ", str(value or "")).strip()

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
        count = self._require_at_most(
            count,
            "pipe head/tail line count",
            self.MAX_TEXT_LINES,
        )
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
        self._reject_regex_alternation_query(pattern, "pipe grep")
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
        self._require_at_most(end - start + 1, "pipe sed line count", self.MAX_TEXT_LINES)
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

    @classmethod
    def _parse_bounded_int(cls, value: str, label: str, *, max_value: int) -> int:
        parsed = cls._parse_non_negative_int(value, label)
        return cls._require_at_most(parsed, label, max_value)

    @classmethod
    def _require_at_most(cls, value: int, label: str, max_value: int) -> int:
        if value > max_value:
            raise PIFSCommandError(
                f"{label} supports at most {max_value}; requested {value}. "
                "Use a smaller value. If you are unsure where to inspect, "
                "use cat <target> --structure first."
            )
        return value

    @staticmethod
    def _parse_find_maxdepth(value: str | None) -> int:
        if value is None:
            raise PIFSCommandError("find -maxdepth requires an integer >= 0")
        try:
            parsed = int(value)
        except ValueError as exc:
            raise PIFSCommandError("find -maxdepth requires an integer >= 0") from exc
        if parsed < 0:
            raise PIFSCommandError("find -maxdepth requires an integer >= 0")
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
