"""
CodeIndexer — tree-sitter based code indexing for PageIndex.
Extracts symbols (functions, classes, constants) and call graphs from source code.
"""

import hashlib
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())
TS_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())

LANG_MAP = {
    ".py": ("python", PY_LANGUAGE),
    ".ts": ("typescript", TS_LANGUAGE),
    ".tsx": ("tsx", TSX_LANGUAGE),
    ".js": ("typescript", TS_LANGUAGE),
    ".jsx": ("tsx", TSX_LANGUAGE),
}

IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "build", "dist", ".next", ".nuxt", "target", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "egg-info", ".eggs",
}

IGNORE_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "bun.lockb"}


@dataclass
class Symbol:
    name: str
    type: str  # "function" | "class" | "const" | "method"
    line_range: Tuple[int, int]
    signature: str
    docstring: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    callers: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    complexity: int = 1


@dataclass
class ImportInfo:
    source: str
    names: List[str]
    target_file: Optional[str] = None


@dataclass
class FileMetrics:
    total_lines: int
    code_lines: int
    symbol_count: int
    complexity_avg: float = 0.0


@dataclass
class FileIndex:
    repo_name: str
    file_path: str  # relative
    language: str
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    metrics: Optional[FileMetrics] = None
    indexed_at: float = 0.0
    content_hash: str = ""


@dataclass
class RepoIndex:
    repo_name: str
    repo_path: str
    language_composition: Dict[str, int] = field(default_factory=dict)
    files: List[FileIndex] = field(default_factory=list)
    call_graph: Dict[str, List[str]] = field(default_factory=dict)
    total_files: int = 0
    total_symbols: int = 0
    indexed_at: float = 0.0
    metadata: Dict = field(default_factory=dict)


class CodeIndexer:
    def __init__(self, max_files: int = 500):
        self.max_files = max_files
        self._parsers: Dict[str, Parser] = {}

    def _get_parser(self, lang_name: str, language: Language) -> Parser:
        if lang_name not in self._parsers:
            p = Parser(language)
            self._parsers[lang_name] = p
        return self._parsers[lang_name]

    async def index_directory(
        self,
        dir_path: str,
        language_filter: str = None,
        repo_name: str = None,
    ) -> RepoIndex:
        dir_path = os.path.abspath(dir_path)
        if not repo_name:
            repo_name = Path(dir_path).name

        files = self._scan_files(dir_path, language_filter)
        lang_comp: Dict[str, int] = {}
        file_indices: List[FileIndex] = []

        for fpath in files:
            fi = self._parse_file(fpath, dir_path, repo_name)
            if fi:
                file_indices.append(fi)
                lang_comp[fi.language] = lang_comp.get(fi.language, 0) + 1

        call_graph = self._build_call_graph(file_indices)

        # Populate callers from call_graph
        for caller_key, callees in call_graph.items():
            for callee_key in callees:
                for fi in file_indices:
                    for sym in fi.symbols:
                        sym_key = f"{fi.file_path}:{sym.name}"
                        if sym_key == callee_key and caller_key not in sym.callers:
                            sym.callers.append(caller_key)

        total_syms = sum(len(fi.symbols) for fi in file_indices)

        return RepoIndex(
            repo_name=repo_name,
            repo_path=dir_path,
            language_composition=lang_comp,
            files=file_indices,
            call_graph=call_graph,
            total_files=len(file_indices),
            total_symbols=total_syms,
            indexed_at=time.time(),
        )

    def _scan_files(self, dir_path: str, language_filter: str = None) -> List[str]:
        results = []
        ext_filter = None
        if language_filter:
            ext_map = {"python": {".py"}, "py": {".py"}, "typescript": {".ts", ".tsx"}, "ts": {".ts", ".tsx"}, "js": {".js", ".jsx"}}
            ext_filter = ext_map.get(language_filter.lower())

        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
            for fname in sorted(files):
                if fname in IGNORE_FILES:
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in LANG_MAP:
                    continue
                if ext_filter and ext not in ext_filter:
                    continue
                results.append(os.path.join(root, fname))
                if len(results) >= self.max_files:
                    return results
        return results

    def _parse_file(self, file_path: str, dir_path: str, repo_name: str) -> Optional[FileIndex]:
        ext = os.path.splitext(file_path)[1].lower()
        lang_info = LANG_MAP.get(ext)
        if not lang_info:
            return None

        lang_name, language = lang_info
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except Exception:
            return None

        parser = self._get_parser(lang_name, language)
        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)

        rel_path = os.path.relpath(file_path, dir_path).replace("\\", "/")
        symbols = self._extract_symbols(tree, lang_name, source, source_bytes)
        imports = self._extract_imports(tree, lang_name, source)

        lines = source.split("\n")
        total_lines = len(lines)
        code_lines = sum(1 for l in lines if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("//"))
        avg_complexity = sum(s.complexity for s in symbols) / max(len(symbols), 1)

        content_hash = hashlib.md5(source.encode("utf-8")).hexdigest()[:12]

        return FileIndex(
            repo_name=repo_name,
            file_path=rel_path,
            language=lang_name,
            symbols=symbols,
            imports=imports,
            metrics=FileMetrics(
                total_lines=total_lines,
                code_lines=code_lines,
                symbol_count=len(symbols),
                complexity_avg=round(avg_complexity, 2),
            ),
            indexed_at=time.time(),
            content_hash=content_hash,
        )

    def _extract_symbols(self, tree, lang_name: str, source: str, source_bytes: bytes = None) -> List[Symbol]:
        symbols = []
        root = tree.root_node
        # If source_bytes not provided, encode source (fallback for compatibility)
        if source_bytes is None:
            source_bytes = source.encode("utf-8")
        # Store as instance variable for use in recursive calls
        self._source_bytes = source_bytes

        if lang_name == "python":
            self._extract_python_symbols(root, source, symbols)
        elif lang_name in ("typescript", "tsx"):
            self._extract_ts_symbols(root, source, symbols)

        return symbols

    def _extract_python_symbols(self, node, source: str, symbols: List[Symbol], depth: int = 0):
        for child in node.children:
            if child.type == "function_definition" and depth < 2:
                sym = self._python_function(child, source, "function" if depth == 0 else "method")
                if sym:
                    symbols.append(sym)
            elif child.type == "class_definition":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = self._node_text(name_node)
                    line_range = (child.start_point[0] + 1, child.end_point[0] + 1)
                    docstring = self._python_docstring(child, source)
                    symbols.append(Symbol(
                        name=name, type="class", line_range=line_range,
                        signature=f"class {name}", docstring=docstring,
                    ))
                    # Extract methods
                    body = child.child_by_field_name("body")
                    if body:
                        self._extract_python_symbols(body, source, symbols, depth + 1)
            elif child.type == "expression_statement" and depth == 0:
                # Top-level assignments (constants)
                for expr_child in child.children:
                    if expr_child.type == "assignment":
                        left = expr_child.child_by_field_name("left")
                        if left and left.type == "identifier":
                            name = self._node_text(left)
                            if name.isupper():  # Convention: UPPER_CASE = constant
                                symbols.append(Symbol(
                                    name=name, type="const",
                                    line_range=(child.start_point[0] + 1, child.end_point[0] + 1),
                                    signature=self._node_text(child)[:120],
                                ))

    def _python_function(self, node, source: str, sym_type: str) -> Optional[Symbol]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name = self._node_text(name_node)
        params_node = node.child_by_field_name("parameters")
        params = self._node_text(params_node) if params_node else "()"
        ret = node.child_by_field_name("return_type")
        ret_str = f" -> {self._node_text(ret)}" if ret else ""
        sig = f"def {name}{params}{ret_str}"
        docstring = self._python_docstring(node, source)
        line_range = (node.start_point[0] + 1, node.end_point[0] + 1)
        complexity = self._estimate_complexity(node, source)
        deps = self._extract_call_names(node, source)

        return Symbol(
            name=name, type=sym_type, line_range=line_range,
            signature=sig, docstring=docstring, dependencies=deps,
            complexity=complexity,
        )

    def _python_docstring(self, node, source: str) -> Optional[str]:
        body = node.child_by_field_name("body")
        if not body:
            return None
        for child in body.children:
            if child.type == "expression_statement":
                for c in child.children:
                    if c.type == "string":
                        text = self._node_text(c)
                        # Strip triple quotes
                        for q in ('"""', "'''"):
                            if text.startswith(q) and text.endswith(q):
                                return text[3:-3].strip()
                        return text.strip("'\"").strip()
            break
        return None

    def _extract_ts_symbols(self, node, source: str, symbols: List[Symbol], in_class: bool = False):
        for child in node.children:
            if child.type == "function_declaration":
                sym = self._ts_function(child, source, "function")
                if sym:
                    symbols.append(sym)
            elif child.type == "method_definition":
                sym = self._ts_function(child, source, "method")
                if sym:
                    symbols.append(sym)
            elif child.type == "class_declaration":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = self._node_text(name_node)
                    symbols.append(Symbol(
                        name=name, type="class",
                        line_range=(child.start_point[0] + 1, child.end_point[0] + 1),
                        signature=f"class {name}",
                    ))
                    body = child.child_by_field_name("body")
                    if body:
                        self._extract_ts_symbols(body, source, symbols, in_class=True)
            elif child.type in ("export_statement", "export_default_declaration"):
                self._extract_ts_symbols(child, source, symbols, in_class)
            elif child.type == "lexical_declaration":
                # const/let/var
                for decl in child.children:
                    if decl.type == "variable_declarator":
                        name_node = decl.child_by_field_name("name")
                        value_node = decl.child_by_field_name("value")
                        if name_node:
                            name = self._node_text(name_node)
                            if value_node and value_node.type in ("arrow_function", "function_expression"):
                                sym = self._ts_arrow_function(name, child, value_node, source)
                                if sym:
                                    symbols.append(sym)
                            elif name and (name[0].isupper() or name.isupper()):
                                symbols.append(Symbol(
                                    name=name, type="const",
                                    line_range=(child.start_point[0] + 1, child.end_point[0] + 1),
                                    signature=self._node_text(child)[:120],
                                ))

    def _ts_function(self, node, source: str, sym_type: str = "function") -> Optional[Symbol]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name = self._node_text(name_node)
        params_node = node.child_by_field_name("parameters")
        params = self._node_text(params_node) if params_node else "()"
        prefix = "function" if sym_type == "function" else "method"
        sig = f"{prefix} {name}{params}"
        line_range = (node.start_point[0] + 1, node.end_point[0] + 1)
        deps = self._extract_call_names(node, source)
        complexity = self._estimate_complexity(node, source)
        return Symbol(name=name, type=sym_type, line_range=line_range,
                      signature=sig[:200], dependencies=deps, complexity=complexity)

    def _ts_arrow_function(self, name: str, decl_node, arrow_node, source: str) -> Optional[Symbol]:
        params_node = arrow_node.child_by_field_name("parameters")
        params = self._node_text(params_node) if params_node else "()"
        sig = f"const {name} = {params} =>"
        line_range = (decl_node.start_point[0] + 1, decl_node.end_point[0] + 1)
        deps = self._extract_call_names(arrow_node, source)
        complexity = self._estimate_complexity(arrow_node, source)
        return Symbol(name=name, type="function", line_range=line_range,
                      signature=sig[:200], dependencies=deps, complexity=complexity)

    def _extract_imports(self, tree, lang_name: str, source: str) -> List[ImportInfo]:
        imports = []
        root = tree.root_node

        if lang_name == "python":
            for child in root.children:
                if child.type == "import_from_statement":
                    mod_node = child.child_by_field_name("module_name")
                    mod = self._node_text(mod_node) if mod_node else ""
                    names = [self._node_text(n) for n in child.children
                             if n.type == "dotted_name" and n != mod_node]
                    if not names:
                        names = [self._node_text(n) for n in child.children if n.type == "aliased_import"]
                    imports.append(ImportInfo(source=mod, names=names))
                elif child.type == "import_statement":
                    for n in child.children:
                        if n.type == "dotted_name":
                            imports.append(ImportInfo(source=self._node_text(n), names=[]))
        elif lang_name in ("typescript", "tsx"):
            for child in root.children:
                if child.type == "import_statement":
                    source_node = child.child_by_field_name("source")
                    mod = self._node_text(source_node).strip("'\"") if source_node else ""
                    names = []
                    for n in child.children:
                        if n.type == "import_clause":
                            names.extend(self._collect_import_names(n, source))
                    imports.append(ImportInfo(source=mod, names=names))

        return imports

    def _collect_import_names(self, node, source: str) -> List[str]:
        names = []
        if node.type == "identifier":
            names.append(self._node_text(node))
        for child in node.children:
            names.extend(self._collect_import_names(child, source))
        return names

    def _extract_call_names(self, node, source: str) -> List[str]:
        """Extract function call names within a node (dependencies)."""
        calls = set()
        self._walk_calls(node, source, calls)
        return sorted(calls)

    def _walk_calls(self, node, source: str, calls: set):
        if node.type == "call" or node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node:
                text = self._node_text(func_node)
                # Get last identifier (e.g., "self.foo" → "foo", "module.bar" → "bar")
                parts = text.split(".")
                name = parts[-1] if parts else text
                if name and not name.startswith("_") and name not in ("print", "len", "str", "int", "float", "list", "dict", "set", "tuple", "range", "enumerate", "zip", "map", "filter", "sorted", "isinstance", "hasattr", "getattr", "setattr", "super", "type", "open"):
                    calls.add(name)
        for child in node.children:
            self._walk_calls(child, source, calls)

    def _estimate_complexity(self, node, source: str) -> int:
        """Rough cyclomatic complexity: 1 + branches."""
        text = self._node_text(node)
        branch_keywords = ["if ", "elif ", "else:", "for ", "while ", "except ", "case ", "? "]
        return 1 + sum(text.count(kw) for kw in branch_keywords)

    def _build_call_graph(self, files: List[FileIndex]) -> Dict[str, List[str]]:
        """Build cross-file call graph from symbol dependencies."""
        # Map symbol names to their qualified keys
        sym_name_to_keys: Dict[str, List[str]] = {}
        for fi in files:
            for sym in fi.symbols:
                key = f"{fi.file_path}:{sym.name}"
                sym_name_to_keys.setdefault(sym.name, []).append(key)

        graph: Dict[str, List[str]] = {}
        for fi in files:
            for sym in fi.symbols:
                caller_key = f"{fi.file_path}:{sym.name}"
                callees = []
                for dep_name in sym.dependencies:
                    targets = sym_name_to_keys.get(dep_name, [])
                    for t in targets:
                        if t != caller_key:
                            callees.append(t)
                if callees:
                    graph[caller_key] = callees

        return graph

    def _node_text(self, node) -> str:
        """Extract node text using byte offsets (fixes Windows UTF-8 issue)."""
        if not hasattr(self, '_source_bytes') or self._source_bytes is None:
            raise RuntimeError("_source_bytes not initialized in _extract_symbols")
        return self._source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def save_index(self, index: RepoIndex, output_path: str):
        import json
        data = _to_dict(index)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def _to_dict(obj):
    """Recursively convert dataclass instances to dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj
