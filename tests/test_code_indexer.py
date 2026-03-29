"""Tests for CodeIndexer and CodeSearcher."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from unittest.mock import patch

from pageindex.code_indexer import CodeIndexer, LANG_MAP
from pageindex.code_searcher import CodeSearcher


SAMPLE_PY = '''
"""Sample module."""

import os
from pathlib import Path

MAX_SIZE = 1024

class Engine:
    """Core engine class."""

    def __init__(self, name: str):
        self.name = name

    def run(self, steps: int = 10) -> bool:
        """Execute the engine."""
        for i in range(steps):
            self.process(i)
        return True

    def process(self, step: int):
        pass

def helper(x: int) -> str:
    """Convert to string."""
    return str(x)

def main():
    e = Engine("test")
    e.run()
    helper(42)
'''

SAMPLE_TS = '''
import { Server } from "mcp";
import { readFile } from "fs/promises";

export const VERSION = "1.0.0";

export class Handler {
    async handle(request: Request): Promise<Response> {
        const data = await readFile("config.json");
        return new Response(data);
    }
}

export function createServer(port: number): Server {
    return new Server(port);
}

const processData = (input: string) => {
    return input.trim();
};
'''


@pytest.fixture
def sample_repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "engine.py").write_text(SAMPLE_PY)
    (tmp_path / "src" / "server.ts").write_text(SAMPLE_TS)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.py").write_text("x = 1")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.py").write_text("y = 2")
    return tmp_path


@pytest.mark.asyncio
async def test_index_directory(sample_repo):
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(sample_repo), repo_name="test-repo")

    assert index.repo_name == "test-repo"
    assert index.total_files == 2
    assert index.total_symbols > 0
    assert "python" in index.language_composition
    assert "typescript" in index.language_composition


@pytest.mark.asyncio
async def test_ignore_dirs(sample_repo):
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(sample_repo))
    paths = [fi.file_path for fi in index.files]
    assert not any("node_modules" in p for p in paths)
    assert not any(".git" in p for p in paths)


@pytest.mark.asyncio
async def test_python_symbols(sample_repo):
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(sample_repo), language_filter="python")
    assert index.total_files == 1
    py_file = index.files[0]
    names = {s.name for s in py_file.symbols}
    assert "Engine" in names
    assert "helper" in names
    assert "main" in names
    assert "MAX_SIZE" in names

    # Check method extraction
    run_sym = next(s for s in py_file.symbols if s.name == "run")
    assert run_sym.type == "method"
    assert run_sym.docstring == "Execute the engine."


@pytest.mark.asyncio
async def test_ts_symbols(sample_repo):
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(sample_repo), language_filter="ts")
    assert index.total_files == 1
    ts_file = index.files[0]
    names = {s.name for s in ts_file.symbols}
    assert "Handler" in names
    assert "createServer" in names
    assert "VERSION" in names


@pytest.mark.asyncio
async def test_call_graph(sample_repo):
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(sample_repo))
    # main() calls Engine and helper
    main_key = "src/engine.py:main"
    assert main_key in index.call_graph
    callees = index.call_graph[main_key]
    callee_names = [c.split(":")[-1] for c in callees]
    assert "Engine" in callee_names or "run" in callee_names or "helper" in callee_names


@pytest.mark.asyncio
async def test_save_and_load(sample_repo, tmp_path):
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(sample_repo), repo_name="save-test")
    out = tmp_path / "results" / "save-test_code_index.json"
    out.parent.mkdir(exist_ok=True)
    indexer.save_index(index, str(out))

    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    assert data["repo_name"] == "save-test"
    assert data["total_files"] == 2

    # Test searcher can load it
    searcher = CodeSearcher(results_dir=str(out.parent))
    indices = searcher._load_indices()
    assert len(indices) == 1


@pytest.mark.asyncio
async def test_searcher_definition(sample_repo, tmp_path):
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(sample_repo), repo_name="def-test")
    out = tmp_path / "results" / "def-test_code_index.json"
    out.parent.mkdir(exist_ok=True)
    indexer.save_index(index, str(out))

    searcher = CodeSearcher(results_dir=str(out.parent))
    result = await searcher.search("helper", mode="definition")
    assert "helper" in result
    assert "def-test" in result


@pytest.mark.asyncio
async def test_searcher_impact(sample_repo, tmp_path):
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(sample_repo), repo_name="impact-test")
    out = tmp_path / "results" / "impact-test_code_index.json"
    out.parent.mkdir(exist_ok=True)
    indexer.save_index(index, str(out))

    searcher = CodeSearcher(results_dir=str(out.parent))
    result = await searcher.search("helper", mode="impact")
    assert "Impact analysis" in result


@pytest.mark.asyncio
async def test_language_filter(sample_repo):
    indexer = CodeIndexer()
    py_index = await indexer.index_directory(str(sample_repo), language_filter="python")
    assert all(f.language == "python" for f in py_index.files)
    ts_index = await indexer.index_directory(str(sample_repo), language_filter="ts")
    assert all(f.language in ("typescript", "tsx") for f in ts_index.files)


@pytest.mark.asyncio
async def test_python_async_symbol_extraction(tmp_path):
    """Async functions and complex type annotations must not corrupt symbol names."""
    async_py = '''
from typing import Optional, Dict, List

class AsyncWorker:
    async def fetch(self, url: str, timeout: int = 30) -> Optional[Dict]:
        pass

    async def process_batch(self, items: List[str]) -> List[Dict]:
        for item in items:
            r = await self.fetch(item)
        return []

async def run_all(workers: List[AsyncWorker]) -> None:
    pass
'''
    (tmp_path / "worker.py").write_text(async_py)
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(tmp_path), language_filter="python")
    py_file = index.files[0]
    names = {s.name for s in py_file.symbols}
    assert "AsyncWorker" in names
    assert "fetch" in names
    assert "process_batch" in names
    assert "run_all" in names
    # All names must be valid Python identifiers (no corruption)
    for sym in py_file.symbols:
        assert sym.name and sym.name.isidentifier(), f"Corrupt symbol name: {sym.name!r}"


@pytest.mark.asyncio
async def test_ts_class_method_call_graph(tmp_path):
    """TS class methods should appear in call graph when they call this.otherMethod()."""
    ts_with_method_calls = '''
export class Service {
    async run(): Promise<void> {
        const result = this.compute(42);
        this.log(result);
    }

    compute(n: number): number {
        return n * 2;
    }

    log(msg: any): void {
        console.log(msg);
    }
}
'''
    (tmp_path / "service.ts").write_text(ts_with_method_calls)
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(tmp_path), repo_name="ts-method-test")

    names = {s.name for fi in index.files for s in fi.symbols}
    assert "run" in names
    assert "compute" in names
    assert "log" in names

    # run() calls compute and log — should appear in call graph
    run_key = "service.ts:run"
    assert run_key in index.call_graph, f"run not in call_graph. Keys: {list(index.call_graph.keys())}"
    callee_names = [c.split(":")[-1] for c in index.call_graph[run_key]]
    assert "compute" in callee_names or "log" in callee_names, f"Expected compute/log in callees: {callee_names}"


@pytest.mark.asyncio
async def test_max_files():
    indexer = CodeIndexer(max_files=2)
    # index pageindex itself
    index = await indexer.index_directory(str(Path(__file__).parent.parent))
    assert index.total_files <= 2


@pytest.mark.asyncio
async def test_ts_static_method_call_graph(tmp_path):
    """Static methods called as ClassName.method() must appear in call graph."""
    ts_static = '''
export class MathHelper {
    static compute(n: number): number {
        return n * 2;
    }
}

export class Runner {
    run(): void {
        const result = MathHelper.compute(10);
        console.log(result);
    }
}
'''
    (tmp_path / "static.ts").write_text(ts_static)
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(tmp_path), repo_name="ts-static-test")

    names = {s.name for fi in index.files for s in fi.symbols}
    assert "compute" in names
    assert "run" in names

    # run() calls compute — should appear in call graph
    run_key = "static.ts:run"
    assert run_key in index.call_graph, f"run not in call_graph. Keys: {list(index.call_graph.keys())}"
    callee_names = [c.split(":")[-1] for c in index.call_graph[run_key]]
    assert "compute" in callee_names, f"Expected compute in callees: {callee_names}"


@pytest.mark.asyncio
async def test_ts_super_method_call(tmp_path):
    """super.method() calls in subclass must be tracked in call graph."""
    ts_inherit = '''
export class Base {
    init(): void {
        console.log("base init");
    }
}

export class Child extends Base {
    init(): void {
        super.init();
        console.log("child init");
    }
}
'''
    (tmp_path / "inherit.ts").write_text(ts_inherit)
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(tmp_path), repo_name="ts-super-test")

    # Child.init() should have dependency on Base.init()
    child_file = index.files[0]
    child_init = next((s for s in child_file.symbols if s.name == "init" and s.line_range[0] > 5), None)
    assert child_init is not None, "Child.init() not found"
    assert "init" in child_init.dependencies, f"super.init() not in dependencies: {child_init.dependencies}"


@pytest.mark.asyncio
async def test_ts_promise_chain_callback(tmp_path):
    """Function calls inside .then()/.catch() callbacks must be tracked."""
    ts_promise = '''
export class Fetcher {
    process(data: string): string {
        return data.trim();
    }

    fetch(url: string): void {
        fetch(url)
            .then((response) => {
                const text = this.process(response.text());
                return text;
            })
            .catch((err) => {
                this.handleError(err);
            });
    }

    handleError(err: Error): void {
        console.error(err);
    }
}
'''
    (tmp_path / "fetcher.ts").write_text(ts_promise)
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(tmp_path), repo_name="ts-promise-test")

    names = {s.name for fi in index.files for s in fi.symbols}
    assert "fetch" in names
    assert "process" in names

    fetch_sym = next((s for fi in index.files for s in fi.symbols if s.name == "fetch"), None)
    assert fetch_sym is not None
    # process or handleError should be in dependencies (called inside callbacks)
    assert "process" in fetch_sym.dependencies or "handleError" in fetch_sym.dependencies, \
        f"Expected process/handleError in deps: {fetch_sym.dependencies}"


@pytest.mark.asyncio
async def test_ts_arrow_function_callback(tmp_path):
    """Calls inside arrow function arguments to HOFs (map, forEach) must be tracked."""
    ts_arrow_hof = '''
export class Processor {
    transform(item: string): string {
        return item.toUpperCase();
    }

    processAll(items: string[]): string[] {
        return items.map((item) => this.transform(item));
    }
}
'''
    (tmp_path / "processor.ts").write_text(ts_arrow_hof)
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(tmp_path), repo_name="ts-arrow-test")

    names = {s.name for fi in index.files for s in fi.symbols}
    assert "processAll" in names
    assert "transform" in names

    process_all = next((s for fi in index.files for s in fi.symbols if s.name == "processAll"), None)
    assert process_all is not None
    assert "transform" in process_all.dependencies, \
        f"Expected transform in deps: {process_all.dependencies}"


@pytest.mark.asyncio
async def test_relevance_search_with_fallback(sample_repo, tmp_path):
    """Fallback chain: first model raises exception, second model succeeds."""
    indexer = CodeIndexer()
    index = await indexer.index_directory(str(sample_repo), repo_name="fallback-test")
    out = tmp_path / "results" / "fallback-test_code_index.json"
    out.parent.mkdir(exist_ok=True)
    indexer.save_index(index, str(out))

    searcher = CodeSearcher(results_dir=str(out.parent))

    call_count = {"n": 0}

    def mock_api(model, prompt, api_key=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("primary model unavailable")
        # Second call (fallback model) succeeds
        return "[0]"

    with patch("pageindex.code_searcher.ChatGPT_API", side_effect=mock_api):
        result = await searcher.search("engine run", mode="relevance")

    assert call_count["n"] >= 2, "Fallback was not attempted"
    assert "Code search" in result
