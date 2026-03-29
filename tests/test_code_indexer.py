"""Tests for CodeIndexer and CodeSearcher."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

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
async def test_max_files():
    indexer = CodeIndexer(max_files=2)
    # index pageindex itself
    index = await indexer.index_directory(str(Path(__file__).parent.parent))
    assert index.total_files <= 2
