import asyncio
import http.server
import json
import select
import socket
import tempfile
import threading
import time
from pathlib import Path
import unittest

from pageindex.page_index_md import extract_nodes_from_markdown


class ExtractNodesFromMarkdownTest(unittest.TestCase):
    def test_skips_bold_heading_with_only_whitespace(self):
        nodes, _ = extract_nodes_from_markdown("**   **\n**Valid heading**")

        self.assertEqual(
            nodes,
            [
                {
                    "node_title": "Valid heading",
                    "line_num": 2,
                    "level": 1,
                }
            ],
        )

class _MarkdownProviderServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, mode):
        super().__init__(("127.0.0.1", 0), _MarkdownProviderHandler)
        self.mode = mode
        self.connections = []
        self.received = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()

class _MarkdownProviderHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        failing = "# Failure" in request["messages"][-1]["content"]
        with self.server.lock:
            self.server.connections.append(self.connection)
            if len(self.server.connections) == 2:
                self.server.received.set()
        if not self.server.received.wait(5):
            raise AssertionError("Summary requests did not overlap")
        status = 200
        if self.server.mode == "failure" and failing:
            status = 404
            body = {"error": {"message": "fixture rejected model", "type": "invalid_request_error", "code": "model_not_found"}}
        else:
            if self.server.mode != "success":
                self.server.release.wait(10)
            body = {"id": "fixture", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Summary"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(body).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

async def _wait_markdown_provider(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.01)
    return True

def _markdown_provider_peer_open(connection):
    if connection.fileno() < 0:
        return False
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        return not readable or connection.recv(1, socket.MSG_PEEK) != b""
    except OSError:
        return False

async def _native_markdown_summaries(mode):
    from pageindex import md_to_tree
    from pageindex.utils import _llm_backend
    server = _MarkdownProviderServer(mode)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    before = asyncio.all_tasks()
    token = _llm_backend.set({"api_base": f"http://127.0.0.1:{server.server_port}/v1", "api_key": "fixture-only"})
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "note.md"
            path.write_text("# Failure\n\nFirst body.\n\n# Slow\n\nSecond body.\n")
            task = asyncio.create_task(md_to_tree(path, if_add_node_summary="yes", summary_token_threshold=0, model="gpt-4o-mini"))
            assert await _wait_markdown_provider(server.received.is_set, 8), "Actual provider HTTP requests must run concurrently"
            if mode == "cancel":
                task.cancel()
            error = None
            try:
                result = await asyncio.wait_for(task, 0.05) if mode == "timeout" else await task
            except BaseException as exc:
                error = type(exc).__name__

            await _wait_markdown_provider(lambda: not any(_markdown_provider_peer_open(c) for c in server.connections), 0.3)
            pending = [t for t in asyncio.all_tasks() - before if not t.done() and "get_node_summary" in t.get_coro().__qualname__]
            out = {"mode": mode, "requests": len(server.connections), "public_error": error, "pending_owned_summary_tasks": len(pending), "open_provider_peers": sum(_markdown_provider_peer_open(c) for c in server.connections)}
            if mode == "success":
                assert all(node["summary"] == "Summary" for node in result["structure"])
            # The proof owns cleanup even on unchanged source; no request is left alive.
            for own in pending:
                own.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            return out
    finally:
        _llm_backend.reset(token)
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join()


class MarkdownSummaryTaskCleanupTest(unittest.TestCase):
    def check_cleanup(self, mode, expected_error):
        result = asyncio.run(_native_markdown_summaries(mode))
        self.assertEqual(result["requests"], 2)  # both native requests overlapped
        self.assertEqual(result["public_error"], expected_error)
        self.assertEqual(result["pending_owned_summary_tasks"], 0)
        self.assertEqual(result["open_provider_peers"], 0)

    def test_success_preserves_concurrent_summaries(self):
        self.check_cleanup("success", None)

    def test_fatal_provider_error_closes_sibling_request_before_return(self):
        self.check_cleanup("failure", "NotFoundError")

    def test_caller_cancellation_closes_owned_requests(self):
        self.check_cleanup("cancel", "CancelledError")

    def test_caller_timeout_closes_owned_requests(self):
        self.check_cleanup("timeout", "TimeoutError")


class MarkdownCliTest(unittest.TestCase):
    def test_md_cli_runs_without_llm_or_key(self):
        """--md_path with no flags makes zero LLM calls: config.yaml's PDF
        summary default must not leak in, so the run completes without any
        provider key and writes the structure file."""
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        script = Path(__file__).resolve().parent.parent / "run_pageindex.py"
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp) / "notes.md"
            md.write_text("# Title\n\nIntro.\n\n## Section\n\nBody.\n")
            env = dict(os.environ)
            # present-but-empty beats deletion: utils' load_dotenv() does
            # not override existing vars, so the repo .env key stays out
            env["OPENAI_API_KEY"] = ""
            env["CHATGPT_API_KEY"] = ""
            env["PYTHONPATH"] = str(script.parent)
            res = subprocess.run(
                [sys.executable, str(script), "--md_path", str(md)],
                capture_output=True, cwd=tmp, env=env, timeout=180)
            self.assertEqual(res.returncode, 0, res.stderr.decode())
            out = Path(tmp) / "results" / "notes_structure.json"
            self.assertTrue(out.exists(), res.stdout.decode())
            json.loads(out.read_text())

    def test_md_cli_summary_model_drives_summary_calls(self):
        """--summary-model owns the markdown summary lane: node summaries
        and the doc description bill it, never the index model given
        alongside — the same chain the flag's help promises on PDFs."""
        import json
        import os
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        script = Path(__file__).resolve().parent.parent / "run_pageindex.py"
        driver = (
            "import json, runpy, sys\n"
            "import pageindex.utils as U\n"
            "seen = []\n"
            "async def fake_acompletion(model, prompt, **kw):\n"
            "    seen.append(model)\n"
            "    return 'node summary'\n"
            "def fake_completion(model, prompt, **kw):\n"
            "    seen.append(model)\n"
            "    return 'doc description'\n"
            "U.llm_acompletion = fake_acompletion\n"
            "U.llm_completion = fake_completion\n"
            "target = sys.argv[1]\n"
            "sys.argv = [target] + sys.argv[2:]\n"
            "runpy.run_path(target, run_name='__main__')\n"
            "print('MODELS_SEEN=' + json.dumps(sorted(set(seen))))\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp) / "notes.md"
            md.write_text("# Title\n\nIntro.\n\n## Section\n\nBody.\n")
            drv = Path(tmp) / "driver.py"
            drv.write_text(driver)
            env = dict(os.environ)
            env["PYTHONPATH"] = str(script.parent)
            res = subprocess.run(
                [sys.executable, str(drv), str(script),
                 "--md_path", str(md),
                 "--if-add-node-summary", "yes",
                 "--if-add-doc-description", "yes",
                 "--summary-token-threshold", "1",
                 "--summary-model", "SUMMARY-SENTINEL",
                 "--index-model", "INDEX-DECOY"],
                capture_output=True, cwd=tmp, env=env, timeout=180)
            self.assertEqual(res.returncode, 0, res.stderr.decode())
            line = next(l for l in res.stdout.decode().splitlines()
                        if l.startswith("MODELS_SEEN="))
            self.assertEqual(json.loads(line[len("MODELS_SEEN="):]),
                             ["SUMMARY-SENTINEL"], res.stdout.decode())


if __name__ == "__main__":
    unittest.main()
