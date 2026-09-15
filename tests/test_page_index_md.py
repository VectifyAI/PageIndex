import unittest
from unittest.mock import AsyncMock, patch

from pageindex.page_index_md import extract_nodes_from_markdown, get_node_summary


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


class GetNodeSummaryTest(unittest.IsolatedAsyncioTestCase):
    async def test_below_threshold_uses_node_text_without_calling_llm(self):
        node = {"text": "short section"}

        with patch("pageindex.page_index_md.count_tokens", return_value=199), patch(
            "pageindex.page_index_md.generate_node_summary", new_callable=AsyncMock
        ) as generate_summary:
            summary = await get_node_summary(node, summary_token_threshold=200)

        self.assertEqual(summary, node["text"])
        generate_summary.assert_not_awaited()

    async def test_threshold_boundary_generates_a_summary(self):
        node = {"text": "boundary section"}

        with patch("pageindex.page_index_md.count_tokens", return_value=200), patch(
            "pageindex.page_index_md.generate_node_summary",
            new_callable=AsyncMock,
            return_value="generated summary",
        ) as generate_summary:
            summary = await get_node_summary(node, summary_token_threshold=200)

        self.assertEqual(summary, "generated summary")
        generate_summary.assert_awaited_once_with(node, model=None)


if __name__ == "__main__":
    unittest.main()
