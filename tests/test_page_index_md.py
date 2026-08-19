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


if __name__ == "__main__":
    unittest.main()


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
            env = {k: v for k, v in os.environ.items()
                   if k not in ("OPENAI_API_KEY", "CHATGPT_API_KEY")}
            env["PYTHONPATH"] = str(script.parent)
            res = subprocess.run(
                [sys.executable, str(script), "--md_path", str(md)],
                capture_output=True, cwd=tmp, env=env, timeout=180)
            self.assertEqual(res.returncode, 0, res.stderr.decode())
            out = Path(tmp) / "results" / "notes_structure.json"
            self.assertTrue(out.exists(), res.stdout.decode())
            json.loads(out.read_text())
