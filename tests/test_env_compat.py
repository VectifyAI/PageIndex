"""CHATGPT_API_KEY must keep working as an alias for OPENAI_API_KEY (backward
compat carried over from the pre-SDK pageindex.utils; PR #272 review).

The alias runs at import time in pageindex/__init__.py, so each case runs in a
fresh subprocess with a controlled environment. cwd is a temp dir so load_dotenv
can't pick up the repo's own .env and skew the result."""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_PRINT_OPENAI = "import pageindex, os; print(os.environ.get('OPENAI_API_KEY', ''))"


def _run(tmp_path, **overrides):
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENAI_API_KEY", "CHATGPT_API_KEY")}
    env["PYTHONPATH"] = str(REPO)
    env.update(overrides)
    r = subprocess.run(
        [sys.executable, "-c", _PRINT_OPENAI],
        env=env, cwd=str(tmp_path), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_chatgpt_api_key_aliases_openai(tmp_path):
    # Only CHATGPT_API_KEY set -> OPENAI_API_KEY gets filled from it.
    assert _run(tmp_path, CHATGPT_API_KEY="sk-alias-123") == "sk-alias-123"


def test_existing_openai_api_key_is_not_overwritten(tmp_path):
    # Both set -> the real OPENAI_API_KEY wins; the alias must not clobber it.
    assert _run(tmp_path, OPENAI_API_KEY="sk-real", CHATGPT_API_KEY="sk-alias") == "sk-real"
