import builtins
import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_pifs_config(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))


def pifs_config_path(tmp_path):
    return tmp_path / "xdg-config" / "pageindex" / "pifs.json"


def test_pifs_config_file_overrides_default_location(monkeypatch, tmp_path):
    from pageindex.filesystem import cli

    default_path = pifs_config_path(tmp_path)
    override_path = tmp_path / "custom-pifs.json"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_text(
        json.dumps({"workspace": str(tmp_path / "default-workspace")}),
        encoding="utf-8",
    )
    override_path.write_text(
        json.dumps({"workspace": str(tmp_path / "override-workspace")}),
        encoding="utf-8",
    )

    monkeypatch.setenv("PIFS_CONFIG_FILE", str(override_path))

    assert cli._configured_workspace() == str(tmp_path / "override-workspace")


class FakeFileSystem:
    def __init__(self, workspace, **kwargs):
        self.workspace = Path(workspace)
        self.kwargs = kwargs


def test_cli_workspace_does_not_eagerly_open_projection(monkeypatch, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)

    filesystem = cli._filesystem_from_workspace(str(workspace))

    assert filesystem.workspace == workspace
    assert filesystem.kwargs.get("summary_projection_embedding_model") is None


def test_cli_workspace_without_projection_index_does_not_require_sqlite_vec(
    monkeypatch, tmp_path
):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    real_import = builtins.__import__

    monkeypatch.delitem(sys.modules, "pageindex.filesystem.semantic_projection", raising=False)
    monkeypatch.delitem(sys.modules, "pageindex.filesystem.semantic_index", raising=False)
    monkeypatch.delitem(sys.modules, "sqlite_vec", raising=False)

    def block_sqlite_vec(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] == "sqlite_vec":
            raise ModuleNotFoundError("No module named 'sqlite_vec'", name="sqlite_vec")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_sqlite_vec)

    filesystem = cli._filesystem_from_workspace(str(workspace))

    assert filesystem.workspace == workspace


def test_cli_workspace_uses_embedding_config(monkeypatch, tmp_path):
    from pageindex.filesystem import cli

    config_path = pifs_config_path(tmp_path)
    workspace = tmp_path / "workspace"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "workspace": str(workspace),
                "embedding_model": "gemini-embedding-2-preview",
                "embedding_dimensions": 3072,
                "embedding_timeout": 12.5,
                "embedding_base_url": "https://example.invalid/openai/",
                "embedding_api_key": "config-key",
            }
        ),
        encoding="utf-8",
    )

    class ConfiguredFileSystem:
        def __init__(self, workspace, **kwargs):
            self.workspace = Path(workspace)
            self.kwargs = kwargs

    monkeypatch.setenv("PIFS_EMBEDDING_API_KEY", "ignored-env-key")
    monkeypatch.setenv("OPENAI_API_KEY", "ignored-openai-key")
    monkeypatch.setenv("PIFS_EMBEDDING_BASE_URL", "https://ignored.invalid/")
    monkeypatch.setattr(cli, "PageIndexFileSystem", ConfiguredFileSystem)

    filesystem = cli._filesystem_from_workspace(str(workspace))

    assert filesystem.workspace == workspace
    assert filesystem.kwargs == {
        "summary_projection_embedding_model": "gemini-embedding-2-preview",
        "summary_projection_embedding_dimensions": 3072,
        "summary_projection_embedding_timeout": 12.5,
        "summary_projection_embedding_base_url": "https://example.invalid/openai/",
        "summary_projection_embedding_api_key": "ignored-env-key",
    }
    assert os.environ["PIFS_EMBEDDING_API_KEY"] == "ignored-env-key"
    assert os.environ["PIFS_EMBEDDING_BASE_URL"] == "https://ignored.invalid/"


@pytest.mark.parametrize(
    ("config_key", "pifs_key", "openai_key", "expected"),
    [
        ("config-key", None, None, "config-key"),
        ("config-key", "pifs-key", "openai-key", "pifs-key"),
        ("config-key", None, "openai-key", "config-key"),
        (None, None, "openai-key", "openai-key"),
    ],
)
def test_cli_embedding_api_key_precedence(
    config_key, pifs_key, openai_key, expected, monkeypatch, tmp_path
):
    from pageindex.filesystem import cli

    config_path = pifs_config_path(tmp_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"workspace": str(tmp_path / "workspace")}
    if config_key is not None:
        config["embedding_api_key"] = config_key
    config_path.write_text(json.dumps(config), encoding="utf-8")
    if pifs_key is None:
        monkeypatch.delenv("PIFS_EMBEDDING_API_KEY", raising=False)
    else:
        monkeypatch.setenv("PIFS_EMBEDDING_API_KEY", pifs_key)
    if openai_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)

    filesystem = cli._filesystem_from_workspace(str(tmp_path / "workspace"))

    assert filesystem.kwargs["summary_projection_embedding_api_key"] == expected


def test_cli_passthrough_invokes_pifs_command_executor(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    executor_instances = []

    class FakeExecutor:
        def __init__(self, filesystem):
            self.filesystem = filesystem
            self.commands = []
            executor_instances.append(self)

        def execute(self, command):
            self.commands.append(command)
            return f"executed:{command}"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "PIFSCommandExecutor", FakeExecutor)

    status = cli.main(["--workspace", str(workspace), "tree", "/documents", "-L", "1"])

    assert status == 0
    assert capsys.readouterr().out == "executed:tree /documents -L 1\n"
    assert len(executor_instances) == 1
    assert executor_instances[0].filesystem.workspace == workspace
    assert executor_instances[0].commands == ["tree /documents -L 1"]


def test_cli_passthrough_returns_nonzero_for_failed_json_envelope(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"

    class FakeExecutor:
        def __init__(self, filesystem):
            self.filesystem = filesystem

        def execute(self, command):
            return json.dumps(
                {"success": False, "error": {"message": "bad"}, "next_steps": []}
            )

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "PIFSCommandExecutor", FakeExecutor)

    status = cli.main(["--workspace", str(workspace), "find", "/documents"])

    assert status == 2
    assert json.loads(capsys.readouterr().out)["success"] is False


def test_cli_set_workspace_persists_default(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    config_path = pifs_config_path(tmp_path)
    workspace = tmp_path / "workspace"

    status = cli.main(["set", "workspace", str(workspace)])

    assert status == 0
    output = capsys.readouterr().out
    assert f"workspace: {workspace}" in output
    assert f"config: {config_path}" in output
    assert config_path.read_text(encoding="utf-8") == (
        '{\n  "workspace": "' + str(workspace) + '"\n}\n'
    )


def test_cli_setmeta_replaces_document_metadata(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    calls = []

    class FakeSetMetaFileSystem(FakeFileSystem):
        def set_metadata(self, target, metadata, *, clear=False):
            calls.append((self.workspace, target, metadata, clear))
            return {
                "path": "/documents/report.md",
                "file_ref": "file_report",
                "metadata": metadata,
            }

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeSetMetaFileSystem)

    status = cli.main(
        [
            "--workspace",
            str(workspace),
            "setmeta",
            "/documents/report.md",
            '{"ticker":"AAPL"}',
        ]
    )

    assert status == 0
    assert calls == [
        (workspace, "/documents/report.md", {"ticker": "AAPL"}, False)
    ]
    assert json.loads(capsys.readouterr().out) == {
        "document_id": None,
        "metadata": {"ticker": "AAPL"},
        "metadata_status": {},
        "path": "/documents/report.md",
        "status": None,
        "title": None,
    }


def test_cli_setmeta_clear_uses_empty_object(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    calls = []

    class FakeSetMetaFileSystem(FakeFileSystem):
        def set_metadata(self, target, metadata, *, clear=False):
            calls.append((self.workspace, target, metadata, clear))
            return {
                "path": "/documents/report.md",
                "file_ref": "file_report",
                "metadata": metadata,
            }

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeSetMetaFileSystem)

    status = cli.main(
        ["--workspace", str(workspace), "setmeta", "--clear", "/documents/report.md"]
    )

    assert status == 0
    assert calls == [(workspace, "/documents/report.md", {}, True)]
    assert json.loads(capsys.readouterr().out) == {
        "document_id": None,
        "metadata": {},
        "metadata_status": {},
        "path": "/documents/report.md",
        "status": None,
        "title": None,
    }


def test_cli_passthrough_uses_configured_workspace(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    config_path = pifs_config_path(tmp_path)
    workspace = tmp_path / "workspace"
    executor_instances = []
    monkeypatch.delenv("PIFS_WORKSPACE", raising=False)

    class FakeExecutor:
        def __init__(self, filesystem):
            self.filesystem = filesystem
            self.commands = []
            executor_instances.append(self)

        def execute(self, command):
            self.commands.append(command)
            return f"executed:{command}"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "PIFSCommandExecutor", FakeExecutor)

    assert cli.main(["set", "workspace", str(workspace)]) == 0
    capsys.readouterr()

    status = cli.main(["tree", "/documents", "-L", "1"])

    assert status == 0
    assert capsys.readouterr().out == "executed:tree /documents -L 1\n"
    assert executor_instances[0].filesystem.workspace == workspace


def test_cli_ask_invokes_agent_with_question(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    agent_calls = []

    def fake_run_pifs_agent(filesystem, question, **kwargs):
        agent_calls.append((filesystem, question, kwargs))
        return "agent answer"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "run_pifs_agent", fake_run_pifs_agent)

    status = cli.main(
        [
            "ask",
            "--workspace",
            str(workspace),
            "--model",
            "test-model",
            "--stream-mode",
            "off",
            "--max-turns",
            "7",
            "--max-seconds",
            "3.5",
            "--reasoning-effort",
            "low",
            "--reasoning-summary",
            "concise",
            "What",
            "is",
            "inside?",
        ]
    )

    assert status == 0
    assert capsys.readouterr().out == "agent answer\n"
    filesystem, question, kwargs = agent_calls[0]
    assert filesystem.workspace == workspace
    assert question == "What is inside?"
    assert kwargs == {
        "model": "test-model",
        "stream_mode": "off",
        "max_turns": 7,
        "max_seconds": 3.5,
        "reasoning_effort": "low",
        "reasoning_summary": "concise",
    }


def test_cli_ask_defaults_to_global_agent_model(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    agent_calls = []
    monkeypatch.delenv("PIFS_AGENT_MODEL", raising=False)
    monkeypatch.delenv("PIFS_MODEL", raising=False)

    def fake_run_pifs_agent(filesystem, question, **kwargs):
        agent_calls.append(kwargs)
        return "agent answer"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "run_pifs_agent", fake_run_pifs_agent)

    status = cli.main(["ask", "--workspace", str(workspace), "What?"])

    assert status == 0
    assert capsys.readouterr().out == "agent answer\n"
    assert agent_calls[0]["model"] == "gpt-5.4"


def test_cli_ask_loads_env_file_before_running_agent(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    agent_keys = []

    def fake_run_pifs_agent(filesystem, question, **kwargs):
        agent_keys.append(os.environ.get("OPENAI_API_KEY"))
        return "agent answer"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "run_pifs_agent", fake_run_pifs_agent)

    status = cli.main(
        [
            "ask",
            "--workspace",
            str(workspace),
            "--env-file",
            str(env_file),
            "What",
            "is",
            "inside?",
        ]
    )

    assert status == 0
    assert capsys.readouterr().out == "agent answer\n"
    assert agent_keys == ["from-dotenv"]


def test_cli_chat_runs_one_question_and_exits(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    inputs = iter(["", "Summarize the workspace", "exit"])
    session_instances = []
    session_questions = []

    class FakeSession:
        def __init__(self, filesystem, **kwargs):
            self.filesystem = filesystem
            self.kwargs = kwargs
            session_instances.append(self)

        def run(self, question):
            session_questions.append((self, question))
            return f"answer:{question}"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "PIFSAgentSession", FakeSession)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    status = cli.main(["chat", "--workspace", str(workspace), "--model", "test-model"])

    assert status == 0
    assert capsys.readouterr().out == ""
    assert len(session_instances) == 1
    assert session_instances[0].filesystem.workspace == workspace
    assert session_questions == [(session_instances[0], "Summarize the workspace")]
    assert session_instances[0].kwargs["model"] == "test-model"
    assert session_instances[0].kwargs["stream_mode"] == "all"


def test_cli_chat_sanitizes_control_input(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    inputs = iter(["\x12", "he\x7fllo\x1b[A", "exit"])
    agent_calls = []

    class FakeSession:
        def __init__(self, filesystem, **kwargs):
            pass

        def run(self, question):
            agent_calls.append(question)
            return f"answer:{question}"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "PIFSAgentSession", FakeSession)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    status = cli.main(["chat", "--workspace", str(workspace), "--stream-mode", "off"])

    assert status == 0
    assert agent_calls == ["hllo"]
    assert capsys.readouterr().out == "answer:hllo\n"


def test_cli_ask_does_not_reprint_streamed_agent_output(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"

    def fake_run_pifs_agent(filesystem, question, **kwargs):
        print("streamed answer")
        return "returned answer"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "run_pifs_agent", fake_run_pifs_agent)

    status = cli.main(
        [
            "ask",
            "--workspace",
            str(workspace),
            "--stream-mode",
            "all",
            "What",
            "is",
            "inside?",
        ]
    )

    assert status == 0
    assert capsys.readouterr().out == "streamed answer\n"


def test_cli_ask_prints_final_answer_with_tool_stream(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"

    def fake_run_pifs_agent(filesystem, question, **kwargs):
        print("[tool stream]")
        return "FOUND 2 documents"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "run_pifs_agent", fake_run_pifs_agent)

    status = cli.main(
        [
            "ask",
            "--workspace",
            str(workspace),
            "--stream-mode",
            "tools",
            "What",
            "is",
            "inside?",
        ]
    )

    assert status == 0
    assert capsys.readouterr().out == "[tool stream]\nFOUND 2 documents\n"


def test_cli_chat_stream_mode_can_be_overridden(monkeypatch, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    inputs = iter(["Summarize the workspace", "exit"])
    session_kwargs = []

    class FakeSession:
        def __init__(self, filesystem, **kwargs):
            session_kwargs.append(kwargs)

        def run(self, question):
            return f"answer:{question}"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "PIFSAgentSession", FakeSession)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    status = cli.main(
        [
            "chat",
            "--workspace",
            str(workspace),
            "--stream-mode",
            "tools",
        ]
    )

    assert status == 0
    assert session_kwargs[0]["stream_mode"] == "tools"


def test_cli_chat_reuses_one_agent_session_for_multiple_questions(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    inputs = iter(["first", "second", "exit"])
    sessions = []

    class FakeSession:
        def __init__(self, filesystem, **kwargs):
            self.questions = []
            sessions.append(self)

        def run(self, question):
            self.questions.append(question)
            return f"answer:{question}"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "PIFSAgentSession", FakeSession)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    status = cli.main(["chat", "--workspace", str(workspace), "--stream-mode", "off"])

    assert status == 0
    assert len(sessions) == 1
    assert sessions[0].questions == ["first", "second"]
    assert capsys.readouterr().out == "answer:first\nanswer:second\n"
