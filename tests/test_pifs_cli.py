import os
from pathlib import Path


class FakeFileSystem:
    def __init__(self, workspace):
        self.workspace = Path(workspace)


def test_cli_passthrough_invokes_pifs_command_executor(monkeypatch, capsys, tmp_path):
    from pageindex.filesystem import cli

    workspace = tmp_path / "workspace"
    executor_instances = []

    class FakeExecutor:
        def __init__(self, filesystem, *, json_output=False):
            self.filesystem = filesystem
            self.json_output = json_output
            self.commands = []
            executor_instances.append(self)

        def execute(self, command):
            self.commands.append(command)
            return f"executed:{command}"

    monkeypatch.setattr(cli, "PageIndexFileSystem", FakeFileSystem)
    monkeypatch.setattr(cli, "PIFSCommandExecutor", FakeExecutor)

    status = cli.main(["--workspace", str(workspace), "ls", "/documents", "--json"])

    assert status == 0
    assert capsys.readouterr().out == "executed:ls /documents\n"
    assert len(executor_instances) == 1
    assert executor_instances[0].filesystem.workspace == workspace
    assert executor_instances[0].json_output is True
    assert executor_instances[0].commands == ["ls /documents"]


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
