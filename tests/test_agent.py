from pageindex.agent import AgentRunner, OPEN_SYSTEM_PROMPT, SCOPED_SYSTEM_PROMPT
from pageindex.backend.protocol import AgentTools


def test_agent_runner_init():
    tools = AgentTools(function_tools=["mock_tool"])
    runner = AgentRunner(tools=tools, model="gpt-4o")
    assert runner._model == "gpt-4o"


def test_open_prompt_has_tool_instructions():
    assert "list_documents" in OPEN_SYSTEM_PROMPT
    assert "get_document_structure" in OPEN_SYSTEM_PROMPT
    assert "get_page_content" in OPEN_SYSTEM_PROMPT


def test_scoped_prompt_omits_list_documents():
    assert "list_documents" not in SCOPED_SYSTEM_PROMPT
    assert "get_document_structure" in SCOPED_SYSTEM_PROMPT
    assert "get_page_content" in SCOPED_SYSTEM_PROMPT


def test_run_works_inside_running_event_loop(monkeypatch):
    """Regression: Runner.run_sync raises RuntimeError under a running loop
    (Jupyter/FastAPI); AgentRunner.run must offload to a worker thread."""
    import asyncio
    agents = __import__("agents")

    class FakeResult:
        final_output = "ok"

    async def fake_run(agent, question):
        return FakeResult()

    def fail_run_sync(agent, question):
        raise AssertionError("run_sync must not be called inside a running loop")

    monkeypatch.setattr(agents.Runner, "run", fake_run)
    monkeypatch.setattr(agents.Runner, "run_sync", fail_run_sync)
    monkeypatch.setattr(agents, "Agent", lambda **kwargs: object())

    runner = AgentRunner(tools=AgentTools(function_tools=[]), model="gpt-4o")

    async def main():
        return runner.run("question")  # sync call from inside a running loop

    assert asyncio.run(main()) == "ok"
