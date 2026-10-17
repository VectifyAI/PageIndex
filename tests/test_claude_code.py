"""The claude-code/ route — indexing and chat as Claude Code CLI runs on the
user's subscription — over a faked CLI session: no subprocess, no login."""
import asyncio
import time
from types import SimpleNamespace

import pytest

import pageindex.integrations.claude_agent_sdk as cc
from pageindex import PageIndexAPIError, PageIndexLocalClient
from pageindex.local_chat import CHAT_HEADER
from pageindex.local_store import DocStore
from pageindex.utils import (LLMRetriesExhausted, _is_unrecoverable,
                             llm_acompletion, llm_completion)

try:
    from claude_agent_sdk import (AssistantMessage, ResultMessage,
                                  StreamEvent, TextBlock, ToolResultBlock,
                                  ToolUseBlock, UserMessage)
    _HAS_CLAUDE = True
except ImportError:
    _HAS_CLAUDE = False

needs_claude = pytest.mark.skipif(not _HAS_CLAUDE,
                                  reason="claude-agent-sdk not installed")


@pytest.fixture
def chat_client(tmp_path):
    store = str(tmp_path / "store")
    meta = {"id": "pi-doc1", "name": "fruit.pdf", "description": "Fruit notes",
            "status": "completed", "createdAt": "2026-10-17T10:00:00.000000",
            "pageNum": 1, "folderId": None, "metadata": None, "mode": "flash"}
    tree = [{"title": "Fruit", "node_id": "0000", "start_index": 1,
             "end_index": 1, "summary": "apples"}]
    DocStore(store).save_document("pi-doc1", meta, tree,
                                  [{"page_index": 1, "markdown": "Apples"}])
    return PageIndexLocalClient(storage_path=store,
                                chat_model="claude-code/haiku")


def fake_cli(monkeypatch, *messages, account=None, models=None, during=None):
    """Replace the CLI session with one whose handshake reports
    ``account`` as the login and ``models`` as the model list, that records
    what it is sent and replays ``messages``; ``during`` runs as the turn
    starts (a tool call, a delay). Returns the sessions: .prompt (None
    until sent), .options, .closed."""
    sessions = []

    class Session:
        def __init__(self, options):
            self.options, self.prompt, self.closed = options, None, False
            sessions.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            await asyncio.sleep(0)  # the real client awaits its transport's close
            self.closed = True

        async def get_server_info(self):
            return {"account": {"apiProvider": "firstParty", **(account or {})},
                    **({"models": models} if models else {})}

        async def query(self, prompt):
            self.prompt = (prompt if isinstance(prompt, str)
                           else [m async for m in prompt])

        async def receive_response(self):
            if during:
                await during()
            for message in messages:
                yield message
    monkeypatch.setattr(cc, "_claude_client", Session)
    return sessions


def result(text="done", is_error=False, subtype="success", status=None,
           usage=None):
    return ResultMessage(subtype=subtype, duration_ms=1, duration_api_ms=1,
                         is_error=is_error, num_turns=1, session_id="s",
                         result=text, usage=usage, api_error_status=status)


def delta(text):
    return StreamEvent(uuid="u", session_id="s", event={
        "type": "content_block_delta", "index": 0,
        "delta": {"type": "text_delta", "text": text}})


# ── refusals: no CLI run needed ──

def test_standard_mode_refuses_claude_code(tmp_path, sample_pdf):
    client = PageIndexLocalClient(storage_path=str(tmp_path / "s"),
                                  index_model="claude-code/haiku")
    with pytest.raises(PageIndexAPIError, match="flash mode only"):
        client.submit_document(sample_pdf, mode="standard")
    # the classic entry itself, as run_pageindex.py --mode standard
    # reaches it: refused before any run is spent
    from pageindex.page_index_classic import page_index_main
    with pytest.raises(ValueError, match="flash mode only"):
        page_index_main(sample_pdf, SimpleNamespace(
            model="gpt-4.1", summary_model="claude-code/haiku"))


def test_chat_refuses_what_the_cli_cannot_take(chat_client):
    with pytest.raises(PageIndexAPIError, match="takes no temperature"):
        chat_client.chat_completions("Q", temperature=0)
    with pytest.raises(PageIndexAPIError, match="takes no backend"):
        chat_client.chat("Q", backend={"api_key": "k"})
    for protocol in ("responses", "messages"):
        with pytest.raises(PageIndexAPIError, match="answer lane"):
            chat_client.chat("Q", protocol=protocol)
    # an image or tool history: said outright, not sent off to the
    # protocol lanes, which refuse claude-code/ too
    image = [{"type": "text", "text": "What is this?"},
             {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]
    for messages in ([{"role": "user", "content": image}],
                     [{"role": "tool", "tool_call_id": "c", "content": "x"},
                      {"role": "user", "content": "Q"}]):
        with pytest.raises(PageIndexAPIError,
                           match="claude-code/haiku takes text messages only") as err:
            chat_client.chat(messages)
        assert "use an API model with" in str(err.value)
    with pytest.raises(PageIndexAPIError, match="drives the Anthropic API"):
        chat_client.anthropic_runner_config()


@needs_claude
def test_claude_agent_config_keeps_your_claude_code_login(chat_client):
    config = chat_client.claude_agent_config()
    assert config["model"] == "haiku"
    assert config["env"] == {"ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": "",
                             "ANTHROPIC_BASE_URL": ""}


# ── indexing ──

@needs_claude
def test_index_call_runs_isolated_on_the_subscription(monkeypatch):
    sessions = fake_cli(monkeypatch, result('{"summary": "S"}'))
    assert asyncio.run(llm_acompletion("claude-code/haiku", "P",
                                       reasoning_effort="none")) == '{"summary": "S"}'
    assert llm_completion("claude-code/haiku", "Q") == '{"summary": "S"}'
    first, second = sessions
    assert (first.prompt, second.prompt) == ("P", "Q")
    opts = first.options
    assert opts.model == "haiku" and opts.tools == [] and opts.setting_sources == []
    assert opts.strict_mcp_config and opts.max_turns == 1 and not opts.mcp_servers
    assert opts.thinking == {"type": "disabled"} and second.options.thinking is None
    # an @/path in document text must not make the CLI read that file
    assert opts.verbatim_prompts
    # a .env written for API models must not bill or reroute the run
    assert all(opts.env[var] == "" for var in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"))
    assert "no-session-persistence" in opts.extra_args
    assert opts.max_buffer_size > 2**20  # a page image's tool result is one line
    assert first.closed and second.closed


@needs_claude
@pytest.mark.parametrize("status,fatal", [(429, True), (500, True), (400, False)])
def test_index_failures_are_final_except_a_400(monkeypatch, status, fatal):
    sessions = fake_cli(monkeypatch, result("boom", is_error=True, status=status))
    with pytest.raises(LLMRetriesExhausted) as err:
        asyncio.run(llm_acompletion("claude-code/haiku", "P"))
    assert err.value.status_code == status
    assert _is_unrecoverable(err.value) is fatal
    assert sessions[0].closed  # shut down inline, not left to the loop


@needs_claude
def test_missing_login_says_how_to_log_in(monkeypatch):
    fake_cli(monkeypatch,
             AssistantMessage(content=[TextBlock("Not logged in")], model="haiku",
                              error="authentication_failed"),
             result("Not logged in · Please run /login", is_error=True))
    with pytest.raises(LLMRetriesExhausted, match="log in") as err:
        asyncio.run(llm_acompletion("claude-code/haiku", "P"))
    assert err.value.status_code == 401


@needs_claude
@pytest.mark.parametrize("account,why", [
    ({"apiKeySource": "ANTHROPIC_API_KEY"}, "API key"),
    ({"apiProvider": "bedrock"}, "routed to bedrock")])
def test_a_login_off_the_subscription_is_refused_before_any_prompt(
        monkeypatch, account, why):
    sessions = fake_cli(monkeypatch, result("billed"), account=account)
    with pytest.raises(LLMRetriesExhausted, match=why) as err:
        asyncio.run(llm_acompletion("claude-code/haiku", "P"))
    assert err.value.status_code == 403 and _is_unrecoverable(err.value)
    assert sessions[0].prompt is None and sessions[0].closed  # nothing billed


@needs_claude
def test_an_effort_claude_code_does_not_list_is_refused_before_any_prompt(monkeypatch):
    models = [{"value": "haiku", "resolvedModel": "claude-haiku-5-5",
               "supportedEffortLevels": ["low", "high"]},
              {"value": "opus", "supportedEffortLevels": ["low", "max"]},
              {"value": "tiny", "supportsEffort": False}]
    sessions = fake_cli(monkeypatch, result("ok"), models=models)
    # a typo, another vendor's level, a level the model lacks (also by its
    # full id), a typo on a model the CLI doesn't list: the CLI would warn
    # and run at its default effort
    for model, effort, levels in (
            ("haiku", "hgih", "low, high"), ("haiku", "minimal", "low, high"),
            ("haiku", "max", "low, high"), ("claude-haiku-5-5", "max", "low, high"),
            ("claude-sonnet-4-5", "hgih", "low, high, max")):
        with pytest.raises(LLMRetriesExhausted,
                           match=f"{model} takes reasoning_effort none, {levels}, not") as err:
            asyncio.run(llm_acompletion(f"claude-code/{model}", "P", reasoning_effort=effort))
        assert _is_unrecoverable(err.value)  # fatal, not one prompt's miss
    assert all(s.prompt is None and s.closed for s in sessions)  # nothing billed
    # no effort and "none" are never checked; an unlisted model takes any
    # listed model's levels; a model without effort levels drops it
    for model, effort in (("haiku", None), ("haiku", "none"),
                          ("claude-haiku-5-5", "high"), ("claude-sonnet-4-5", "max"),
                          ("tiny", "high")):
        assert asyncio.run(llm_acompletion(f"claude-code/{model}", "P",
                                           reasoning_effort=effort)) == "ok"
    assert sessions[-1].options.effort == "high"


@needs_claude
def test_flash_checks_a_claude_code_summary_effort_before_expand_bills(
        monkeypatch, sample_pdf):
    from pageindex.flash.api import page_index_flash
    sessions = fake_cli(monkeypatch, result("ok"), models=[
        {"value": "haiku", "supportedEffortLevels": ["low", "high"]}])
    with pytest.raises(PageIndexAPIError, match="haiku takes reasoning_effort none, low, high, not"):
        page_index_flash(sample_pdf, summary_model="claude-code/haiku",
                         summary_reasoning_effort="minimal")
    (check,) = sessions  # the check alone: no expand or summary run
    assert check.prompt is None and check.closed
    asyncio.run(cc.claude_code_check("claude-code/haiku", "low"))
    assert sessions[-1].prompt is None and sessions[-1].closed


@needs_claude
def test_index_calls_are_capped(monkeypatch):
    live = peak = 0

    async def turn():
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
    fake_cli(monkeypatch, result("ok"), during=turn)

    async def many():
        return await asyncio.gather(*(llm_acompletion("claude-code/haiku", "P")
                                      for _ in range(20)))
    assert asyncio.run(many()) == ["ok"] * 20
    assert peak == cc.CLAUDE_CODE_CONCURRENCY
    assert cc._GATES == {}  # the cap goes with its last call, not kept per loop


@needs_claude
def test_flash_indexing_on_claude_code(monkeypatch, tmp_path, sample_pdf):
    sessions = fake_cli(monkeypatch, result("A two-page fruit memo."))
    client = PageIndexLocalClient(storage_path=str(tmp_path / "s"),
                                  index_model="claude-code/haiku")
    doc = client.submit_document(sample_pdf)
    assert client.get_document(doc["doc_id"])["description"] == "A two-page fruit memo."
    assert sessions and all(s.options.model == "haiku" for s in sessions)


# ── chat ──

@needs_claude
def test_chat_sends_one_message_with_a_block_per_turn(monkeypatch, chat_client):
    usage = {"input_tokens": 5, "cache_read_input_tokens": 100,
             "cache_creation_input_tokens": 20, "output_tokens": 7}
    sessions = fake_cli(monkeypatch, result("A2", usage=usage))
    history = [{"role": "user", "content": "Q1"},
               {"role": "assistant", "content": "A1"},
               {"role": "user", "content": "Q2"}]
    assert chat_client.chat(history, doc_id="pi-doc1") == "A2"
    (session,) = sessions
    (message,) = session.prompt
    blocks = message["message"]["content"]
    assert "fruit.pdf" in blocks[0]["text"]  # the doc targeting leads
    assert [b["text"] for b in blocks[1:]] == [
        '<turn role="user">\nQ1\n</turn>', '<turn role="assistant">\nA1\n</turn>',
        '<turn role="user">\nQ2\n</turn>']
    assert not any("cache_control" in b for b in blocks)
    opts = session.options
    assert opts.system_prompt.startswith(CHAT_HEADER)
    assert "<turn> blocks" in opts.system_prompt
    assert list(opts.mcp_servers) == ["pageindex"]
    assert opts.allowed_tools == ["mcp__pageindex"] and opts.tools == []
    assert opts.max_turns == 10 and not opts.include_partial_messages

    envelope = chat_client.chat(history, doc_id="pi-doc1", protocol="chat_completions")
    assert envelope["model"] == "haiku"
    assert envelope["usage"]["prompt_tokens"] == 125
    assert envelope["usage"]["prompt_tokens_details"]["cached_tokens"] == 100


@needs_claude
def test_chat_stream_weaves_the_run(monkeypatch, chat_client):
    sessions = fake_cli(
        monkeypatch,
        AssistantMessage(content=[ToolUseBlock(
            id="t1", name="mcp__pageindex__get_document",
            input={"doc_name": "fruit.pdf"})], model="haiku"),
        UserMessage(content=[ToolResultBlock(
            tool_use_id="t1", content=[{"type": "text", "text": "1 page"}])]),
        delta("It has "), delta("1 page."), result("It has 1 page."))
    events = list(chat_client.chat("How long?", doc_id="pi-doc1", stream=True).events)
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "answer", "answer"]
    assert events[0]["name"] == events[1]["name"] == "get_document"
    assert sessions[0].options.include_partial_messages
    text = "".join(chat_client.chat("How long?", doc_id="pi-doc1", stream=True))
    assert text == ('[tool_call] get_document {"doc_name": "fruit.pdf"}\n'
                    "[tool_result] get_document: 1 page\n\nIt has 1 page.")


@needs_claude
def test_chat_completions_stream_chunks(monkeypatch, chat_client):
    fake_cli(monkeypatch, delta("Hi "), delta("there."),
             result("Hi there.", usage={"input_tokens": 3, "output_tokens": 2}))
    chunks = list(chat_client.chat_completions("Q", stream=True,
                                               stream_metadata=True))
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert [c["choices"][0]["delta"]["content"] for c in chunks[1:3]] == [
        "Hi ", "there."]
    assert chunks[3]["choices"][0]["finish_reason"] == "stop"
    assert chunks[4]["choices"] == [] and chunks[4]["usage"]["completion_tokens"] == 2
    assert len(chunks) == 5
    assert "".join(chat_client.chat_completions("Q", stream=True)) == "Hi there."


@needs_claude
@pytest.mark.parametrize("surface", ["chat", "chat_completions"])
def test_an_abandoned_stream_shuts_the_cli_down(monkeypatch, chat_client, surface):
    sessions = fake_cli(monkeypatch, *[delta("x") for _ in range(100)],
                        result("x" * 100))
    stream = getattr(chat_client, surface)("Q", stream=True)
    next(stream)
    stream.close()
    deadline = time.monotonic() + 5
    while not sessions[0].closed and time.monotonic() < deadline:
        time.sleep(0.02)
    assert sessions[0].closed


@needs_claude
def test_a_bare_sdk_exception_reads_as_a_claude_code_failure(monkeypatch, chat_client):
    async def timeout():  # the SDK raises a bare Exception for this one
        raise Exception("Control request timeout: initialize")
    sessions = fake_cli(monkeypatch, result("never"), during=timeout)
    with pytest.raises(PageIndexAPIError, match="Claude Code failed: Control request timeout"):
        chat_client.chat("Q")
    assert sessions[0].closed


@needs_claude
def test_chat_stream_display_elides_image_payloads(monkeypatch, chat_client):
    """A tool's image shows in the display minus its base64 payload (the
    CLI hands it back as a Claude block); the event keeps it whole."""
    image = {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png",
                                         "data": "A" * 8192}}
    fake_cli(monkeypatch,
             AssistantMessage(content=[ToolUseBlock(
                 id="t1", name="mcp__pageindex__get_document_image",
                 input={"image_path": "fig.png"})], model="haiku"),
             UserMessage(content=[ToolResultBlock(tool_use_id="t1",
                                                  content=[image])]),
             result("A chart."))
    text = "".join(chat_client.chat("Show the figure", stream=True))
    assert ('[tool_result] get_document_image: {"type": "image", "source": '
            '{"type": "base64", "media_type": "image/png", "data": "..."}}') in text
    assert "AAAA" not in text
    events = list(chat_client.chat("Show the figure", stream=True).events)
    assert events[1]["output"] == [image]


@needs_claude
@pytest.mark.parametrize("end", ["answer", "max_turns"])
def test_cloud_tool_limits_fail_the_chat(monkeypatch, end):
    """A cloud tool's USAGE_LIMIT_REACHED (402) fails chat() as on the
    bridge lanes, instead of reaching the model as text it answers around
    — and stays the reported cause when the run then fails in the CLI."""
    import json

    import claude_agent_sdk
    import pageindex.mcp_bridge as mcp_bridge
    from pageindex import PageIndexCloudClient

    class Bridge:
        def __init__(self, url, headers):
            pass

        def list_tools(self):
            return [{"name": "get_document", "description": "d",
                     "annotations": {"readOnlyHint": True},
                     "inputSchema": {"type": "object", "properties": {}}}]

        def instructions(self):
            return "GUIDANCE"

        def call_tool(self, name, arguments):
            return [{"type": "text", "text": json.dumps({
                "success": False, "error": "Usage limit reached",
                "errorCode": "USAGE_LIMIT_REACHED"})}], True
    monkeypatch.setattr(mcp_bridge, "McpBridge", Bridge)
    tools = {}

    def server(**kwargs):
        tools.update({tool.name: tool for tool in kwargs["tools"]})
        return {"type": "sdk", "name": kwargs["name"], "instance": None}
    monkeypatch.setattr(claude_agent_sdk, "create_sdk_mcp_server", server)

    async def the_model_calls_the_tool():
        reply = await tools["get_document"].handler({})
        assert reply["is_error"]
    fake_cli(monkeypatch,
             *((delta("I could not "), delta("read it."), result("I could not read it."))
               if end == "answer"
               else (result(None, is_error=True, subtype="error_max_turns"),)),
             during=the_model_calls_the_tool)

    cloud = PageIndexCloudClient(api_key="pi-test-key",
                                 chat_model="claude-code/haiku")
    with pytest.raises(PageIndexAPIError, match="Usage limit reached") as err:
        cloud.chat("Q")
    assert err.value.status_code == 402
    # streamed too, the run stops at the failed tool: no answer around it
    events = []
    with pytest.raises(PageIndexAPIError, match="Usage limit reached"):
        for event in cloud.chat("Q", stream=True).events:
            events.append(event)
    assert not [event for event in events if event["type"] == "answer"]
    # in process, not the relay: the server's annotations (readOnlyHint
    # keeps the read tools parallel) plus the inline cap past ~100k results
    assert tools["get_document"].annotations.model_dump(
        by_alias=True, exclude_none=True) == {"readOnlyHint": True,
                                              "maxResultSizeChars": 500_000}


@needs_claude
def test_chat_max_turns(monkeypatch, chat_client):
    fake_cli(monkeypatch, result(None, is_error=True, subtype="error_max_turns"))
    with pytest.raises(PageIndexAPIError, match=r"max_turns \(3\)"):
        chat_client.chat("Q", max_turns=3)
