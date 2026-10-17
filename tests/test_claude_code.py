"""The claude-code/ route — indexing and chat as Claude Code CLI runs on the
user's subscription — over a faked CLI run: no subprocess, no login."""
import asyncio

import pytest

import pageindex.integrations.claude_agent_sdk as cc
from pageindex import PageIndexAPIError, PageIndexLocalClient
from pageindex.local_chat import CHAT_HEADER
from pageindex.local_store import DocStore
from pageindex.utils import (LLMRetriesExhausted, _is_unrecoverable,
                             llm_acompletion, llm_completion)

try:
    from claude_agent_sdk import (AssistantMessage, ResultMessage,
                                  StreamEvent, SystemMessage, TextBlock,
                                  ToolResultBlock, ToolUseBlock, UserMessage)
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


def fake_cli(monkeypatch, *messages):
    """Replace the CLI run: record each run's prompt and options, then
    replay ``messages``."""
    runs = []

    async def run(prompt, options):
        sent = prompt if isinstance(prompt, str) else [m async for m in prompt]
        runs.append((sent, options))
        for message in messages:
            yield message
    monkeypatch.setattr(cc, "_claude_query", run)
    return runs


def init(source="none"):
    return SystemMessage(subtype="init", data={"apiKeySource": source})


def result(text="done", is_error=False, subtype="success", status=None,
           usage=None):
    return ResultMessage(subtype=subtype, duration_ms=1, duration_api_ms=1,
                         is_error=is_error, num_turns=1, session_id="s",
                         result=text, usage=usage, api_error_status=status)


# ── refusals: no CLI run needed ──

def test_standard_mode_refuses_claude_code(tmp_path, sample_pdf):
    client = PageIndexLocalClient(storage_path=str(tmp_path / "s"),
                                  index_model="claude-code/haiku")
    with pytest.raises(PageIndexAPIError, match="flash mode only"):
        client.submit_document(sample_pdf, mode="standard")


def test_chat_refuses_what_the_cli_cannot_take(chat_client):
    with pytest.raises(PageIndexAPIError, match="takes no temperature"):
        chat_client.chat_completions("Q", temperature=0)
    with pytest.raises(PageIndexAPIError, match="takes no backend"):
        chat_client.chat("Q", backend={"api_key": "k"})
    for protocol in ("responses", "messages"):
        with pytest.raises(PageIndexAPIError, match="answer lane"):
            chat_client.chat("Q", protocol=protocol)


# ── indexing ──

@needs_claude
def test_index_call_runs_isolated_on_the_subscription(monkeypatch):
    runs = fake_cli(monkeypatch, init(), result('{"summary": "S"}'))
    assert asyncio.run(llm_acompletion("claude-code/haiku", "P",
                                       reasoning_effort="none")) == '{"summary": "S"}'
    assert llm_completion("claude-code/haiku", "Q") == '{"summary": "S"}'
    (prompt, opts), (prompt2, opts2) = runs
    assert (prompt, prompt2) == ("P", "Q")
    assert opts.model == "haiku" and opts.tools == [] and opts.setting_sources == []
    assert opts.strict_mcp_config and opts.max_turns == 1 and not opts.mcp_servers
    assert opts.thinking == {"type": "disabled"} and opts2.thinking is None
    assert opts.env["ANTHROPIC_API_KEY"] == ""
    assert "no-session-persistence" in opts.extra_args


@needs_claude
@pytest.mark.parametrize("status,fatal", [(429, True), (500, True), (400, False)])
def test_index_failures_are_final_except_a_400(monkeypatch, status, fatal):
    fake_cli(monkeypatch, init(), result("boom", is_error=True, status=status))
    with pytest.raises(LLMRetriesExhausted) as err:
        asyncio.run(llm_acompletion("claude-code/haiku", "P"))
    assert err.value.status_code == status
    assert _is_unrecoverable(err.value) is fatal


@needs_claude
def test_missing_login_says_how_to_log_in(monkeypatch):
    fake_cli(monkeypatch, init(),
             AssistantMessage(content=[TextBlock("Not logged in")], model="haiku",
                              error="authentication_failed"),
             result("Not logged in · Please run /login", is_error=True))
    with pytest.raises(LLMRetriesExhausted, match="log in") as err:
        asyncio.run(llm_acompletion("claude-code/haiku", "P"))
    assert err.value.status_code == 401


@needs_claude
def test_an_api_key_login_is_refused(monkeypatch):
    fake_cli(monkeypatch, init("ANTHROPIC_API_KEY"), result("billed"))
    with pytest.raises(LLMRetriesExhausted, match="API key") as err:
        asyncio.run(llm_acompletion("claude-code/haiku", "P"))
    assert _is_unrecoverable(err.value)


@needs_claude
def test_index_calls_are_capped(monkeypatch):
    live = peak = 0

    async def run(prompt, options):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        yield result("ok")
    monkeypatch.setattr(cc, "_claude_query", run)

    async def many():
        return await asyncio.gather(*(llm_acompletion("claude-code/haiku", "P")
                                      for _ in range(20)))
    assert asyncio.run(many()) == ["ok"] * 20
    assert peak == cc.CLAUDE_CODE_CONCURRENCY


@needs_claude
def test_flash_indexing_on_claude_code(monkeypatch, tmp_path, sample_pdf):
    runs = fake_cli(monkeypatch, init(), result("A two-page fruit memo."))
    client = PageIndexLocalClient(storage_path=str(tmp_path / "s"),
                                  index_model="claude-code/haiku")
    doc = client.submit_document(sample_pdf)
    assert client.get_document(doc["doc_id"])["description"] == "A two-page fruit memo."
    assert runs and all(opts.model == "haiku" for _, opts in runs)


# ── chat ──

@needs_claude
def test_chat_sends_one_message_with_a_block_per_turn(monkeypatch, chat_client):
    usage = {"input_tokens": 5, "cache_read_input_tokens": 100,
             "cache_creation_input_tokens": 20, "output_tokens": 7}
    runs = fake_cli(monkeypatch, init(), result("A2", usage=usage))
    history = [{"role": "user", "content": "Q1"},
               {"role": "assistant", "content": "A1"},
               {"role": "user", "content": "Q2"}]
    assert chat_client.chat(history, doc_id="pi-doc1") == "A2"
    ((message,), opts), = runs
    blocks = message["message"]["content"]
    assert "fruit.pdf" in blocks[0]["text"]  # the doc targeting leads
    assert [b["text"] for b in blocks[1:]] == [
        '<turn role="user">\nQ1\n</turn>', '<turn role="assistant">\nA1\n</turn>',
        '<turn role="user">\nQ2\n</turn>']
    assert not any("cache_control" in b for b in blocks)
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
    def delta(text):
        return StreamEvent(uuid="u", session_id="s", event={
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text}})
    runs = fake_cli(
        monkeypatch, init(),
        AssistantMessage(content=[ToolUseBlock(
            id="t1", name="mcp__pageindex__get_document",
            input={"doc_name": "fruit.pdf"})], model="haiku"),
        UserMessage(content=[ToolResultBlock(
            tool_use_id="t1", content=[{"type": "text", "text": "1 page"}])]),
        delta("It has "), delta("1 page."), result("It has 1 page."))
    events = list(chat_client.chat("How long?", doc_id="pi-doc1", stream=True).events)
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "answer", "answer"]
    assert events[0]["name"] == events[1]["name"] == "get_document"
    assert runs[0][1].include_partial_messages
    text = "".join(chat_client.chat("How long?", doc_id="pi-doc1", stream=True))
    assert text == ('[tool_call] get_document {"doc_name": "fruit.pdf"}\n'
                    "[tool_result] get_document: 1 page\n\nIt has 1 page.")


@needs_claude
def test_cloud_tool_limits_fail_the_chat(monkeypatch):
    """A cloud tool's USAGE_LIMIT_REACHED (402) fails chat() as on the
    bridge lanes, instead of reaching the model as text it answers around."""
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

    async def cli(prompt, options):
        [m async for m in prompt]
        reply = await tools["get_document"].handler({})  # the model's call
        assert reply["is_error"]
        yield init()
        yield result("I could not read the document.")
    monkeypatch.setattr(cc, "_claude_query", cli)

    cloud = PageIndexCloudClient(api_key="pi-test-key",
                                 chat_model="claude-code/haiku")
    with pytest.raises(PageIndexAPIError, match="Usage limit reached") as err:
        cloud.chat("Q")
    assert err.value.status_code == 402
    # in process, not the relay: the server's annotations (readOnlyHint
    # keeps the read tools parallel) plus the inline cap past ~100k results
    assert tools["get_document"].annotations.model_dump(
        by_alias=True, exclude_none=True) == {"readOnlyHint": True,
                                              "maxResultSizeChars": 200_000}


@needs_claude
def test_chat_max_turns(monkeypatch, chat_client):
    fake_cli(monkeypatch, init(),
             result(None, is_error=True, subtype="error_max_turns"))
    with pytest.raises(PageIndexAPIError, match=r"max_turns \(3\)"):
        chat_client.chat("Q", max_turns=3)
