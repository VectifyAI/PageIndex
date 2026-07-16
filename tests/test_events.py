from pageindex.events import QueryEvent
from pageindex.backend.protocol import AgentTools


def test_query_event():
    event = QueryEvent(type="text_delta", data="hello")
    assert event.type == "text_delta"
    assert event.data == "hello"


def test_query_event_types():
    for t in ["reasoning", "tool_call", "tool_result", "text_delta", "text_done"]:
        event = QueryEvent(type=t, data="test")
        assert event.type == t


def test_agent_tools_default_empty():
    tools = AgentTools()
    assert tools.function_tools == []
    assert tools.mcp_servers == []


def test_agent_tools_with_values():
    tools = AgentTools(function_tools=["tool1"], mcp_servers=["server1"])
    assert len(tools.function_tools) == 1
    assert len(tools.mcp_servers) == 1
