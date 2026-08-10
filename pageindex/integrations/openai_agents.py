"""OpenAI Agents SDK adapter for the Agent(tools=...) slot.

Cloud clients default to the full live tool set as plain FunctionTools via
the MCP bridge; pass hosted=True to use a single HostedMCPTool instead
(the model connects to the PageIndex cloud MCP server from OpenAI's side).
Local clients get the in-process tools wrapped as FunctionTools.
"""
from __future__ import annotations

from ..errors import PageIndexAPIError


def build_openai_tools(client, include_management: bool = False,
                       hosted: bool = False) -> list:
    try:
        from agents import HostedMCPTool, function_tool
    except ImportError as exc:
        raise PageIndexAPIError(
            "as_openai_tools requires the OpenAI Agents SDK — "
            "pip install openai-agents (or pip install 'pageindex[openai]')."
        ) from exc
    if getattr(client, "api_key", None) and hosted:
        # Same gate as the in-process path, enforced by OpenAI: tools the
        # server annotates read-only run freely, everything else goes
        # through the Responses API approval flow.
        require_approval = ("never" if include_management
                            else {"never": {"read_only": True}})
        return [HostedMCPTool(tool_config={
            "type": "mcp",
            "server_label": "pageindex",
            "server_url": f"{client.BASE_URL}/mcp",
            "headers": {"Authorization": f"Bearer {client.api_key}"},
            "require_approval": require_approval,
        })]
    from ..agent_tools import build_agent_tools
    return [function_tool(tool)
            for tool in build_agent_tools(client, include_management)]
