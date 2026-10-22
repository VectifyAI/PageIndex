"""Framework adapters for the agent tools layer.

These modules import their target frameworks lazily, at call time, so
importing pageindex never loads them. openai-agents is a dependency; the
anthropic and claude-agent-sdk extras are optional.
"""
