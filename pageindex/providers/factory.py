# llm/factory.py

from pageindex.providers.open_router_provider import OpenRouterProvider
from pageindex.providers.badrock_provider import BedrockProvider
from pageindex.providers.groq_provider import GroqProvider
from pageindex.providers.anthropic_provider import AnthropicProvider
from pageindex.providers.gemini_provider import GeminiProvider
from pageindex.providers.open_ai_provider import OpenAIProvider


class LLMFactory:

    @staticmethod
    def create(provider: str, api_key: str):

        if provider == "openai":
            return OpenAIProvider(api_key)

        elif provider == "gemini":
            return GeminiProvider(api_key)
        elif provider =="anthropic":
            return AnthropicProvider(api_key)
        elif provider =='groq':
            return GroqProvider(api_key)
        elif provider =='aws-badrock':
            return BedrockProvider(api_key)
        elif provider == 'open-router':
            return OpenRouterProvider(api_key)
        else:
            raise ValueError("Unsupported provider")


