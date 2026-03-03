# import anthropic
# from pageindex.providers.base_llm import BaseLLM
# from pageindex.response_schema import LLMResponse

# class AnthropicLLM(BaseLLM):

#     def __init__(self, api_key: str, model: str):
#         self.client = anthropic.Anthropic(api_key=api_key)
#         self.model = model

#     def generate(self, messages, **kwargs):
#         response = self.client.messages.create(
#             model=self.model,
#             messages=messages,
#             max_tokens=kwargs.get("max_tokens", 1024)
#         )

#         return LLMResponse(
#             content=response.content[0].text,
#             finish_reason=response.stop_reason,
#             raw=response
#         )

#     async def generate_async(self, messages, **kwargs):
#         response = await self.client.messages.create(
#             model=self.model,
#             messages=messages,
#             max_tokens=kwargs.get("max_tokens", 1024)
#         )

#         return LLMResponse(
#             content=response.content[0].text,
#             finish_reason=response.stop_reason,
#             raw=response
#         )

from anthropic import Anthropic, AsyncAnthropic
from pageindex.providers.base_llm import BaseLLM


class AnthropicProvider(BaseLLM):

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def generate(self, model, messages, **kwargs):
        response = self.client.messages.create(
            model=model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0)
        )

        # Claude returns a list of content blocks
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return {
            "content": content,
            "finish_reason": response.stop_reason,
            "raw": response
        }

    async def agenerate(self, model, messages, **kwargs):
        async with AsyncAnthropic(api_key=self.client.api_key) as client:
            response = await client.messages.create(
                model=model,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0)
            )

            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            return {
                "content": content,
                "finish_reason": response.stop_reason,
                "raw": response
            }