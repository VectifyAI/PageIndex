# from groq import Groq, AsyncGroq
# from pageindex.providers.base_llm import BaseLLM
# from pageindex.response_schema import LLMResponse

# class GroqLLM(BaseLLM):

#     def __init__(self, api_key: str, model: str):
#         self.client = Groq(api_key=api_key)
#         self.async_client = AsyncGroq(api_key=api_key)
#         self.model = model

#     def generate(self, messages, **kwargs):
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             **kwargs
#         )
#         return LLMResponse(
#             content=response.choices[0].message.content,
#             finish_reason=response.choices[0].finish_reason,
#             raw=response
#         )

#     async def agenerate(self, model, messages, **kwargs):
#         response = await self.async_client.chat.completions.create(
#             model=self.model,
#             messages=messages,
#             **kwargs
#         )
#         return LLMResponse(
#             content=response.choices[0].message.content,
#             finish_reason=response.choices[0].finish_reason,
#             raw=response
#         )

from groq import Groq, AsyncGroq
from pageindex.providers.base_llm import BaseLLM


class GroqProvider(BaseLLM):

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def generate(self, model, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", 0)
        )

        return {
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "raw": response
        }

    async def agenerate(self, model, messages, **kwargs):
        async with AsyncGroq(api_key=self.client.api_key) as client:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0)
            )

            return {
                "content": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason,
                "raw": response
            }