import  openai
from pageindex.providers.base_llm import BaseLLM

class OpenAIProvider(BaseLLM):

    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)

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
        async with openai.AsyncOpenAI(api_key=self.client.api_key) as client:
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