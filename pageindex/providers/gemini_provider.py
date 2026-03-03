from google import genai
from pageindex.providers.base_llm import BaseLLM
from google.genai import types


class GeminiProvider(BaseLLM):

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def _convert_messages(self, messages):
        """
        Convert OpenAI-style messages into Gemini contents format.
        """
        contents = []

        for msg in messages:
            contents.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            })

        return contents

    def generate(self, model, messages, **kwargs):
        contents = self._convert_messages(messages)

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=2512
            )
           
        )

        return {
            "content": response.text,
            "finish_reason": getattr(response, "finish_reason", None),
            "raw": response
        }

    async def agenerate(self, model, messages, **kwargs):
        contents = self._convert_messages(messages)

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=2500
            )
        )

        return {
            "content": response.text,
            "finish_reason": getattr(response, "finish_reason", None),
            "raw": response
        }