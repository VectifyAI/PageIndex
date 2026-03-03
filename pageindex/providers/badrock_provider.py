import json
import boto3
from pageindex.providers.base_llm import BaseLLM


class BedrockProvider(BaseLLM):

    def __init__(self, aws_access_key, aws_secret_key, region):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )

    def generate(self, model, messages, **kwargs):
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0)
        })

        response = self.client.invoke_model(
            modelId=model,
            body=body
        )

        response_body = json.loads(response["body"].read())

        content = ""
        for block in response_body["content"]:
            if block["type"] == "text":
                content += block["text"]

        return {
            "content": content,
            "finish_reason": response_body.get("stop_reason"),
            "raw": response_body
        }

    async def agenerate(self, *args, **kwargs):
        raise NotImplementedError("Async not implemented for Bedrock (boto3 is sync)")