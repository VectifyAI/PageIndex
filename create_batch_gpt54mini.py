"""Create and submit GPT-5.4 mini batch translation jobs to OpenAI."""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-5.4-mini"
BASE_INSTRUCTIONS = (
    "Translate the following English text from Rudolf Steiner into Hebrew. "
    "Preserve philosophical terminology, anthroposophical terms, and the register of the original."
)


def build_glossary_instructions(glossary_path="glossary.json"):
    with open(glossary_path, encoding="utf-8") as f:
        glossary = json.load(f)
    terms = ", ".join(f"{en}={he}" for en, he in glossary.items())
    return f"{BASE_INSTRUCTIONS}\n\nUse these exact Hebrew terms: {terms}"


def create_batch_file(input_path, output_path, model, instructions):
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            entry = {
                "custom_id": d["custom_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "instructions": instructions,
                    "input": d["body"]["input"],
                    "temperature": 0,
                    "max_output_tokens": 16000,
                },
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    count = sum(1 for _ in open(output_path))
    print(f"Created {output_path} ({count} lines)")


def submit_batch(client, file_path):
    with open(file_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"Uploaded {file_path} → file_id={uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"Submitted batch: id={batch.id}, status={batch.status}")
    return batch


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    source = "batch_input_no_glossary.jsonl"

    glossary_instructions = build_glossary_instructions()
    no_glossary_instructions = BASE_INSTRUCTIONS

    # Create batch files
    create_batch_file(source, "batch_gpt54mini_with_glossary.jsonl", MODEL, glossary_instructions)
    create_batch_file(source, "batch_gpt54mini_no_glossary.jsonl", MODEL, no_glossary_instructions)

    # Submit both batches
    print("\n--- Submitting WITH glossary ---")
    batch_with = submit_batch(client, "batch_gpt54mini_with_glossary.jsonl")

    print("\n--- Submitting WITHOUT glossary ---")
    batch_without = submit_batch(client, "batch_gpt54mini_no_glossary.jsonl")

    print("\n=== Summary ===")
    print(f"With glossary:    batch_id={batch_with.id}")
    print(f"Without glossary: batch_id={batch_without.id}")
    print("\nMonitor with:")
    print(f"  python3 -c \"from openai import OpenAI; c=OpenAI(); print(c.batches.retrieve('{batch_with.id}').status)\"")
    print(f"  python3 -c \"from openai import OpenAI; c=OpenAI(); print(c.batches.retrieve('{batch_without.id}').status)\"")


if __name__ == "__main__":
    main()
