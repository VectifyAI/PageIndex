# Project Setup

## API Keys Check

At the start of every session, check if `.env` exists and contains both `RUNPOD_API_KEY` and `OPENAI_API_KEY`. If either is missing, immediately ask the user to provide the missing key(s) before proceeding with any other work. Once provided, write them to `.env` in this format:

```
RUNPOD_API_KEY=<key>
OPENAI_API_KEY=<key>
```

Then configure runpodctl if the Runpod key was just added:
```bash
runpodctl config --apiKey <RUNPOD_API_KEY>
```

## Tools

- `runpodctl` — Manage Runpod GPU pods, serverless endpoints, templates, volumes
- `openai` Python SDK — For OpenAI API interactions (fine-tuning, file management, etc.)
