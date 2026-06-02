# Incremental Markdown Update Demo

`incremental_update_demo.py` — index a Markdown doc, then incrementally update
it so only changed sections are re-summarized.

Set an API key first (e.g. `export OPENAI_API_KEY=...`) and configure the model
in `pageindex/config.yaml`.

```bash
python examples/incremental_update_demo.py
```

## How it works

- `client.get_doc_id_by_path(path)` returns the `doc_id` already indexed for a
  file path, or `None`.
- First run for a path → `client.index(path)` builds the tree fresh.
- Later runs → `client.update(doc_id)` re-summarizes **only** the sections whose
  content hash changed; unchanged sections reuse their cached summary. No diff →
  `{"status": "unchanged"}` (zero LLM work).

Re-indexing the same file path reuses its `doc_id` and overwrites the same
workspace JSON instead of creating a duplicate document.

The script copies the sample (`documents/sample.md`) into a stable workspace
path so re-runs reuse the same `doc_id`.

## Workspace

Indexed documents persist under `examples/workspace/` as `<doc_id>.json` plus a
`_meta.json` index. Generated files there are throwaway test artifacts.
