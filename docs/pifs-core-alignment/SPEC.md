# PIFS Core Alignment Spec

## Purpose

Align PageIndex FileSystem (PIFS) core around a small BashLike command surface
where file discovery and file inspection have separate responsibilities.

## Scope

In scope:

- Keep PIFS BashLike commands as the agent-facing interface.
- Make `tree` and `browse` the only file discovery commands.
- Make `stat`, `cat`, and `grep` require a concrete file locator.
- Ensure metadata scope file locators round-trip through all file inspection
  commands.

Out of scope:

- Replacing PIFS with typed MCP tools.
- Adding benchmark-specific prompts, result rows, or scoring logic to PIFS core.
- Adding a new search/find command.

## Command Surface

| Command | Role |
|---|---|
| `tree` | Structural orientation and file leaf discovery |
| `browse` | Relevance-ranked document discovery |
| `stat` | Single document identity and metadata |
| `cat` | Structure and page content reads |
| `grep` | Single-document lexical evidence fallback |

`ls` remains only as an exact alias for `tree -L 1`.

## `tree`

Syntax:

```text
tree <scope> [-L depth] [--page N]
ls <scope>
```

Rules:

- `tree` returns folders, files, metadata axes, or metadata values.
- Ordinary folder scopes list directly visible file leaves.
- Metadata value scopes such as `/@company/3M` list matching file leaves.
- File leaves returned by `tree` are round-trippable locators for `stat`,
  `cat`, and `grep`.
- `-L` controls traversal depth only.
- Pagination is one global page over displayed children in stable order:
  folders, files, then metadata axes or metadata values.

## `browse`

Syntax:

```text
browse <scope> "<query>" [--page N] [--where JSON] [-R]
```

Rules:

- Query is required.
- Results are relevance-ranked documents inside the scope.
- Each result `path` is a round-trippable file locator.
- Metadata-scoped browse returns navigation-local locators, for example
  `/@company/3M/3M_2018_10K.pdf`.
- Browse output does not expose physical/debug paths to the agent.

## File Inspection Commands

`stat`, `cat`, and `grep` inspect exactly one file locator:

```text
stat <file>
cat <file> --structure
cat <file> --page N[-M]
grep <query> <file>
```

Rules:

- These commands share one file inspection resolver.
- Scope-only targets fail and point the agent back to `tree <scope>` or
  `browse <scope> "<query>"`.
- File inspection commands do not discover files, return candidate lists, or
  silently select singleton scopes.
