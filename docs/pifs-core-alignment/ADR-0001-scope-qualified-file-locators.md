# ADR-0001: Scope-Qualified File Locators

Status: Accepted

Date: 2026-07-08

## Context

PIFS agents often navigate through metadata scopes before selecting documents,
for example `/@company/3M`. A failure mode appears when the agent finds the
right scope, then tries to inspect the scope itself as a file. That makes
`cat`, `stat`, or `grep` reject the target after discovery already succeeded.

The fix is an agent-facing path contract, not benchmark-specific prompting.

## Decision

`tree` and `browse` are the file discovery commands. Whenever they return a
file, the returned `path` must be a round-trippable file locator that can be
copied directly into `stat`, `cat`, or `grep`.

A metadata-scoped file is represented as:

```text
<scope path>/<file leaf>
```

Example:

```text
/@company/3M/3M_2018_10K.pdf
```

The file leaf is the leaf exposed by `tree` or `browse`; it is not inferred from
corpus-specific metadata such as `doc_name`.

`tree <metadata value scope>` lists matching file leaves. `tree -L` controls
only traversal depth; large child sets still use pagination.

`stat`, `cat`, and `grep` share one file inspection resolver. If a target is a
scope without a file leaf, such as `/@company/3M`, the command fails and points
the agent back to `tree <scope>` or `browse <scope> "<query>"`. File inspection
commands do not return candidate files or silently choose one.

Agent-facing command output exposes navigation-local locators only. Physical or
debug paths may be kept in internal traces, but not shown to the agent.

## Consequences

- Agents can keep their current navigation context from metadata discovery to
  page reads.
- `tree` and `browse` own file discovery.
- `cat`, `stat`, and `grep` stay simple and share one resolver.
- Tests assert that `tree` and `browse` paths round-trip through `stat`,
  `cat --structure`, and `grep`.

## Rejected

- Letting `cat /@company/3M` resolve a singleton scope automatically.
- Showing physical/debug paths to the agent for convenience.
- Using `doc_name` or any corpus-specific metadata field as the generic file
  resolver key.
- Returning browse hits whose `path` cannot be inspected as a file.
