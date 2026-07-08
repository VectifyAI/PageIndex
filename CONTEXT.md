# PageIndex FileSystem Context

This context defines the product language for PageIndex FileSystem (PIFS), an
agent-facing virtual filesystem for navigating document workspaces.

## Language

**PIFS Browse**:
A relevance-ranked view of files inside one PIFS scope for a required query.
It returns files only. Browse result `path` values are file locators and must
round-trip through `stat`, `cat`, and `grep`.
_Avoid_: search-summary, semantic-grep, display-only browse paths

**PIFS File Locator**:
An idempotent path-like locator returned by PIFS commands that identifies one
file for inspection commands.
_Avoid_: storage_uri, source_path, physical/debug path

**Scope-Qualified File Locator**:
A file locator formed by appending a concrete file leaf to the current scope,
for example `/@company/3M/3M_2018_10K.pdf`. The file leaf comes from `tree` or
`browse`; file commands do not infer it from corpus-specific metadata.
_Avoid_: treating metadata-qualified file paths as folders

**Round-Trippable File Locator**:
A locator returned by `tree` or `browse` that can be copied directly into
`stat`, `cat`, or `grep`. If visible leaves collide inside a scope, discovery
commands must return a disambiguated locator; inspection commands do not guess.
_Avoid_: same-looking file leaves, command-specific resolver heuristics

**Scope-Only File Command Error**:
The error returned when `stat`, `cat`, or `grep` receives a scope path without a
file leaf. The command fails and points the agent back to `tree <scope>` or
`browse <scope> "<query>"`.
_Avoid_: silently selecting one file, returning candidate lists from cat/stat/grep

**PIFS File Inspection Resolver**:
The shared resolver used by `stat`, `cat`, and `grep`. A scope-qualified file
locator that works for one file inspection command must work for all three.
_Avoid_: command-specific path parsing

**PIFS Tree File Leaf**:
A concrete file child exposed by `tree` when a scope contains visible files.
Ordinary folder scopes list directly visible file leaves; metadata value scopes
list matching file leaves.
_Avoid_: using cat to discover files

**PIFS Tree Depth**:
The `tree -L` option controls traversal depth only. It does not increase page
size or bypass pagination.
_Avoid_: treating depth as page size

**PIFS Tree Pagination**:
Tree pagination is one global page over displayed child nodes in stable order:
folders, files, then metadata axes or metadata values.
_Avoid_: per-group cursors, unstable page ordering
