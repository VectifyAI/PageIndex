# PageIndex naming rules v1

Chat, Compute and the Python SDK share this contract. The byte-identical
`naming-v1.json` fixtures run in Vitest and pytest; update all three copies
when changing the contract. Names preserve case. Existing duplicate-name
scopes and database comparison behavior are unchanged.

## New names

- Apply Unicode NFKC, then normalize quote variants as in Chat (`‘`, `’`,
  `ʼ` to an apostrophe; `“`, `”` to a double quote). Collapse whitespace using
  the JavaScript whitespace set and trim leading/trailing whitespace.
  This order is idempotent, including characters that expand into quotes.
- Allow Unicode text, emoji (including joined emoji), and interior spaces.
- Disallow `/ \ : * ? " < > |`, C0/C1 controls including DEL, and Unicode
  line/paragraph separators. Check controls before whitespace normalization.
- Disallow dot-only names, trailing periods, and case-insensitive Windows
  device names CON, PRN, AUX, NUL, COM0–9 and LPT0–9, also with extensions.
- The final name, including extension, hash and collision suffix, must fit
  **180 UTF-8 bytes**. Do not split a Unicode code point when truncating.

## Upload allocation

Replace disallowed characters with `_`, remove trailing periods, prefix
reserved device names with `_` (preserving the extension), and use `untitled`
when nothing remains. Replace unpaired surrogates with U+FFFD.

Shorten overlong names with `_` plus the first eight hexadecimal characters
of the MD5 of the cleaned name. Preserve the extension where possible; if the
extension itself exceeds the budget, shorten it too. MD5 is only a stable
label here, not a security mechanism. A duplicate adds `_1`, `_2`, etc.
before the extension, reapplying the byte budget to the entire result.
Each backend keeps its existing collision-attempt limit.
The allocating backend returns the actual final name alongside its upload
URL or document ID. Chat and cloud SDK callers use that response unchanged.
The local SDK is its own backend and performs the same allocation locally.

## Folder creation and rename

Clients validate without rewriting the request. The backend normalizes
Unicode and spaces, then rejects invalid characters, reserved names
or excessive byte length with an actionable error. Never turn `/Research/`
into `Research`. Check duplicates using the normalized name before saving.
ZIP import reports invalid folder entries; it does not silently rename their
path components. Generated ZIP root folder names follow the upload rules.

## Reading and rollout

Read assigned names literally. Apply no new cleaning, truncation or case
folding to persisted names or to a final upload name submitted for processing.
At the processing boundary, still reject path separators, controls and `.`/`..`
so the name remains one path component. Existing names are not migrated.

Deploy Compute first (through dev verification), then Chat and the SDK.
Compute's internal `/files/upload-url` response becomes
`{ "url": "...", "headers": {}, "name": "final-name.pdf" }` for both S3 and
Azure. Consumers must use `name`, not reconstruct it from the input or URL.
