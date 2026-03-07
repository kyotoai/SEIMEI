# SEIMEI Patch Format

`seimei.editing.apply_patch` accepts patch payloads wrapped by:

```text
*** Begin Patch
...
*** End Patch
```

Only `*** Update File:` operations are supported.

## Default Format (Line-Range)

This is the default format to generate from `edit_file`.

```text
*** Begin Patch
*** Update File: relative/path.py
@@<line>
<lines to insert before <line>>
@@<start>-<end>
<replacement lines for the deleted range>
*** End Patch
```

Rules:
- Paths must be relative and stay inside the workspace.
- Line numbers are 1-based.
- All line numbers are interpreted against the original file content before any hunk is applied.
- `@@<line>` inserts text before `<line>` and must include at least one inserted line.
- `@@<start>-<end>` replaces the inclusive range `<start>...<end>`.
- For delete-only, use `@@<start>-<end>` with no replacement lines.
- For one-line replacement/deletion, use `@@15-15`.
- Hunks must not overlap, and insertion points cannot be inside a replaced range.

## Legacy Fallback Format

For compatibility, legacy Codex-style update hunks are still supported in update blocks:
- `@@` or `@@ <context>`
- Lines prefixed by space (` `), `+`, and `-`

Legacy and line-range hunks cannot be mixed in the same `*** Update File:` block.

## Example

```text
*** Begin Patch
*** Update File: app/main.py
@@3
from typing import Any
@@10-12
    value = normalize(value)
*** End Patch
```
