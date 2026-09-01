# rev2025/ — revision pass on the base GT folder

Files here are `{story}-{slug}-rev2.csv`, one per story that received a revision pass over the base folder's original 2024 annotation. Per the parent folder's readme: for any story with a file both here and in the base folder, **this version supersedes the original and is canonical**. Only use the base-folder version if no file exists for that story here.

32 stories currently have a `rev2025/` file (see below for the story added today).

## Changes from today's session (2026-09-01)

### Type of Source taxonomy normalization (commits `fd6d33b`, `0731656`)

Same cleanup described in detail in `../GT-2026/readme.md` — applied here first since `GT-2026/` is built from these files. Affected 24 of the files in this folder: normalized casing/typo variants for `Unnamed Group of People`, `Named Organization`, and the legacy `named individual` synonym for `Named Person`; stripped embedded newlines from `Type of Source` cells in stories 6 and 24. All changes verified via CSV round-trip testing before being applied, so only the intended cells changed.

### Legacy anonymity column removed (commit `fd6d33b`)

Dropped a `Y/N` anonymity column (spelled `Anomynity? (Y/N)`, `Anonymity? (Y/N)`, `Anonymity Y/N`, `Anomynity (Y/N)`, or `If Unnamed, Anomynity? (Y/N)` depending on the file) from 24 files here. It predated the `Anonymous Source` / `Unnamed Person` category split and had become fully redundant with `Type of Source` — verified before removal (see `GT-2026/readme.md` for the numbers).

### `17-homelessness-santa-clara-rev2.csv` added (commit `554cee4`)

New file — story 17 had no `rev2025/` revision before today. Its base-folder version had a row-numbering bug (2 statements with a blank `No`, never assigned a sequence number); rather than fix that in place in the base folder, the fix was applied and promoted into a proper revision here, matching how story 22's fix was handled earlier (see commit `53b384e`). The base-folder original is left untouched, still carrying the numbering bug, since it's treated as a frozen original.

## Going forward

`GT-2026/` (one level up) is now the canonical consolidated baseline — it was built from this folder plus the base folder's non-revised stories, with filenames normalized. New GT revision work should happen in `GT-2026/` directly rather than here; this folder and the base folder are kept as the historical record of how the 2026 baseline was derived, not as an actively maintained parallel copy.
