# GT dataset migration methodology (schema v55 → v59/60)

This is the actual step-by-step process for upgrading a batch of GT stories
from the older 5-field schema (v55) to the current 6-field schema (v59/60).
It was reconstructed after the fact from the GT-2026 (stories 1-43)
migration commit history rather than planned out in advance — this document
is what should be *followed from the start* for every subsequent batch
(GT-II and beyond), instead of re-deriving it again.

Student annotators should read this before doing a final human review pass
on files that have gone through this process: it explains what kinds of
changes to expect and why, and where genuinely uncertain calls get flagged
rather than silently decided.

For the underlying design reasoning behind the rules referenced here (the
credentialing/non-credentialing test, the atomic-word rule, the Anonymous
Source / Unnamed Person boundary, etc.), see `development-of-v59.md`. For
the full chronological log of every individual finding, fix, and open
question surfaced while applying this methodology, see that same file's
"Migration punchlist" section.

## Phase 0 — Structural setup (no story text needed yet)

Purely mechanical, can happen before any article text is in hand:

1. Add the 6th column, Source Descriptors, as an empty field to every
   story's CSV (already done once for the whole schema; for a new raw batch
   arriving in spreadsheet form, this means converting each file to CSV
   with the current canonical header and an empty Source Descriptors
   column added).
2. Normalize file-level inconsistencies while converting: header casing
   and singular/plural variants (e.g. "Sourced Statement" vs "Sourced
   Statements"), inconsistent header row position, stray extra columns
   that aren't part of the six-field schema (kept, not discarded, but
   tracked separately rather than mixed into the main file).

## Phase 1 — Anonymous Source reclassification (corpus-wide, before anything else)

For every row currently typed Anonymous Source, check for an actual
disclosure statement in the article text — explicit ("spoke on condition of
anonymity") or a blanket statement earlier in the story covering multiple
sources (see `development-of-v59.md`'s "Anonymous Source / Unnamed Person
boundary" section for the full reasoning on why this stays narrow). No
disclosure found → reclassify to Unnamed Person (or Unnamed Group of People
if the source is a group).

This runs across the *whole* batch before any Source Descriptors work,
since it determines which canonical type a row even belongs to before
Phase 2's type-by-type work begins.

## Phase 2 — Source Descriptors population, one Type of Source at a time

v59/60 has 6 canonical Types of Source. Phase 2 works through all 6, done in
this order deliberately: simplest/fewest-rules types first, then Anonymous
Source deliberately placed ahead of the three multi-pass types (given its
sensitivity, not its row count — see 2.3 below), then the remaining
multi-pass types in increasing volume/nuance, highest-volume/most-nuanced
type last. Three of the six types are simple enough to need only a single
pass; three need multiple sub-passes, broken out below.

### 2.1 — Unnamed Group of People (single pass)

Move any common-noun values sitting in Name of Source (a legacy v55 misuse —
this field must be null for this type) into Title of Source or Source
Descriptors correctly, per the credentialing test. Backfill Source
Descriptors from Source Justification text where the descriptor word is
present there but wasn't captured into its own field.

### 2.2 — Unnamed Person (single pass)

Same credentialing-test-based population of Title of Source / Source
Descriptors as Unnamed Group of People.

### 2.3 — Anonymous Source (single pass)

Same credentialing-test-based Title of Source / Source Descriptors
population as the other single-pass types, applied to whatever rows are
left after Phase 1's reclassification. Placed third, ahead of the three
multi-pass types, rather than last — Anonymous Source is a sensitive type
(it's the schema's main lens on how transparently a newsroom discloses
anonymity), and it deserves earlier, deliberate attention in the pass
order rather than being handled only once the bulk of the row-count work
is done. **Also easy to miss on its own merits**: Phase 1 reclassifies most
Anonymous Source rows away, so very few genuine ones are usually left —
this type has been skipped by accident on the first pass at this
methodology twice now (once during GT-2026, caught only after the other
five types were already marked done; once again while first writing this
document down for GT-II), specifically *because* it's easy to forget when
so few rows remain. Don't skip it just because the row count looks small,
and don't let its low volume push it to the back of the queue either.

### 2.4 — Document (3 sub-passes)

- **Pass 1 — Schema-violation fix + Source Descriptors population.** Fix
  Name/Title of Source misuse (Document never carries a Title of Source),
  and populate Source Descriptors.
- **Pass 2 — Missed-Name recovery.** Re-check the article text for a stated
  document title that wasn't captured into Name of Source.
- **Pass 3 — Genre-word recovery.** Recover genre-word descriptors (memo,
  report, lawsuit, deposition, etc.) from Sourced Statement/Source
  Justification text not yet captured into Source Descriptors.

### 2.5 — Named Organization (3 sub-passes)

- **Pass 1 — Schema-violation audit.** Clear invalid Title of Source values
  (this type never carries one); recover full organization names from
  text; reclassify informal/generic references — e.g. "police" with no
  formal name stated at that point — to Unnamed Person/Unnamed Group of
  People rather than inferring the formal name from elsewhere in the
  article; split rows jointly crediting multiple organizations into one
  row per organization (per Note 12); fix naming inconsistencies.
- **Pass 2 — Name recovery.** Folded into Pass 1's name-recovery work
  above rather than run as a fully separate pass.
- **Pass 3 — Category-word Source Descriptors population.** Bare
  category-word descriptors only (e.g. "nonprofit," "app," "think tank," —
  not fuller function/mission language, which belongs in Source
  Justification instead).

### 2.6 — Named Person (4 sub-passes, largest volume)

- **Pass 0 — General annotation-error audit.** Formatting artifacts,
  Name/Title of Source mix-ups, full-name recovery, joint-credit row
  splits (per Note 12), correcting over-inferred titles.
- **Pass 1 — Credentialing-test audit on Title of Source.** Catches
  non-credentialing words wrongly sitting in Title of Source (they belong
  in Source Descriptors instead), including organizational-affiliation
  text that belongs in Source Justification instead.
- **Pass 2 — Targeted missed-title recovery.** Specific flagged candidates
  where the article states a credentialing title that was never captured.
- **Pass 3 — Systematic Source Descriptors recovery.** From Sourced
  Statement/Source Justification text, done in batches.

## Phase 3 — Corpus-wide Source Justification / Title quality check

Run once, after all of Phase 2's type-by-type passes are done, across all
six types together (this check is not type-specific). Three checks:

- **(a) Source Justification must not merely state the medium of contact**
  (e.g. "said in a video posted Friday," "wrote in an email to X") — that's
  not substantive justification for *why* the source is relevant to the
  story, only a description of *how* the reporter obtained the quote.
  Blank the value (don't replace it with something invented) if no other
  substantive content is available to preserve.
- **(b) Title of Source must not carry a bare relational descriptor** (e.g.
  "Victim's granddaughter") — those are non-credentialing and belong in
  Source Descriptors instead, per the same rule used throughout Phase 2.
- **(c) Flag single-word Source Justification values for review.** A
  one-word SJ (e.g. "Speaker") is legal under the definition ("a few words"
  is allowed) but is frequently too thin to actually justify the source's
  standing in the story, and is worth a manual look every time rather than
  assumed correct. Caught first in `GT-II/74-AV_San_Ramon_Pride.csv`
  (Source Justification "Speaker" for a source introduced as "One speaker,
  a student at Gale Ranch Middle School" — expanded to "speaker, a student
  at Gale Ranch Middle School" to actually convey the source's standing).
  Not every single-word SJ needs a fix — some genuinely are sufficient on
  their own — but every one should be looked at, not skipped past because
  it technically satisfies the minimum length the definition allows.
  Applies to any batch, including new v60-native annotations once those
  start arriving (single-word SJ is a thinness problem independent of
  which schema version produced the row).
- **(d) Source Descriptors compliance audit.** Re-check every populated
  Source Descriptors value against the org-affiliation-stripping rule, the
  atomic-word rule, the capture-exactly-as-written rule, and (for Named
  Organization) the category-vs-composition/function distinction. Unlike
  Title of Source, which gets a dedicated audit pass of its own (Named
  Person Pass 1, 2.6 above), Source Descriptors was populate-once with no
  self-check anywhere in this methodology until this item was added — the
  gap was found 2026-09-15 and fixed forward across all existing migrated
  data (GT-II new-material batch, GT-II original 48-story batch, GT-2026)
  before being formalized here; see `development-of-v59.md` punchlist item
  29 for the full history. Run corpus-wide, after all six types' Phase 2
  population is done — not folded into each type's own population pass.

## Running practice throughout all of the above

Not a discrete step — applies constantly across every phase:

- **Always check the actual article text, not just the annotation cells.**
  Most of the real fixes in every phase above (recovering a missed title,
  confirming an organization's formal name is or isn't actually stated,
  judging whether a descriptor word is genuinely present) require reading
  the source article, not just reasoning from what's already in the CSV
  row.
- **Log real findings as they're found, rather than deciding silently.**
  Genuine ambiguities, schema gaps, or new design questions get written
  down (in `development-of-v59.md`'s Migration punchlist, for the
  internal/schema-facing side of this work) instead of being resolved
  unilaterally mid-pass. If a batch turns up a pattern not covered by an
  existing rule, that's a signal to stop and flag it, not to guess and
  move on.
- **Every fix should be traceable to specific reasoning**, not just a
  changed value — this is what lets a human reviewer (including the
  original student annotator, doing a final check on their own story)
  understand *why* something changed, not just *that* it changed.
