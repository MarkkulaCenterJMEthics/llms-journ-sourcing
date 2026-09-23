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

## At a glance

- **Phase 0** — Structural setup: XLSx→CSV conversion, header
  normalization. Mechanical, no article text needed.
- **Phase 1** — Anonymous Source reclassification: check every Anonymous
  Source row for an actual disclosure statement in the text; no
  disclosure → reclassify to Unnamed Person/Unnamed Group of People.
  Corpus-wide, before anything else.
- **Phase 2** — Source Descriptors population, one Type of Source at a
  time (the credentialing test: credentialing words → Title of Source,
  non-credentialing → Source Descriptors):
  - **2.1** Unnamed Group of People — single pass.
  - **2.2** Unnamed Person — single pass, same credentialing-test logic.
  - **2.3** Anonymous Source — single pass, same logic, applied to
    whatever's left after Phase 1's reclassification.
  - **2.4** Document — 3 sub-passes: schema-violation fix + SD
    population, missed-name recovery, genre-word recovery.
  - **2.5** Named Organization — 3 sub-passes: schema-violation audit
    (incl. Note 12 joint-org splits), name recovery, category-word SD
    population.
  - **2.6** Named Person — 4 sub-passes (largest volume): general
    annotation-error audit, Title credentialing-test audit, missed-title
    recovery, systematic SD recovery.
- **Phase 3** — Corpus-wide Source Justification/Title quality check,
  run once after all of Phase 2: (a) SJ can't just state medium of
  contact, (b) Title can't carry a bare relational descriptor, (c) flag
  single-word SJ for review, (d) SD compliance audit, (e)
  qualifier-stacking check on Title only, not Source Descriptors.
- **Phase 4** — Sourced Statement row-set audit, per file, run last
  after Phase 0-3: re-read the full article against the complete row
  set, both directions. Candidates only either way — needs explicit
  approval before anything's added, removed, or moved.
  - **4a (recall)** — find real sourced-statement candidates with no
    row at all.
  - **4b (precision)** — find existing rows with no manifest
    attribution signal (no quote marks, no attribution verb) that read
    as reporter narration of a source's background/inner experience
    rather than something drawn from the source — flag for removal,
    with a check for whether the content should migrate into Source
    Justification on the nearest downstream properly-attributed row
    from the same source first.

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
disclosure statement in the article text — explicit or a blanket statement
earlier in the story covering multiple sources (see `development-of-v59.md`'s
"Anonymous Source / Unnamed Person boundary" section for the full reasoning
on why this stays narrow). No disclosure found → reclassify to Unnamed
Person (or Unnamed Group of People if the source is a group).

Disclosure phrasing varies — check for all of these aliases, not just the
most common one: "spoke on condition of anonymity," "requested anonymity,"
"spoke anonymously," "not authorized to speak [publicly/on the record],"
"declined to be named," "asked not to be named," "requested not to be
named," or any other phrasing along the lines of the source asking that
they "not be named."

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
six types together (this check is not type-specific). Five checks:

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
- **(e) Qualifier-stacking check on Title of Source — Title of Source
  only, not Source Descriptors.** Re-check every populated Title of
  Source value for a political-orientation, editorializing, or otherwise
  characterizing qualifier stacked onto an otherwise-valid credentialing
  title (e.g., "far-right," "hard-line," "mainstream"). Apply the same
  test as (d)'s category-vs-elaboration distinction: does removing the
  qualifier still leave an accurate description of the same title
  (elaboration — trim it), or does it change what position is being
  described (category-defining — keep it)? For elaboration: trim the
  qualifier from Title of Source, and merge it into Source Justification
  — appended via ";" if Source Justification already has other content,
  set directly if it was blank — **unless the exact qualifier is already
  present somewhere in the existing Source Justification text**, in which
  case nothing needs to be added there, only the trim from Title.
  **This check does not apply to Source Descriptors.** A qualifier
  combined with a non-credentialing descriptor (e.g., "progressive
  rival," "far-left activist") is exactly what Source Descriptors is
  *for* — the prompt's own definition uses "far-left activist"/"far-right
  activist" as its canonical worked example of a reporter's reductive
  labeling, which the schema exists to capture, not trim away. Confirmed
  by item 34's retroactive sweep (`development-of-v59.md`), which
  initially checked both fields, found `GT-II/63-Maine_Data_Center.csv`'s
  "progressive rival" (Source Descriptors) as an apparent hit, and on
  review determined it wasn't a violation at all — it clarified that this
  check was never meant to reach Source Descriptors in the first place.
  Found 2026-09-16
  during Phase 2.6 Pass 1's Named Person Title-of-Source audit
  (`GT-II/178-AFD_Win_Europe.csv`: Salvini, Wilders;
  `GT-II/180-UK_Israel_Ban.csv`: Ben-Gvir; `GT-II/179-AFD_Majority_Fail.csv`:
  Merz) — see `development-of-v59.md` punchlist item 33 for the full
  worked examples, and item 20 for the two candidate prompt notes (21a,
  24a) drafted from this finding but not yet merged into
  `system_prompt_v60.txt`. **Note:** whether a qualifier should ever be
  dropped outright rather than always preserved in Source Justification
  (e.g., a case where the qualifier is purely the reporter's own value
  judgment with no independent factual content) is a real open question,
  deliberately deferred rather than decided here — default to always
  preserving in Source Justification until that's worked through.

## Phase 4 — Sourced Statement row-set audit (per file, after Phase 0-3 are done)

Run last, one file at a time, only after that file has gone completely
through Phase 0-3 — Phase 4 deliberately depends on the migrator already
having full context on the story (every existing Sourced Statement, Type,
Name, Title, Source Descriptors, and Source Justification), which is what
makes it possible to recognize a genuinely uncaptured statement instead of
a paraphrase or partial match of something already there (4a), or a
genuinely unattributed narration row instead of a legitimate implied
continuation of nearby attributed material (4b).

Two sub-checks, same file, same pass — both ask whether the row set as a
whole is *right*, just from opposite directions. Both produce candidates
only, never direct edits (see below); run 4a first, then 4b, since 4b's
"nearest downstream attributed row" check is easier once 4a has already
settled whether the row set is complete.

### 4a — Missing Sourced Statement sweep (recall)

**What it checks:** unlike every phase before it, which re-examines rows
that already exist, 4a checks for *rows that don't exist at all* —
re-read the full article text end to end and compare it against the
complete existing row set, looking for sourced-statement candidates (a
quote, a paraphrase attributed to someone, an "according to X" claim,
etc.) that aren't captured by any row, even loosely. This is a recall
check, not a precision check — the previous phases assume the row set is
fixed and improve what's in it; this phase asks whether the row set is
even complete.

**Known false-positive risk, watch for it:** first attempt at designing
this check (2026-09-17) nearly produced a false positive against
`GT-II/181-US_Canada_Dairy.csv` — a claimed "missing" row for a second
anonymous official's quote that in fact already existed (row 24, typed
Unnamed Person). The near-miss happened because the check that found it
was actually the Phase 1 Anonymous-Source-only scan, not a real Phase 4
run — it only looked at rows already typed Anonymous Source, so it never
saw row 24 (a different type) and wrongly concluded no row existed at
all. The lesson for 4a specifically: a genuine "is this statement
captured anywhere in the row set" check must compare against *all* rows
regardless of type, not a type-filtered subset — otherwise it reproduces
exactly this mistake. See `development-of-v59.md` punchlist item 37 for
the full incident.

**Second known false-positive risk, watch for it:** a distinct,
uncaptured clause naming a specific actor doing something distinctive
is not automatically a sourced-statement candidate — it also has to
actually be a *statement* (a claim, viewpoint, or experience drawn from
the source), not mere conduct/behavior narration. Caught 2026-09-23 on
`GT-II/166-DC_Essential_Workers.csv`: three parked candidates ("Local
rideshare drivers offered free rides...", "José Andrés' World Central
Kitchen delivered pizzas...", "crews of National Park Service and
Capitol employees were out collecting [trash]...") all looked like
plausible misses (distinct clauses, specific named actors, no existing
row) but are actually reporter narration of what these people/orgs
*did* — no claim, viewpoint, or statement is being attributed, just
action. The Sourced Statement definition already excludes this
explicitly: "when journalists report out what they are witnessing
first hand or what they conclude about a source or a source's actions,
and they are not attributing it, those lines are not sourced
statements." All three rejected, no rows added. **Screening rule for
every future 4a run:** before proposing a candidate, check not just
"is this clause captured anywhere" but "is this actually a statement
drawn from the source" — a specific named actor performing a specific
documented action is not sufficient on its own; there has to be an
actual claim, viewpoint, or experience being attributed, not just
conduct being described.

### 4b — Over-capture / unattributed-narration sweep (precision)

**What it checks:** the mirror image of 4a — rows that *do* exist but
arguably shouldn't, because they carry no manifest attribution signal at
all (no quotation marks, no attribution verb like "said," "recalled,"
"told us," "according to") and instead read as the reporter's own
third-person narration of what a source knew, believed, saw, felt, or
lived through. This is background/context about the source — what Source
Justification is for — not a statement the article is attributing *from*
the source, which is what Sourced Statement requires. Added
2026-09-22, arising from `development-of-v59.md` punchlist item 44 and a
concrete instance found across the "solidarity journalism initiative"
story range (`GT-II/174-PPP_Loans_Low_Income.csv` rows 1-4,
`GT-II/175-Indigenous_Health_COVID.csv` rows 24-27) — both cases showed
the same shape: a run of unattributed narration rows immediately followed
by a properly-attributed row from the same source.

For each hit, check whether it's a clean violation or falls under an
existing exception before flagging it as a removal candidate:
- **Implied continuation (Notes 9/10 exception, already established):**
  a row immediately adjacent to other directly-attributed material from
  the same source, functioning as a continuation of that attribution,
  is not a violation — don't flag it. (`GT-II/186-Homeless_Camp_Sweeps.csv`
  row 67 was reviewed and looked like this kind of case, not a clean hit.)
- **First-person voice without quotation marks** may be a stylistic
  "as-told-to"/paraphrase choice rather than reporter-voice narration —
  treat as a separate, distinct question, not automatically the same
  failure (`GT-II/169-Black_Mothers_Gun_Violence.csv` row 22 was flagged
  this way, not folded into the main pattern).

**Standing scope, going forward:** this is now a permanent part of Phase
4 for every future batch/file, not a one-off check — folding it into the
regular phase-based process means it gets caught automatically instead of
relying on someone noticing the pattern again by hand.

**Explicit exception for the Sep17 batch:** do **not** run 4b on
`GT-II/166` through `GT-II/176`, or `GT-II/185` through `GT-II/190` (the
17-story range where this pattern was found). The original annotator is
separately re-reviewing those files directly and will hand back revised
XLSx with these rows already addressed; the migration side will diff the
revision against the already-migrated CSVs to reconcile once that lands,
rather than duplicating the work here. 4a still runs normally on this
range.

### Output and review (applies to both 4a and 4b)

**Output is candidates, not edits.** A Phase 4 hit — either sub-check —
is never written directly into the CSV. Each candidate gets flagged
with the exact article text, why it looks like a miss (4a) or an
over-capture (4b), and a proposed disposition: for 4a, a full proposed
row (Type of Source, Name, Title, Source Descriptors, Source
Justification) if it were to be added; for 4b, whether to drop the row
entirely or migrate its content into Source Justification on the
nearest downstream properly-attributed row from the same source. The
user reviews and approves (or rejects) each candidate before anything is
added, removed, or moved; nothing gets annotated into the row set
unilaterally. This mirrors the existing review-spreadsheet workflow used
for other batch decisions in this migration, not a new mechanism.

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
