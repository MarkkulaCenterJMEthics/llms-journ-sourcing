# Source Justification — annotation policy proposal: row-level only

**Status:** Draft for discussion, not yet adopted.
**Prepared:** 2026-09-24, for review with student annotators.
**Companion document:** `SJ_Review_Proposal.md` covers a related but separate
problem — how the *evaluation/scoring* code currently glues multiple Source
Justification values together across rows before comparing model output to
ground truth (see "Problem 1 — Everything gets mashed together," that
document). This proposal is about the *annotation* side: whether the ground
truth itself should carry that same kind of cross-row duplication in the
first place. If this proposal is adopted, it removes the raw material that
feeds Problem 1 in the eval code, though the eval-side fix may still be
needed independently depending on the benchmark-scoring design chosen (see
below).

---

## The proposal

Source Justification becomes strictly row-level, governed by annotator
judgment rather than mechanical rules:

- Each Sourced Statement row's Source Justification (SJ) is captured based on
  the annotator's own read of which row it most directly connects to — the
  relationship between the source, the candidate justification text, and
  that specific statement.
- The justification text itself may be found **anywhere in the article**
  relative to that statement — upstream, downstream, right next to it. Search
  scope is not restricted.
- What changes: SJ is **not mechanically copied** onto every other row for
  the same source just because the source recurs. No automatic forward
  carry-forward, no automatic backward backfill.
- The same justification text **can** still appear on more than one row for
  a source, but only when the annotator independently judges it applies to
  each — a deliberate call made per row, not a default behavior.
- If no justification content applies to a given row, Source Justification
  is left null for that row, even if another row for the same source
  elsewhere in the story does carry it.
- No ";"-concatenation of multiple justification fragments into one field
  value.

**Rationale.** The row-source link exists for practical/structural
reasons — one dataset sheet, and letting the annotator note *when* in the
story a justification first surfaces relative to a statement — not to make
every row a cumulative summary of everything ever said about that source.
Reconstructing a source's full justification across a story is a downstream
analysis step (group all of that source's rows, read their SJs together in
order), not something that needs to be pre-baked redundantly into the raw
annotation. Whether a full corpus cleanup is actually necessary may itself
depend on a benchmark-scoring design choice made later (row-by-row matching
vs. source-clustered matching of model output against ground truth) — but
the ";"-concatenation convention likely needs undoing regardless of that
choice.

---

## Scope of current usage

How widespread the existing (soon-to-be-replaced) behaviors are, across both
corpora:

| | Files scanned | Carry-forward (identical SJ repeated for same source) | Concatenation (";" in SJ) |
|---|---|---|---|
| **GT-2026** | 43 | 26 files, ~157 rows | 4 files, 30 rows |
| **GT-II** | 116 | 74 files, ~628 rows | 16 files, 68 rows |

Forward carry-forward is pervasive — roughly two-thirds of all files in both
corpora use it, ~785 rows combined. Concatenation is much narrower — 20
files, ~98 rows total. Adopting the new policy is not a small cleanup: it
means revisiting each of those ~785 carried-forward rows individually, not a
mechanical strip.

---

## A worked example that sharpens the open question

Found and applied 2026-09-24, on `GT-II/181-US_Canada_Dairy.csv`, rows 8-9
(Type: Anonymous Source, Title: "Canadian official"). Both rows already had a
valid Source Justification fragment: *"a Canadian official familiar with the
discussions."* Further down the article — several sentences later, after an
unrelated paragraph and a stray extraction artifact — sits the disclosure
sentence that's the actual reason this source is typed Anonymous Source at
all: *"The official spoke on condition of anonymity because they were not
authorized to discuss the talks publicly."* That sentence had never been
captured anywhere. It was added to both rows via the currently-active
";"-concatenation convention: *"a Canadian official familiar with the
discussions; spoke on condition of anonymity because they were not
authorized to discuss the talks publicly."*

This case is compatible in spirit with the row-level proposal above — it's
exactly the "annotator scans the whole article, finds a second genuinely
relevant fact, and judges it belongs on this row" behavior the proposal
wants to keep. But it's also a clean example of the harder edge the proposal
doesn't yet resolve: this isn't a case of the *same* text independently
belonging on *multiple different rows* (the scenario the proposal's "may be
captured on each such row... a deliberate judgment call" language covers) —
it's **two distinct, individually-valid facts landing on the same row**,
discovered at different points in the search, that need to be held together
rather than one replacing the other. For a source type like Anonymous
Source specifically, the disclosure sentence isn't optional color — without
it, the row's own Type assignment has no textual support. Worth using this
example directly in tomorrow's discussion: does the row-level proposal keep
some form of same-row concatenation for cases like this, or does it expect
the annotator to somehow capture both facts without a joining mechanism (a
single combined sentence, written in their own words, which would break the
"extract only the actual text, don't synthesize" rule)? This is a genuinely
open sub-question the current draft doesn't cover.

---

## Prevalence data: how often does the "pull from downstream" pattern actually happen?

Two questions, checked directly against both corpora, motivated by a
real annotation behavior already in use: annotators sometimes pull
justification text from several paragraphs *downstream* of a source's first
quote back into that first quote's row, because that is where the
justification content actually lives in the article. The new policy must
not accidentally forbid this — it should only forbid *mechanical* repetition
across rows, not the underlying whole-article search.

**(a) How often is a source's first-row SJ left null, while a later row for
that same source does carry a real SJ value?**

| | Multi-row sources scanned | First row null, later row has SJ |
|---|---|---|
| **GT-2026** | 115 | 8 (7.0%) |
| **GT-II** | 411 | 19 (4.6%) |

**(b) When a source's first row *does* have an SJ value, how often is that
value actually text located downstream in the article, rather than at or
before the point of that first quote?**

| | First-row-with-SJ cases | Unmatched (skipped) | Downstream | Upstream/same location |
|---|---|---|---|---|
| **GT-2026** | 114 | 4 | 38 (34.5% of scored) | 72 (65.5% of scored) |
| **GT-II** | 279 | 37 | 82 (33.9% of scored) | 160 (66.1% of scored) |

**Takeaway:** (b) confirms this is a real, common pattern, not an edge case —
roughly one in three first-quote justifications are already pulled from
later in the article. Any prompt revision must explicitly preserve
whole-article search scope, not narrow it to "near the quote" or
"upstream only."

*(Methodology note: (b) uses normalized substring matching between each
row's SJ/SS text and the article's own text — quote-curling, whitespace, and
capitalization differences are normalized away, but exact wording drift can
still cause a match to fail; those cases are excluded from the scored total
rather than guessed at, and are reported separately as "unmatched.")*

---

## Current text vs. proposed revision

### `system_prompt_v60.md` — main Source Justification definition

**Current** (unchanged by this proposal, quoted for context):
> This refers to additional source characterization, context, explanation
> that justifies to the reader why the source is in the story or that
> section of the story, how they are connected to the story. Source
> Justification may be a few words, a part of a sentence, multiple
> sentences, or a full paragraph. Any of the six defined types of sources
> may have such justifications and explanations present. Sometimes the
> source justification is part of the source's introduction, but it might
> also come later in the story after some statements have been attributed
> to the source. [...]

No change proposed to this paragraph — it already correctly allows
justification text to appear anywhere relative to the statement.

### `system_prompt_v60.md` — Note 24

**Current:**
> **Note 24:** When the same source appears in multiple sourced statements
> across the story, a prior sourced statement (SS1) attributed to that
> source may itself serve as the Source Justification for a later sourced
> statement (SS2) by the same source. The content of SS1 may explain why the
> source is further quoted in SS2, possibly alongside additional reporter
> interpretive text. This does not happen often, but when it does, copy the
> relevant prior text as the Source Justification. If no Source
> Justification text is found elsewhere in the story either, leave the
> field null.

**Proposed:**
> **Note 24:** Source Justification is captured through the annotator's own
> judgment of where it applies — not through mechanical copying. When the
> article offers justification for a source's presence in the story,
> capture it on whichever sourced statement row you judge it most directly
> connects to, based on your own reading of the relationship between the
> source, the candidate justification text, and that specific statement —
> this text may sit anywhere in the article relative to the statement,
> upstream or downstream. If the same justification text genuinely and
> independently applies to more than one row for that source, it may be
> captured on each such row — but only as a deliberate judgment call made
> separately for each row, never as an automatic copy applied just because
> the source recurs. If no justification content applies to a given row,
> leave Source Justification null for that row, even if another row for the
> same source elsewhere in the story does carry it.

### `user_prompt_v60_csv.md` — Source Justification extraction step

**Current:**
> Next, find the Source Justification for the source in this sourced
> statement using the definition. Directly extract the text that matches
> this definition from the story. Do not synthesize or generate or
> summarize the Source Justification in your own words. Extract only the
> actual text. If the same source is attributed in earlier sourced
> statements, even if there is no source justification text in this
> instance, you may copy the Source Justification from the source's earlier
> sourced statement to this one (system_prompt Note 24). Remember that
> source justification refers to words or sentences doing additional
> characterization or context or explanation for inclusion of that source
> in the story — not why a particular statement is there, but why the
> source is in the story at all (system_prompt Note 23). A mere text of
> attribution (e.g., "Jane Doe said by email", "posted on Instagram") is NOT
> source justification — those are transparency disclosures about how the
> reporter accessed the source, not why the source matters (system_prompt
> Note 22). By contrast, an attribution like "according to three people who
> were present at the meeting" — the "three people present at the meeting"
> — is valid source justification, since it explains their connection to
> what's being reported. If more than one source justification is present
> in the text, concatenate them as one single string separated by ';'.

**Proposed:**
> Next, find the Source Justification for the source in this sourced
> statement using the definition. Directly extract the text that matches
> this definition from the story — it may appear anywhere in the article
> relative to this specific statement, not only nearby or only upstream. Do
> not synthesize or generate or summarize the Source Justification in your
> own words. Extract only the actual text. Capture it only on the row where
> you judge it most directly applies, based on the connection between the
> source, the justification text, and this specific statement — do not copy
> it onto other rows for the same source just because the source recurs
> (system_prompt Note 24). The same text may independently apply to more
> than one row if you judge it does, but that is a deliberate call for each
> row, not an automatic copy. If no justification text applies to this row,
> leave the field null, even if another row for the same source elsewhere
> in the story does carry it. Remember that source justification refers to
> words or sentences doing additional characterization or context or
> explanation for inclusion of that source in the story — not why a
> particular statement is there, but why the source is in the story at all
> (system_prompt Note 23). A mere text of attribution (e.g., "Jane Doe said
> by email", "posted on Instagram") is NOT source justification — those are
> transparency disclosures about how the reporter accessed the source, not
> why the source matters (system_prompt Note 22). By contrast, an
> attribution like "according to three people who were present at the
> meeting" — the "three people present at the meeting" — is valid source
> justification, since it explains their connection to what's being
> reported.

---

## Open discussion points for tomorrow's meeting

1. **Which row does a justification "belong to," when more than one row
   could plausibly carry it?** Not necessarily the mechanically-first
   occurrence — the working idea is the first row where the annotator can
   see a genuine connection between source, justification, and statement.
   But this itself is contestable: an annotator could reasonably argue that
   *any* early quote (even an incidental biographical one) is a fair place
   to attach it, since the fuller picture only emerges once all of a
   source's SJs are read together regardless of which row each one sits on.
   Not resolved here — needs the group's input.
2. **Corpus cleanup scope and sequencing.** ~785 rows across both corpora
   currently rely on the carry-forward behavior this proposal would retire.
   Whether all of them need hand review depends partly on the benchmark
   scoring-design decision (row-by-row vs. source-clustered matching) —
   worth deciding that first, since it could substantially change how much
   cleanup is actually necessary versus how much the eval code can absorb on
   its own.
3. **Concatenation** — removing the ";"-join instruction is more clear-cut
   than the row-attachment question, but should still be confirmed with the
   group, since 20 files / ~98 rows currently use it.

---

## Status of this proposal in the migration punchlist

Tracked as item 48 in `development-of-v59.md`. Part (a) (no backward SJ
backfill) is settled. This document covers parts (b) and (c) (the
concatenation question and the resulting prompt-text changes) — still
pending the stakeholder meeting before either prompt file is actually
edited.
