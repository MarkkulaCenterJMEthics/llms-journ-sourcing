# Source Justification: scoring problems and an annotation policy proposal

This document covers two related but separate Source Justification
questions in one place: **Part 1** is about how the *evaluation/scoring*
code grades Source Justification today (a set of bugs, all fixable). **Part
2** is a *proposed change to annotation itself* — whether Source
Justification should be captured strictly at the row level going forward,
rather than mechanically carried forward/backfilled/concatenated across a
source's rows. They're consolidated here because fixing Part 2 changes the
raw material Part 1's scoring code has to deal with.

---
---

# Part 1: How the scoring code grades Source Justification, and what's wrong with it

**Written for:** anyone on the team, technical or not.
**About:** `Justification_Match_Rate`, the Source Justification score produced by
`eval_multi_llm_article.py`. All line numbers refer to that file as committed on
this branch. A local working copy, `eval_multi_llm_article_v2.py`, inherits the
same code unchanged and is not yet committed.
**Date:** 17 September 2026.
**Short version:** the number is not trustworthy today. Four separate problems, all fixable.
Every figure below was measured on our own data, not estimated.

---

## First, the vocabulary

A few technical words appear throughout. Each is explained the first time and
repeated here for reference.

| Term | What it means in plain words |
|---|---|
| **Source Justification** | The text in an article that explains *why this source is in the story* — "a paralegal for the county attorney", "who was present at the meeting". It is one of the six columns our annotation asks for. |
| **Ground truth (GT)** | The human annotator's answer sheet. What we grade the model against. |
| **Row** | One line in the answer sheet: one sourced statement, plus who said it and the other five fields. |
| **Matched row** | A row where the matcher decided the model's statement and the human's statement are the same statement, so their other fields can be compared. |
| **Embedding** | Turning a piece of text into a list of numbers, so a computer can measure how similar two texts are by meaning rather than by spelling. We use a small standard model called MiniLM. |
| **Cosine similarity** | The number that comparison produces: 0.0 means "nothing in common", 1.0 means "identical in meaning". |
| **Threshold** | The pass mark. Ours is 0.55 — above it the field counts as correct, below it as wrong. It is pass/fail; there is no partial credit. |
| **Denominator** | The "out of how many" in a percentage. If we say 70%, the denominator is what the 70% is a share *of*. |
| **Hash seed** | A random number Python picks fresh every time a program starts. It quietly affects the order things come out of certain containers. |

---

## How the grading works today, in one paragraph

After the matcher pairs up statements, the code looks at each matched row and
asks: *does the model's Source Justification say roughly the same thing as the
human's?* It measures "roughly the same" with cosine similarity (the 0-to-1
meaning score above) and calls the row correct if the score is above 0.55.
The final percentage is how many matched rows passed.

That part is reasonable. **The problem is what happens immediately before the
comparison.**

---

## Problem 1 — Everything gets mashed together (the "glue")

### What it is

Before any comparison, the code finds every row belonging to the same person,
collects all the different justifications written about that person anywhere in
the article, glues them into one long string separated by commas, and writes
that same long string back onto **every row for that person** — including rows
where the human deliberately left the field blank.

It does this to the human's answer sheet *and* to the model's answers.

The code is one line, in `fill_with_same_person()` at
`eval_multi_llm_article.py` line 234:

```python
df.at[index, target_column] = ",".join(list(name_set[name]))
```

### Why it was written that way

The intention was generous. If a source is quoted five times and the reporter
explained who they are only once, the model should not be marked wrong four
times for leaving the field blank. Pooling everything means the model gets
credit as long as it found the justification *somewhere*.

### Why it backfires

It fails whenever the human recorded **more** justifications than the model
found — which is the normal case. The extra text drowns out the part the model
got right.

**A real example from our data** (story 9, Stephen Curry, the run from 15 Sept):

The human wrote two different justifications for him:

> **G1** — "said Wednesday before hosting a charity golf tournament benefiting the
> couple's Eat.Learn.Play. Foundation devoted to Oakland schools."
>
> **G2** — "He is in the process of trademarking the use of 'nuit, nuit' for
> entertainment purposes — the French words for his signature 'night, night'
> catchphrase..."

The model wrote one:

> **L1** — "the 36-year-old Curry said Wednesday before hosting a charity golf
> tournament benefiting the couple's Eat.Learn.Play. Foundation devoted to
> Oakland schools."

L1 is G1 with three extra words at the front. Scored honestly:

| Comparison | Cosine similarity | Verdict at 0.55 |
|---|---|---|
| G1 against L1 | **0.7398** | correct — the model found it |
| G2 against L1 | 0.0802 | unrelated, as expected |
| **"G1,G2" glued, against L1** — what the code actually does | **0.2819** | **wrong** |

0.74 becomes 0.28. G2 is roughly twice as long as G1, so it dominates the
comparison and sinks it. And because the glued string is written onto every one
of Curry's rows, **all** of his rows are marked wrong for an answer that was
right.

The model is being punished for the human being thorough. If the annotator had
recorded only G1, the row would have passed.

### Proposed fix

Stop gluing. Keep the justifications as a list, and compare the model's answer
against each of the human's separately, keeping the best score.

```python
def collect_by_person(df, target_column):
    """Attach every distinct value this Name carries as a LIST,
    without overwriting the row's own value."""
    by_name = defaultdict(list)
    for _, row in df.iterrows():
        name, value = row.get("Name", ""), row.get(target_column, "")
        if name and value and value not in by_name[name]:
            by_name[name].append(value)
    df[target_column + "_All"] = [
        sorted(by_name.get(row.get("Name", ""), []))
        for _, row in df.iterrows()
    ]
    return df


def process_justification_match_a(row, threshold=0.55):
    gt_list, llm_list = row["Justification_All_GT"], row["Justification_All_LLM"]
    if not gt_list and not llm_list:
        return None                    # nothing to judge (see Problem 3)
    if not gt_list or not llm_list:
        return False
    return max(semantic_match(clean_text(l), clean_text(g))
               for g in gt_list for l in llm_list) > threshold
```

The Curry row would then score `max(0.7398, 0.0802) = 0.7398` and pass, which is
the correct outcome. The original generosity is preserved: the model's
justification may come from any of its rows for that source.

---

## Problem 2 — The order of the glued text is random

### What it is

The justifications are collected in a **set** — a container that does not
remember the order things were put into it. Turning a set into a list therefore
produces an arbitrary order, and Python re-randomises that order every time the
program starts, based on the hash seed (the random number mentioned above).

So the same data can produce `"reason A, reason B, reason C"` on one run and
`"reason C, reason A, reason B"` on the next. Different word order → different
embedding → different cosine similarity → the pass mark can fall on the other
side.

### The evidence

Same file, same code, only the hash seed changed:

| Story / model | Seed 0 | Seed 1 | Other metrics |
|---|---|---|---|
| Story 1, Claude Sonnet 5 | **18.8%** | **87.5%** | Name, Title, Recall all unchanged |

That is 3 of 16 matched rows passing versus 14 of 16. Eleven rows flip together,
because one ordering decision for one source (Reuters/Ipsos, which carries most
of that story's rows) is copied onto every row of that source.

It does not fire on every story. It needs a source with several distinct
justifications *and* a glued score sitting near the pass mark. Most sources are
immune because they only ever carry one justification:

| Distinct justifications for one source | 1 | 2 | 3 | 4 | 5 | 7 |
|---|---|---|---|---|---|---|
| Ground truth (43 stories) | 98 | 15 | 4 | 3 | – | 1 |
| Model predictions (68 files) | 270 | 22 | 8 | 10 | 2 | 2 |

Consistent with that, a 17-story Qwen run and a 3-story run showed no change
across seeds, while Sonnet 5 on story 1 swung by 68.7 points.

**The practical consequence: no Source Justification figure this evaluator has
ever produced can be exactly reproduced.** Every run pins `PYTHONHASHSEED=0` so
runs are comparable to each other, but that just freezes one arbitrary ordering.

### Proposed fix

If the glue stays, sort before joining — a one-word change:

```python
-    df.at[index, target_column] = ",".join(list(name_set[name]))
+    df.at[index, target_column] = ",".join(sorted(name_set[name]))
```

The fix for Problem 1 removes this issue as well (its `sorted()` does the same
job), so this is only a stopgap if Problem 1 is not addressed.

---

## Problem 3 — Two blank cells count as a correct answer

### What it is

When the human wrote no justification and the model also wrote none, the code
compares an empty string against an empty string. MiniLM turns both into the
same list of numbers, so the cosine similarity is exactly **1.0**, which clears
the 0.55 pass mark. The row is recorded as correct even though nothing was
checked.

Verified directly:

```
semantic_match("", "")  ->  1.0        passes the 0.55 threshold: True
```

### Why it matters

Stories where the human left justification empty on many rows score better than
they should. The percentage is being lifted by rows where no judgement was made.

(Note the interaction with Problem 1: for a source that has *any* justification
anywhere, the glue overwrites its blank rows with the glued string — so those
rows are instead graded against text the human never attached to them. Both
behaviours are wrong, in opposite directions.)

### Proposed fix

Treat "both blank" as **not applicable** and leave it out of the denominator
(the "out of how many") rather than counting it as a win:

```python
def process_justification_match_a(row, threshold=0.55):
    gt, llm = clean_text(row["Justification_GT"]), clean_text(row["Justification_LLM"])
    if not gt and not llm:
        return None                  # excluded from the denominator, not scored as correct
    if not gt or not llm:
        return False
    return semantic_match(llm, gt) > threshold
```

and when averaging, drop those rows and report how many there were:

```python
verdicts = both_found["justification_match_a"]
scored   = verdicts[verdicts.notna()]
justification_match_rate  = scored.mean() if len(scored) else float("nan")
justification_rows_scored = int(len(scored))
justification_rows_na     = int(verdicts.isna().sum())
```

---

## Problem 4 — The summary and the detailed dump disagree

### What it is

We produce two things: the summary percentage in the results table, and an
optional per-row spreadsheet (`--dump-comparisons`) showing each row's score.
They are computed by **two different functions with similar names**.

- The summary compares the **whole strings**.
- The dump chops both texts into clauses, compares every clause pair, and keeps
  the best one (`using_split=True`, the default of `semantic_compare` at line 122,
  applied to these columns at line 614).

### Why it matters

You cannot verify a published percentage by opening the detailed spreadsheet —
the rows will not add up to it. The two also disagree about blanks: the whole
string version returns 1.0, the clause version returns 0.

This is not hypothetical. When checking whether a recent accuracy gain was real,
recomputing **Title** from the dumped column gave 36% against a reported 96%.
The reported figure was correct; the dump column simply measures something else.

### Proposed fix

Make the dump use the same ruler as the summary, and also write out the actual
pass/fail the metric counted, so the spreadsheet can be checked by hand:

```python
for attr in ["Title", "Justification", "Title_and_Justification"]:
    result_df[f"{attr}_Match_Score"] = result_df.apply(
        semantic_compare, axis=1, gt_col=f"{attr}_GT", llm_col=f"{attr}_LLM",
        using_split=False)                       # same ruler as the reported rate

result_df["Justification_Pass"] = both_found["justification_match_a"]   # True / False / None
```

---

## Problem 5 — The percentage is "out of" a different number each time

### What it is

Every field score is an average over **matched rows only** — rows where the
matcher paired the model's statement with the human's. A model that finds fewer
statements is graded on fewer, and usually easier, rows.

### Why it matters

Comparing "Justification 10% versus 16.7%" between two experiments can be
meaningless if the two found different numbers of statements, because the two
percentages are shares of different row sets.

### Proposed fix

No change to the formula. Two reporting habits:

1. Always print statement recall (the share of the human's statements the model
   found) next to any field percentage.
2. When comparing two experiments, compare only on the rows **both** of them
   matched. We did this recently for Name and Title, and it confirmed a real
   +9 point gain rather than a denominator effect — the same test should be
   routine.

---

## What to do about it

| Problem | Severity | Fix effort | Changes published numbers? |
|---|---|---|---|
| 1. Glue / dilution | **High** — wrong answers on correct work, fires constantly | Medium | Yes — justification scores will rise |
| 2. Random order | Medium — unreproducible, sometimes 68 points | One word | Yes, slightly |
| 3. Blank = correct | Medium — inflates some stories | Small | Yes |
| 4. Dump vs summary | Low for scores, **high for trust** | Small | No |
| 5. Moving denominator | Low — a reading habit, not a bug | None | No |

**Suggested rollout.** Put Problems 1 and 3 behind a switch, so old numbers stay
reproducible and both can be published side by side once:

```
--justification-mode legacy | best-match        (default: legacy)
```

This mirrors the existing `--apply-fixes` pattern already used for the statement
matching fixes. Problems 2 and 4 can be applied unconditionally — one only
removes randomness, the other only fixes a reporting inconsistency.

**Until this lands:** do not quote `Justification_Match_Rate` as a finding in
either direction. In the recent v60 experiments it also carries a 15-point
run-to-run noise band from the model's own variation, on top of everything
above. Name, Title, Type, Recall, Precision and Descriptors are unaffected — they
were identical across every seed tested.

---

## Where the code is

All line numbers are `eval_multi_llm_article.py` as committed on the `v59` branch.

| What | Line |
|---|---|
| The glue — `fill_with_same_person()` | 220, with the offending join at **234** |
| Where the glue is applied, in `preprocess_dataframe()` | 253–254 |
| The blank-vs-blank comparison — `process_justification_match_a()` | 663 |
| The similarity function, no empty-string guard — `semantic_match()` | 173 |
| The dump column's other ruler — `semantic_compare(..., using_split=True)` | 122, applied at 614 |
| The reported percentage | 825 |

The local, uncommitted `eval_multi_llm_article_v2.py` inherits all of this
unchanged by design, so the two stay comparable; only its line numbers differ
(the glue sits at 334 there).

## How to reproduce the evidence

The commands below use `eval_multi_llm_article_v2.py`, a local working copy that
takes its settings on the command line. It is not committed to this branch yet;
the committed `eval_multi_llm_article.py` scores identically but needs its
`main()` edited by hand instead of accepting these flags.

```bash
# Problem 2 -- same file, two seeds (18.8% vs 87.5%)
for s in 0 1; do
  PYTHONHASHSEED=$s python eval_multi_llm_article_v2.py \
    --gt-dir "benchmarking/GT data/GT-2026" \
    --pred-glob "../rai-scu-kite/sourcing_data/v59_gt_benchmark/story01_cl-son5_v59.csv" \
    --pred-regex "story(?P<story>[0-9]+)_(?P<model>.+)_v59[.]csv$" \
    --models cl-son5 --stories 1 --tag seed$s
done

# Problem 3 -- blank against blank
python -c "from eval_multi_llm_article import semantic_match; print(semantic_match('', ''))"   # 1.0
```

Problem 1's Curry figures come from
`llm_results/v60_qwen_test/C/story09_qwen3-7-max_v60.csv` against
`benchmarking/GT data/GT-2026/9-steph-curry-gold.csv`, scoring each justification
pair with `semantic_match` before and after the `",".join(...)` step.

Not to be confused with `sj-analysis-notes.md` on this branch, which asks a
different question entirely — what *kinds* of justification content appear in the
corpus (the SJ typology work). This part of the document is only about how the
score is computed — see below for the related annotation-side question.

**Companion documents:** `eval-justification-issue.md` (the same material in
technical note form), `eval-matching-logic-review.md` (how statements are paired
before any of this runs), and
`benchmarking/metrics/eval_methodology_report_2026-09-15.docx` / `.pdf` (both
topics, written up for circulation).

---
---

# Part 2: Annotation policy proposal — row-level Source Justification capture

**Status:** Draft for discussion, not yet adopted.
**Prepared:** 2026-09-24, for review with student annotators.
**Relationship to Part 1 above:** Part 1 covers a related but separate
problem — how the *evaluation/scoring* code currently glues multiple Source
Justification values together across rows before comparing model output to
ground truth (see "Problem 1 — Everything gets mashed together," above).
This part is about the *annotation* side: whether the ground truth itself
should carry that same kind of cross-row duplication in the first place. If
this proposal is adopted, it removes the raw material that feeds Part 1's
Problem 1 in the eval code, though that eval-side fix may still be needed
independently depending on the benchmark-scoring design chosen (see below).

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
backfill) is settled. This section covers parts (b) and (c) (the
concatenation question and the resulting prompt-text changes) — still
pending the stakeholder meeting before either prompt file is actually
edited.
