# v59 Sourcing Schema — Design Background for Migration Work

This file summarizes the design decisions behind the v59 journalistic sourcing annotation schema, developed in a Claude.ai Project (Markkula Center for Applied Ethics). It exists to give Claude Code the *reasoning* behind the rules, not just the rules themselves — the canonical prompt text lives in `system_prompt_v59.txt` / `system_prompt_v59.md` and `user_prompt_v59_csv.txt` / `user_prompt_v59_csv.md`. When those files and this one disagree on exact wording, the prompt files win; this file is for context on *why*, especially where a fix might look tempting but was already deliberately ruled out.

## What changed from v55 → v59, in one line

A new **Source Descriptors** field (6th CSV column) captures non-credentialing characterization words — role/action words for people ("protestor," "witness"), category/mission words for organizations ("nonprofit," "think tank"), genre words for documents ("affidavit," "memo") — that the old 5-field schema had no structured home for and that were either dropped, mis-annotated into Title of Source, or we had temporarily put in a rule to stuff them into Name of Source (most acutely for Unnamed Group of People). This was risking the field getting polluted with common-noun descriptors instead of holding only proper nouns.

## The credentialing / non-credentialing test (core logic)

Applies to Named Person, Unnamed Person, Anonymous Source, and Unnamed Group of People. For each characterizing word/phrase found for a source:

- **Credentialing** (→ Title of Source): denotes a formal position of power, authority, responsibility, expertise, or leadership — licensed profession, elected/appointed office, institutionally recognized expertise. E.g., director, senator, spokesperson, analyst, "leader" (even informal community leadership counts — it carries real moral/social authority).
- **Non-credentialing** (→ Source Descriptors): denotes what the source is doing in the story — enacting agency, individually or as a community (including democratic agency) — their relationship to someone else, or an informal social identity. E.g., protestor, activist, mother, resident, witness, former felons.
- Family/social-relationship words (mother, father, wife, sister, etc.) are **never** valid Title of Source, even directly adjacent to a name.
- Title of Source and Source Descriptors are **independent, not mutually exclusive** — a source can have both (e.g., an AI researcher who is also described as an activist gets both fields populated).

For Named Organization and Document, there's no credentialing test at all — Title of Source never applies to these two types; any category/function/mission word (org) or genre/type word (document) goes straight to Source Descriptors.

## The atomic-word rule for Source Descriptors

Capture the bare operative word or a short, tightly-bound conventional label — not a fully elaborated descriptive clause.
- "witnesses with criminal records" → Source Descriptors is **"witnesses"** (the operative class word); "with criminal records" stays out (it's elaborating detail, not part of the label).
- "20 former felons" → Source Descriptors is **"former felons"** whole — "former" and "felons" function together as a single conventional term, unlike the witnesses case.
- Rule of thumb: prefer the shorter atomic word when in doubt, but don't break apart genuinely bound compound labels.
- **Resolved: "employees and advisers" splits across both fields** — "employees" → Source Descriptors (non-credentialing, same generic-noun issue as bare "people"), "advisers" → Title of Source (credentialing — a real, formal role with genuine influence). Both apply simultaneously, not either/or, using the same independence principle already established for "AI researcher and activist."
- Generic count-nouns ("people," "person," "individuals") alone, without a distinguishing category word, are **not** meaningful Source Descriptors — they carry no discriminating information the way "witness" or "protestor" does. Leave Source Descriptors null in these cases even when the surrounding phrase ("people familiar with the deliberations") is otherwise substantive enough to be valid Source Justification.

## Source Descriptors and Source Justification are independent, and may overlap

Extracting a Source Descriptor never shortens, truncates, or nulls Source Justification. Source Justification is always extracted in full by applying its own definition, regardless of what Source Descriptors found in the same passage. The same word can legitimately appear in both fields (e.g., Source Descriptors: "witnesses" / Source Justification: "Three of the witnesses with criminal records..."). Do not treat this as redundancy to be cleaned up.

Source Justification also does not require its own attribution verb, and does not need to sit next to the quote — it can be pulled from anywhere in the article (including the opening sentence, before the source is ever named) as long as it explains why the source matters to the story.

## The Anonymous Source / Unnamed Person boundary — deliberately narrow, do not loosen

This is the rule most likely to look "wrong" on a spot-check and tempt a fix. **It is not a bug.**

The rule: We are now defining Anonymous Source as one that requires an actual disclosure in the text — explicit ("spoke on condition of anonymity") or a blanket statement *earlier* in the story covering multiple sources. If no such disclosure exists anywhere for a given source, it is Unnamed Person (or Unnamed Group of People), regardless of how the sourcing language reads ("people familiar with the matter," "sources said," etc.).

**Real GT annotators do not currently follow this strictly** — evidence from two independently-tested stories (an OpenAI board-firing piece and an Apple internal-culture piece) shows GT consistently classifies *every* unnamed-sourcing instance in a piece as Anonymous Source once *any* explicit disclosure appears anywhere earlier in that piece, even for later, different, undisclosed groups. This is exactly the propagating, "it reads anonymous so I'll call it anonymous" pattern the strict rule is designed to *count*, not accommodate. As part of v59, the GT dataset will also be updated to comply with the Anonymous Source vs Unnamed Person line.

Explicit design decision: **the rule stays narrow on purpose.** The goal of the schema is to measure how transparent news organizations actually are about disclosing anonymity — granular, per-instance disclosure vs. loosely running with unnamed sourcing, letting the reader infer anonymity. Loosening the rule to match current GT/annotator practice would destroy the exact measurement the schema exists to produce. **Do not "fix" this by making Note 4 (system prompt) more permissive.** The correct fix runs the other direction — see punchlist below.

## GT dataset upgrade/migration from schema v55 to v59/60

This is the actual step-by-step process for upgrading a batch of GT stories
from the older 5-field schema (v55) to the current 6-field schema (v59/60).
It's written down here because it was reconstructed after the fact from the
GT-2026 (stories 1-43) migration commit history rather than planned out in
advance — this section is what should be *followed from the start* for
every subsequent batch (GT-II and beyond), instead of re-deriving it again.

Student annotators should read this before doing a final human review pass
on files that have gone through this process: it explains what kinds of
changes to expect and why, and where genuinely uncertain calls get flagged
rather than silently decided.

### Phase 0 — Structural setup (no story text needed yet)

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

### Phase 1 — Anonymous Source reclassification (corpus-wide, before anything else)

For every row currently typed Anonymous Source, check for an actual
disclosure statement in the article text — explicit ("spoke on condition of
anonymity") or a blanket statement earlier in the story covering multiple
sources (see "The Anonymous Source / Unnamed Person boundary" above for the
full reasoning on why this stays narrow). No disclosure found → reclassify
to Unnamed Person (or Unnamed Group of People if the source is a group).

This runs across the *whole* batch before any Source Descriptors work,
since it determines which canonical type a row even belongs to before the
next phase's type-by-type work begins.

### Phase 2 — Source Descriptors population, one Type of Source at a time

Done in this order deliberately — simplest/fewest-rules types first,
highest-volume/most-nuanced type last:

1. **Unnamed Group of People** — move any common-noun values sitting in
   Name of Source (a legacy v55 misuse — this field must be null for this
   type) into Title of Source or Source Descriptors correctly, per the
   credentialing test; backfill Source Descriptors from Source
   Justification text where the descriptor word is present there but
   wasn't captured into its own field.
2. **Unnamed Person** — same credentialing-test-based population of Title
   of Source / Source Descriptors.
3. **Document** (3 passes): Pass 1 — fix Name/Title of Source misuse
   (schema violations) and populate Source Descriptors; Pass 2 — recover
   any missed Name of Source values by re-checking the article text for a
   stated document title; Pass 3 — recover genre-word descriptors (memo,
   report, lawsuit, etc.) from Sourced Statement/Source Justification text
   not yet captured.
4. **Named Organization** (3 passes): Pass 1 — schema-violation audit
   (clear invalid Title of Source values, since this type never carries
   one; recover full organization names from text; reclassify informal/
   generic references — e.g. "police" with no formal name stated at that
   point — to Unnamed Person/UGOP rather than inferring the formal name
   from elsewhere in the article; split rows jointly crediting multiple
   organizations into one row per organization; fix naming
   inconsistencies); Pass 2 folded into name recovery; Pass 3 — bare
   category-word Source Descriptors population (e.g. "nonprofit," "app,"
   "think tank" — not fuller function/mission language, which belongs in
   Source Justification instead).
5. **Named Person** (4 passes, largest volume): Pass 0 — general
   annotation-error audit (formatting artifacts, Name/Title of Source
   mix-ups, full-name recovery, joint-credit row splits, correcting
   over-inferred titles); Pass 1 — credentialing-test audit on Title of
   Source (non-credentialing words wrongly sitting in Title of Source,
   including organizational-affiliation text that belongs in Source
   Justification instead); Pass 2 — targeted missed-title recovery on
   specific flagged candidates; Pass 3 — systematic Source Descriptors
   recovery from Sourced Statement/Source Justification text, done in
   batches.

### Running practice throughout all of the above

Not a discrete step — applies constantly across every phase:

- **Always check the actual article text, not just the annotation cells.**
  Most of the real fixes above (recovering a missed title, confirming an
  organization's formal name is or isn't actually stated, judging whether
  a descriptor word is genuinely present) require reading the source
  article, not just reasoning from what's already in the CSV row.
- **Log real findings as they're found, rather than deciding silently.**
  Genuine ambiguities, schema gaps, or new design questions get written
  down (in this file's punchlist, for the internal/schema-facing side of
  this work) instead of being resolved unilaterally mid-pass. If a batch
  turns up a pattern not covered by an existing rule, that's a signal to
  stop and flag it, not to guess and move on.
- **Every fix should be traceable to specific reasoning**, not just a
  changed value — this is what lets a human reviewer (including the
  original student annotator, doing a final check on their own story)
  understand *why* something changed, not just *that* it changed.

## Migration punchlist

### Migration tasks

1. **[PARTIAL — done for GT-2026, not yet for the 50-150 batch] Reclassify GT Anonymous Source rows → Unnamed Person or Unnamed Group of People** wherever no qualifying disclosure (explicit or blanket-earlier-in-story) exists for that specific source. Done for GT-2026: 16 of 17 Anonymous Source rows reclassified to Unnamed Person (all confined to 2 stories — `32-openai_board.csv`, `36-Whats-Wrong-With-Apple.csv`); 1 row confirmed as genuine Anonymous Source with a real disclosure. All 7 pre-existing Unnamed Person rows checked for the reverse error (none found). Still pending for the newer 50-150 story batch once that GT exists.
2. **[DONE — all 6 canonical source types checked for GT-2026, not yet for the 50-150 batch] Populate the new Source Descriptors column** by re-deriving values from existing Title of Source / Name of Source / Source Justification text, applying the credentialing test and the atomic-word rule above.
   - Unnamed Group of People (58 rows), Unnamed Person (23 rows), Document (71 rows, across 3 passes — Name/Title-of-Source misuse cleanup, missed-title recovery, genre-word recovery), Named Organization (109 rows, across 3 passes -- schema-violation audit, name recovery, category-word recovery, including 2 Note-12 joint-org splits), Named Person (384 rows, across 4 passes -- Pass 0 annotation-error audit, Pass 1 Title of Source credentialing-test audit, Pass 2 missed-title recovery, Pass 3 Source Descriptors recovery from SS/SJ text).
   - Anonymous Source: only 1 row exists in all of GT-2026 after the item-1 reclassification (`36-Whats-Wrong-With-Apple.csv` row 3). Explicitly checked -- "three people familiar with the project" is a bare generic count-noun phrase with no distinguishing category word, so per the atomic-word rule's generic-noun exclusion, null is the verified-correct answer, not an unchecked gap.
   - Several real schema gaps and prompt-clarity issues surfaced along the way, logged individually in the Prompt Updates / Prompt Development / Schema gaps sections below rather than restated here.
   - Not yet started for the 50-150 batch once that GT exists.
   - Not yet started for the 50-150 batch.

### Data quality findings

3. **[OPEN] GT housekeeping fixes** (lower priority, human review needed, do not auto-migrate):
   - Multi-source-per-sentence rows crediting two organizations jointly (e.g., "Bloomberg and The Information") — open question whether Note 12's one-row-per-source split rule applies to organizations or only individuals.
   - Rows attributing a description of a company presentation/demo to the company as a Named Organization source — likely miscategorized; probably reporter first-hand observation (not attributable to a source at all) rather than a sourced statement.
4. **[RESOLVED] Found and documented the canonical source-article-text folder — `extracted_articles_boilerplate/`, not `2025_input_stories/`.** Story text lived in four different folders with confusing overlap: `extracted_articles/` (stories 1-30, body only, no metadata header), `extracted_articles_boilerplate/` (stories 1-43, complete, body + Headline/Subtitle/Date/Publisher metadata header -- "boilerplate" here means the metadata header, not junk/ad text, a misleading name), `2025_extracted_articles/` (stories 35-43 only), and `2025_input_stories/` (a handful of files). Audited all overlaps by diff: every file in `2025_extracted_articles/` and every file in `2025_input_stories/` is byte-identical to its counterpart in `extracted_articles_boilerplate/`. Checked `v10-extract-multiple-LLMs.py`'s git history (and every untracked script variant -- sv8/sv9/sv10/cld/gpt/test): `input_dir = "2025_input_stories"` has been hardcoded since the earliest tracked commit, confirming `2025_input_stories/` is a **transient staging folder** the extraction script globs `*.txt` from at runtime, not a stable corpus store -- the workflow is to copy in whichever story/stories you want to run before invoking the script, which is why it currently holds only a leftover partial subset while `llm_results/` has historical output for all 43 stories. Documented this in `CLAUDE.md`'s Directory Layout and Pipeline Architecture sections. Going forward, `extracted_articles_boilerplate/` is the one canonical folder for story text -- the expanded 44+ corpus should land there too, and nothing should be added to or read from `2025_input_stories/` as if it were a corpus.
5. **[RESOLVED] `GT-2026/1-Harris-Poll.csv`'s two Document-typed rows (1, 14) were a real inconsistency, fixed.** Validated by running a full fresh v59 annotation of the story's source article from scratch (system_prompt_v59 + user_prompt_v59_csv, as if annotating for the first time) and comparing it row-by-row against the existing GT. Confirmed: a poll conducted and published by a named organization under its own name is Named Organization per the Name of Source definition ("If a named organization formally releases or stands behind a document in its own name...the Type of Source is Named Organization, not Document"), matching how all the *other* Reuters/Ipsos and Bloomberg News/Morning Consult rows in the same story were already typed. Rows 1 and 14 were retyped Document -> Named Organization, Name of Source -> "Reuters/Ipsos". Source Descriptors intentionally left null for all Reuters/Ipsos and Bloomberg News/Morning Consult rows in this story -- the article never gives either organization a category/mission descriptor (e.g., "a polling firm"), and "poll" itself is document-genre language, not an org-category descriptor, so it doesn't populate Named Organization's Source Descriptors field.
6. **[OPEN] Edge case: UGOP row with no atomic descriptor word in its Source Justification.** `GT-2026/40-SeattleProtest.csv` row 1 (Type: Unnamed Group of People) has Source Justification "The group that organized a takeover protest of a building on the University of Washington campus earlier this week" — this clearly describes the source, but there's no bare atomic word in it the way "witnesses" or "protesters" would be (no single token like "organizers"). Left Source Descriptors null rather than coining a word not actually in the text. Needs a human call: leave null, or is inferring "organizers" from "the group that organized..." within the spirit of the atomic-word rule?
7. **[OPEN] SS segmentation rule check (prevalence unknown, needs corpus-wide review before deciding fixes).** The story-1 fresh-annotation experiment (see item 5) surfaced two segmentation conventions where the existing GT diverges from the system_prompt_v59 rule, independent of any Type/Name error:
   - Contiguous sentences in the same paragraph attributed to the same source should be one Sourced Statement instance per the system prompt, but GT sometimes splits them into separate rows (e.g., story 1's rows 2+3, which a strict v59 reading merges into one).
   - A paragraph mixing sourced content with un-attributed reporter background/analysis should have only its sourced portions extracted as Sourced Statements; GT sometimes bundles the whole paragraph (background sentences included) into one row (e.g., story 1's row 15, which a strict v59 reading splits into two rows with the two background sentences dropped entirely).
   Before deciding whether/how to correct these, need to sample across GT-2026 (and eventually the 50-150 batch) to gauge how common each pattern is -- this may be a large-scale re-segmentation effort or a rare one.
8. **[OPEN] SJ review and fixes (corpus-wide review needed).** Story 1 row 16's Source Justification ("The poll was conducted nationally and gathered responses from 4,253 U.S. adults, including 3,562 registered voters.") is a standard polling-methodology disclosure -- every pollster discloses sample size, dates, and margin of error; it isn't unique context explaining *why Reuters/Ipsos is a source in this story* (Note 21's actual test), it's just generic downstream disclosure that any polling story would carry regardless of which specific poll or story it's about. The story's actual justification for citing this poll is simply its newsworthiness as the story's subject, not this methodological boilerplate. This SJ annotation is very likely incorrect and needs a fix, but flagging as its own review category (distinct from the SS segmentation category above) since it may recur elsewhere in the corpus wherever methodology/process disclosures got annotated as Source Justification instead of being recognized as boilerplate.

### Schema gaps (found during passes -- v59's current fields have no clean home for these; cleared for now, revisit if a field/rule addition is warranted)

9. **[OPEN] Unnamed role-word of a named organization's representative has nowhere to go.** `GT-2026/7-District7-Candidate-Sues.csv` row 6: Type Named Organization, Name of Source "Alameda County" (the county attorneys' brief is formally a county document, "Alameda County" being everyday shorthand for the County of Alameda, the government entity), but "County Attorneys" was sitting in Title of Source -- invalid, since Named Organization never carries a Title of Source. The word isn't useless, though: it's what led the annotator to identify Alameda County as the source (the same pattern as "an Exxon Mobil spokesperson" or "county officials" -- Note 8's unnamed-representative cases). But Source Descriptors for Named Organization is defined as the *org's own* category/mission (e.g., "nonprofit," "think tank"), not the role-title of whoever is speaking on its behalf -- so the role word doesn't fit there either, and there's no other field it belongs in under the current schema. Cleared Title of Source (the only rule-compliant action available) rather than route it somewhere that doesn't fit; the word itself is lost from structured data as a result. Revisit if this pattern turns out to recur often enough to warrant a schema addition (e.g., an explicit "unnamed representative role" sub-field for Named Organization).

10. **[RESOLVED — decision: no new canonical type yet, defer pending prevalence]** When a Sourced Statement is jointly attributed to a mix of named and unnamed organizations, there's no Type of Source for the unnamed remainder — Unnamed Group of People is defined for people, not organizations, so it doesn't cover a residual group like "and more than 20 other companies." **Real example (test annotation, not GT):** RAI-004 (2026-09 CNBC test-annotation, distillation-policy article) — "tech giants Nvidia, Microsoft, Meta, Palantir joined with more than 20 other companies to release a letter... 'Distillation... is a widely used technique...,' they wrote." Decision: apply the Note 12 one-row-per-named-source rule to the four named companies (Nvidia, Microsoft, Meta, Palantir) and leave "more than 20 other companies" unannotated, rather than inventing an "Unnamed Group of Organizations" canonical type. Reasoning: this is the first observed instance of the pattern; adding a new type on a single data point risks a type nobody else needs. Revisit only if this pattern turns out to be prevalent once more of the corpus (GT-2026 already migrated, the 50-161 batch, and further test annotations) has been reviewed.

### Prompt Updates checklist (blocked / pending)

11. **[DONE — drafted into system_prompt_v60] Draft, and eventually add to system_prompt_v59, a general principle for why a single source (individual or group) may carry multiple simultaneous descriptors across fields** — currently only demonstrated through worked examples ("AI researcher and activist," "employees and advisers"), not stated as a principle. Needs careful wording: do NOT frame it as "reporters routinely give sources rich, multi-faceted portrayal" — that asserts good practice as the norm when it isn't. Journalism has a real, live contest here: some reporters reduce sources to flattening, reductive labels (e.g., "far-left activist"), others give fuller, multi-faceted treatment. The schema's job is to capture *all* usages that map into each column so that downstream analysis can distinguish reductive/flattening source treatment from richer, more democratic portrayal — that's the actual reason multi-field capture matters, not an assumption that reporters already do it well.
12. **[DONE — drafted into system_prompt_v60 and user_prompt_v60_csv] Add carry-forward instructions for Title of Source — not implied anywhere in v59 or earlier prompt versions.** Note 22 explicitly instructs carrying forward Source Justification from an earlier sourced statement by the same source when the current instance has no SJ text of its own ("you may copy the Source Justification from the source's earlier sourced statement to this one"). There is no equivalent instruction anywhere for Title of Source: when the same source recurs across multiple sourced statements and a later instance doesn't restate their title, the prompt gives no guidance on whether to carry the title forward the same way. Draft and add an equivalent carry-forward rule for Title of Source, modeled on Note 22's treatment of Source Justification.
    - **Real worked example confirming the gap:** `GT-2026/25-ballot-access-trans.csv` row 6. The Sourced Statement is "He described Democratic vice presidential nominee Tim Walz recently as 'very heavy into the transgender world.'" — no name, no title, just "He." The article names him "former President Donald Trump" once, several paragraphs earlier; by row 6 he's referred to only by pronoun, with "Trump" used by name in the adjacent Source Justification text. Name of Source was originally "former President" (a title sitting where a name belongs) — fixed to Name of Source "Donald Trump", Title of Source "former President". The Title of Source definition itself is silent on whether a title established once, early in the article, should carry forward to every later instance of that same source — unlike Source Justification, whose definition and Note 20 explicitly say it "maybe earlier in the text or in some other sentence," and whose Note 22 gives an explicit carry-forward rule. That asymmetry is exactly why this row's fix required a judgment call rather than a rule lookup.
    - **Proposed addition to v59, in the user's own words (verbatim, to be adapted into prompt language when this item is drafted):** "Title of source is pulled exactly this way - the annotator has to see where the person is first formally introduced and catch it there, or scan downstream from there (occasionally reporters use fiction-writing narrative style where they might introduce with title later, but on the first occasion the source may be quoted just by name too. It's rare. But in any case Title of Source tends to be a global value for the article."
13. **[RESOLVED — decision drafted into system_prompt_v60 as Note 13]** `GT-2026/14-many-police-calls.csv` row 8 said "According to police..." with no department name in that sentence, but "Brattleboro Police Department" is named explicitly elsewhere in the same article -- the original fix (during the Named Organization pass) recovered the full name by inference across sentences. On review, decided **against** allowing this kind of context-based name inference at all, even though it's logically obvious to a human (or likely an LLM) in this specific case. Reasoning: (a) doing it anyway, at scale, risks false positives an LLM can't reliably avoid; (b) more importantly, staying strict on manifest text preserves a measurable signal this schema is designed to capture -- how consistently reporters formally attribute claims to institutions versus relying on informal, implied references to institutional authority. Silently inferring the formal name would erase exactly the "informal deference to institutional power" pattern the schema exists to count. This is worth deliberately re-examining later as a narrower, upgraded-capacity benchmark focused specifically on this class of decision -- separate from the main v59 annotation task -- to see how much everyday reporting actually relies on this kind of implied/informal institutional reference, and whether an inference override would wash that signal out.
    - **Reverted the original fix for consistency:** `14-many-police-calls.csv` row 8 changed back from Named Organization / "Brattleboro Police Department" to Unnamed Group of People / Name of Source null / Title of Source "police" -- matching the treatment already given to the Seattle DogWalker case (`37-DogWalker.csv`, finding C), where no organization name existed anywhere in the article at all. Both cases now get identical treatment regardless of whether the formal name happens to exist elsewhere in the piece.
    - **Proposed prompt addition (brief, paired conceptually with Note 8):** "Do not infer an organization's formal name from elsewhere in the article when a specific attribution only refers to it informally or generically (e.g., 'police,' 'the department,' 'officials'). Classify each attribution using only what is actually stated at that point in the text -- if no organization name is given there, this is not a Named Organization instance; apply the Unnamed Person or Unnamed Group of People rules instead, even if the same organization happens to be named formally somewhere else in the article. This preserves a measurable signal: how consistently reporters formally attribute claims to institutions versus relying on informal, implied references to institutional authority."
14. **[DONE — added to system_prompt_v60] Add "three-panel recount court" as a worked UGOP Source Descriptors example in system_prompt_v59.** `GT-2026/8-primary-recount-complete.csv` row 1: Type Unnamed Group of People, Title of Source "judges" (credentialing), Source Descriptors "three-panel recount court" (see finding D). This isn't an agency/identity word like the prompt's existing UGOP non-credentialing examples ("players, teachers, protestors, attendees, advocates, activists, participants, onlookers, commenters") -- on reflection, though, it still fits the *definition* squarely: the panel was convened specifically to do the recount, so "what the group is doing in the story" is exactly what "three-panel recount court" names. The prompt's example list just doesn't currently illustrate this institutional/convened-body flavor of "what the group is doing," only informal-identity/agency words. Add this as a worked example so the definition's own scope (already broad enough to cover it) is demonstrated, not just stated.
15. **[DONE — drafted into system_prompt_v60 and user_prompt_v60_csv] Named Organization's Source Descriptors definition conflates category, function, and mission -- resolved to a two-bucket split, dropping "mission" as its own category entirely.** Surfaced by `42-Uber-Lyft-CA.csv` row 8 (Gridwise): the definition says a descriptor can name the org's "category, function, or mission," and its own worked example ("an American social media service") is already a multi-word descriptive phrase -- this reads as license to capture full functional elaboration (e.g., "an app that helps drivers track mileage and optimize earnings"), directly in tension with the general Source Descriptors rule elsewhere in the prompt that says capture "the bare operative word... not a fully elaborated descriptive clause."
    - **Resolved during drafting: collapse to two buckets, not three.** In every actual case worked through in this GT-2026 pass, "function" and "mission" received identical treatment -- both got redirected to Source Justification, never compressed into Source Descriptors. Only "category" genuinely behaves differently (atomic, → SD). Keeping "mission" as a named third bucket added a judgment call (is this function or mission?) that never changed the outcome, so it's dropped entirely rather than given its own worked examples. The user's framing: even a short, catchy mission statement (e.g., an org's tagline like "Spread Ideas" or "finding cures, saving children") is still almost always a better fit for Source Justification than Source Descriptors -- reporters tend to quote or paraphrase mission language selectively depending on context, not use it as a fixed, atomic identifying label the way a category word functions. Fewer named sub-types should mean fewer edge-case judgment calls for automated annotation.
    - **Category** words (what kind of entity it is: nonprofit, utility, coalition, foundation, app, think tank, state agency) are naturally atomic or near-atomic -- one or two words, same level of abstraction as "nonprofit." This is the only sub-type that populates Source Descriptors.
    - **Everything else** describing what the org does or why it exists -- however phrased, whether it reads as "function" or "mission" -- belongs in Source Justification instead, not Source Descriptors. If no bare category word is also present in the text, leave Source Descriptors null.
    - **Audited all 9 currently-populated Named Organization Source Descriptors values in GT-2026 against this tightened standard — all clean, no cleanup needed:** "nonprofit sentencing advocacy group" (Sentencing Project), "utility" x3 (Southern California Edison), "nonprofit" (OpenAI), "state agency" (California Coastal Commission), "mix of more than 100 governments, non-governmental organizations and others" x2 (International Coral Relief Initiative), "app" (Gridwise). Every one is a genuine category/composition description, none is mission language that had snuck in.
    - **Fixed a related gap found while drafting this item:** Gridwise's Source Justification was empty even though the functional description ("an app that helps drivers track mileage and optimize earnings") is right there in its own Sourced Statement text -- fixed to match the precedent already set by the Movement Advancement Project row (`25-ballot-access-trans.csv` row 5), whose Source Justification is likewise just the descriptive clause copied out of its own Sourced Statement.
    - **Two more spots needed the same tightening, found after the main definition was already fixed:** (1) Note 2 (under the Named Organization type definition itself) still said "category, function, or mission" -- same fix applied, dropping "mission" and pointing the non-category description to Source Justification. (2) The Source Justification definition's own worked examples only ever exemplified Named Person content (lived experience, expertise, felony conviction, lawmaker background) -- it never gave Named Organization its own example of what typically lands there, even though Source Descriptors' definition cross-references Source Justification as the destination for non-category org description. Added a Named-Organization-specific example (the Gridwise "app" / "an app that helps drivers track mileage and optimize earnings" split) directly to the Source Justification definition so that field's own text demonstrates the pattern instead of only being inferable from the other field's definition.
16. **[DONE — drafted into system_prompt_v60 and user_prompt_v60_csv] Add carry-forward instructions for Source Descriptors — same underlying gap as item 12's Title of Source carry-forward, not implied anywhere in v59.** Note 22 only covers Source Justification carry-forward (copy from an earlier sourced statement by the same source when the current instance has none). Title of Source and Source Descriptors both lack an equivalent rule. No confirmed real example of this specific gap has turned up yet in the GT-2026 pass (a candidate from `32-openai_board.csv` row 1 turned out, on closer check, to be a direct in-row extraction rather than an actual carry-forward instance — logging the gap anyway since the reasoning holds independent of that one row). Whenever a real carry-forward rule is drafted for Source Descriptors, it needs a deliberate override principle, not a blind copy.
    - **Refined framing (connects this item to item 11's multi-faceted-portrayal principle):** the override principle should treat **accumulation as the default, replacement as the exception**. If a source is called "protestor" early in a story and "artist" later, both are very likely still true at once -- the reporter is revealing another facet of the same person, not correcting an earlier label. The default behavior should be to add the new descriptor alongside the carried-forward one (comma-separated, e.g. "protestor, artist"), the same way this GT-2026 pass already accumulated multiple descriptors found together in one row (e.g., Roger Clark: "friend, programmer, DDR enthusiast"). Replacement should be reserved for the narrower case where the source's actual status has genuinely changed between mentions (e.g., "candidate" -> "councilmember" after an election result reported later in the same piece) -- not treated as an equally-likely default alongside accumulation.
    - **This is the same underlying idea as item 11, at a different scope:** item 11 is multi-faceted portrayal *within one row* (Title of Source + Source Descriptors both populated for the same sourced statement, e.g. "AI researcher" + "activist"); this item is multi-faceted portrayal *accumulated across a source's rows over the course of the story* via careful carry-forward. Both exist to serve the same measurement goal -- capturing whether reporters give sources rich, multi-dimensional treatment or flatten them to a single label -- and should probably be drafted together, or at least cross-reference each other, rather than as two unrelated notes.
    - Also carries the same moment-specific caution as before: a non-credentialing descriptor can describe what a source was doing in a specific moment (e.g., "protestor" at a particular rally) and may not still apply later -- when in doubt whether an earlier descriptor still holds, don't force it in the "carried forward and no longer true" direction; leave it out rather than accumulate something no longer accurate.

17. **[DONE — drafted into system_prompt_v60 (Note 19) and user_prompt_v60_csv (Step 7)] Title of Source needs an explicit no-contextual-inference guardrail — a real, demonstrated failure mode, not a hypothetical one.** Source Descriptors already has this instruction explicitly ("Capture the Source Descriptor exactly as the reporter wrote it — do not normalize case, pluralize, or singularize it"), and Note 13 warns against supplying a Document's publishing organization name the reporter didn't actually attribute. Title of Source has no equivalent: nothing in its definition or extraction steps says a title must be stated in connection with *that specific source* somewhere in the text, as opposed to being contextually plausible from the story's general topic, its headline, or another similarly-described source's title.
    - **Real worked example, caught twice in a row:** `GT-2026/43-Chicago-immigration.csv`, Carla Espinoza. Her own introduction in the article says only "Espinoza, who was sworn in as a judge in 2023..." — the word "judge," never "immigration judge," anywhere attached to her. But the original human GT annotator had her Title of Source as "immigration judge" across 5 rows -- a plausible inference given the whole story is about immigration judges and the other named judge (Jennifer Peyton) is explicitly titled "former assistant chief immigration judge" -- but an inference nonetheless, not a captured fact. When splitting a joint-credit row for Espinoza during this pass, the carry-forward principle (item 12) was applied by copying this already-established "immigration judge" value from elsewhere in the same file -- reproducing the over-inference a second time, this time by an LLM (this session) trusting an existing GT value rather than checking it against Espinoza's own literal introduction. Caught only because the user asked a followup question about the inference. Fixed to "judge" across all 5 rows once checked.
    - **Fix needed:** add explicit language to the Title of Source definition and/or extraction steps: only extract a title actually stated in connection with the specific source being annotated; do not borrow, infer, or generalize a title from the story's overall subject, headline, or another source's title, even when highly plausible. Use the Espinoza case as the worked example. This guardrail also needs to apply *before* any future carry-forward rule for Title of Source (item 12) is used -- carry-forward should only ever copy a title that was itself correctly, verbatim-sourced the first time, not propagate an inference.
    - **Carve-out needed, found immediately after drafting the guardrail above:** `GT-2026/32-openai_board.csv`, Ilya Sutskever, row 17. The article says (same paragraph flow, one sentence apart): "...push out one of **the board's members**..." followed immediately by "Another **member**, Ilya Sutskever, thought Mr. Altman..." Bare "member" alone is not credentialing (a "member of the audience" isn't a title) -- the word only becomes "board member" by resolving what "another member" is pointing back to. This is different in kind from the Espinoza case: Espinoza's "immigration" came from the story's general subject and a different person's separately-stated title, with no direct textual antecedent pointing at her specifically. Sutskever's "board" comes from an explicit antecedent one sentence earlier, naming the exact group he's "another member" of. Resolved: Title of Source -> "board member" (not left as bare "member" in Source Descriptors). In plain terms for a human annotator: **the guardrail bans borrowing a qualifier from the story's general topic or from a different person's title; it does not ban resolving a plain "another one," "she," "the other one," etc. back to something the sentence right next to it already named for that specific source.** That's just reading the sentence, not inferring beyond it. Needs careful, plain-English wording when this actually gets drafted into the prompt -- flagging the distinction here, not prescribing the final phrasing.
18. **[DONE — drafted into system_prompt_v60 and user_prompt_v60_csv] State explicitly where a non-credentialed source's organizational affiliation belongs: usually Source Justification, not Source Descriptors.** Source Descriptors is for the bare non-credentialing word only (e.g., "organizer," "parent") -- it should never carry a proper-noun organization name comma-attached to it, since that pollutes the field with proper nouns the way Name of Source is specifically protected against for Unnamed Group of People. **Resolved via signal-scan research (Named Person pass 1, items 1 and 6):**
    - Janice Guzman (`16-America's-Sleeping-Giant.csv`): Title of Source was "Organizer, Massachusetts Poor People's Campaign" -- cleared; Source Descriptors -> "organizer"; the org affiliation (plus a second org, SEIU 1199, found in her clean introduction sentence one paragraph earlier but never previously captured anywhere) recovered into Source Justification, which was empty: "an organizer with the Massachusetts Poor People's Campaign and SEIU 1199". The parent org ("Poor People's Campaign") already exists as its own Named Organization row elsewhere in this file, but the specific state chapter and the second org did not exist anywhere until this fix.
    - Marina Muñoz (`6-OUSD-basic-job.csv`): Title of Source was "Marina Muñoz, organizer, Communities for a Better Environment." (her own name erroneously embedded) -- cleared; Source Descriptors -> "parent, organizer"; the org affiliation was already correctly present in her existing Source Justification ("parent at Madison Park Academy and an organizer with Communities for a Better Environment.") -- no new SJ needed, just stopped duplicating it badly into Title of Source.
    - **Fix needed:** add explicit language (Source Descriptors definition and/or Source Justification definition) stating that when a non-credentialed source's organizational affiliation is given in the text, it typically belongs in Source Justification rather than Source Descriptors, since reporters usually explain a source's presence/stakeholdership using language that independently satisfies the Source Justification definition anyway. This is **not a hard guarantee** -- unstructured news writing is not perfectly predictable, and there will be cases where the affiliation appears in text that doesn't cleanly fit the Source Justification definition (mere attribution per Note 20, for instance). The instruction should describe the *usual* pattern and give annotators/LLMs a default expectation, not an absolute rule.
    - **Item 5 (Paul Boden, `5-Humboldt-SF-homeless-bus.csv`) also resolved via signal scan, same pattern.** Checked whether "advocate" here could denote a credentialed role (e.g., a lawyer) given the homelessness-nonprofit context -- his full introduction in the article is "Paul Boden, a homeless advocate with the Western Regional Advocacy Project," nothing more; no formal position (director, attorney, etc.) is stated anywhere for him in the piece. Per the same manifest-text-only principle as item 17 (Espinoza), stuck with "advocate" as non-credentialing, same bucket as "activist." "Western Regional Advocacy Project" appears nowhere else in the file (matching the Muñoz pattern, not the Guzman one) and Source Justification was empty for all 3 rows -- so, unlike Muñoz, this needed genuine recovery, not just de-duplication. Title of Source cleared across all 3 rows; Source Descriptors -> "homeless advocate" (kept whole rather than trimmed to bare "advocate" -- "homeless" specifies which kind of advocate, a distinguishing qualifier rather than padding, same logic as keeping "Federal" in "Federal contracts"); Source Justification (previously empty) -> "a homeless advocate with the Western Regional Advocacy Project" for all 3 rows.

19. **[DONE — drafted into system_prompt_v60 (Note 15) and user_prompt_v60_csv] Add "paralegal" as a non-credentialing Source Descriptors example.** `GT-2026/18-wyoming-primary-dems.csv`, Becky Blackburn (Named Person pass 2, candidate 1): Source Justification calls her "A paralegal for the Republican county attorney." Initially proposed as a Title of Source recovery (a defined professional legal-office position, on the same reasoning as "legal fellow, Public Counsel" being credentialing), but on reflection this doesn't hold: a paralegal doesn't carry the licensed authority the attorney they work for has -- no independent standing to represent clients in court, give formal legal advice, or otherwise exercise the licensed profession's authority. It's a real job with real duties, but not itself a credentialing position. Moved to Source Descriptors ("paralegal") instead, org affiliation left in Source Justification (already correctly there, no change needed there). Add "paralegal" to the Source Descriptors examples list, and note the same non-credentialing reasoning applies to other legal-adjacent support roles that lack independent licensed authority (e.g., legal secretary, law clerk) as a category distinct from the profession's own licensed practitioners.

### Prompt Development checklist (new schema/design questions, likely v60/v61 -- distinct from Prompt Updates above, which are refinements to existing v59 rules)

20. **[OPEN — design question, not yet scoped] Secondary-source signal for Named Organization ("reported earlier by," "first reported by," etc.).** `GT-2026/36-Whats-Wrong-With-Apple.csv` row 5/6 (see finding E): "Some details of Apple's changes to its Siri team and challenges were previously reported by Bloomberg and The Information" -- Bloomberg and The Information are named organizations, but specifically in the sub-case the system prompt already names as "secondary source" (another news organization the reporter is crediting for earlier reporting), distinct from an organization making its own statement or releasing its own material. v59 has no way to distinguish these two sub-cases of Named Organization from each other in the data today. The idea floated: a flag/signal on Named Organization rows triggered by cue text ("reported earlier by," "first reported by," "according to a report in," etc.) that, combined with Type = Named Organization, marks the row as a secondary-source citation. This is NOT yet designed -- open questions include whether this needs an actual schema change (a new column, or a controlled-vocabulary addition to Source Descriptors) or can be handled as a detection rule without changing the CSV shape at all, and how reliably an LLM could detect the cue text without false positives. Likely a v60/v61 feature, not v59.

21. **[OPEN — serious edge case, needs more real examples before a rule can be drafted] Document vs. Named Organization is unclear when a named organization's own research paper/report is used to source a claim about itself.** **Real example (test annotation, not GT):** RAI-004 (2026-09 CNBC test-annotation, distillation-policy article) — "Nvidia, for instance, used distillation as part of the training process for its Llama Nemotron series of models, as detailed in an accompanying research paper." Two readings collide: (a) "research paper" is exactly the genre word the Document type / Source Descriptors exists to capture, and the phrasing reads as the reporter citing an independent piece of research; (b) the Name of Source definition's rule — when "a named organization formally releases or stands behind a document in its own name... the Type of Source is Named Organization, not Document" — arguably applies here too, since the paper is Nvidia's own technical documentation about its own product, not a third-party or independently-obtained document. This is not a simple wording fix: it's genuinely unclear how much weight "independent research" should carry versus "company-authored material a company stands behind," and the same tension will recur for company blog posts, technical reports, whitepapers, and press-released studies about a company's own product or work. Needs a wider sample of real cases — across GT-2026, the 50-161 batch, and further test annotations — before a rule can be written; flagging now, not drafting a fix yet.

22. **[OPEN — new schema/design principle, needs guardrail sign-off before drafting into a prompt] An unnamed organizational spokesperson attribution can resolve backward to a later-named spokesperson for the same organization.** **Real example:** Story #109, "Waymo driverless cars are coming to the West Valley" (sanjosespotlight.com, from the user's "1-Examples for API completion" doc). First sentence: "A spokesperson said the latest expansion covers approximately 60 square miles..." — Waymo is named in the same sentence, but the spokesperson isn't, so under Note 8's plain reading this defaults to Named Organization / Waymo. The very next paragraph then names the speaker: "'...,' spokesperson Sandy Karp told San José Spotlight." Confirmed intentional: reclassify the *first* statement too, as Named Person / Sandy Karp / Spokesperson — not Named Organization / Waymo, even though her name hadn't appeared yet at that point in the text.
    - **Reasoning (user's, grounded in reporting practice):** a reporter is very unlikely to get access to two different spokespeople from the same company for the same story at the same time, and companies don't typically deploy multiple spokespeople for a single story even when they employ several spokespeople overall. So when exactly one spokesperson is later named for a given organization in a story, resolving an earlier unnamed "a spokesperson" mention for that same organization to that same named person isn't an unsupported inference — it reflects how organizational access to reporters actually works.
    - **Proposed guardrail (mine, to confirm before drafting into a prompt) — this needs to stay narrow, the same way the other inference guardrails (items 13, 17) do:**
      - Only applies to Note 8's generic-designation language ("a spokesperson," "an official," "a representative," etc.) attributed to a *specific named organization* — not to Unnamed Person/UGOP/Anonymous Source generally.
      - Only resolves backward when exactly **one** person is later named as speaking for that *same* organization in the story. If two or more differently-named people are credited as speaking for the same org, do NOT resolve backward — leave the earlier mention as Named Organization per Note 8's default, since which of them made the earlier statement is genuinely ambiguous.
      - Name of Source *and* Title of Source both resolve backward together (the later-identified "Spokesperson" title applies to the earlier row too) — this is the existing "Title of Source is a global value for the article" principle (item 12), just applied backward for this specific identity-resolution case rather than only forward.
      - Distinguish this from the no-inference guardrails (items 13, 17): those ban inferring a *more specific* qualifier than the text supports at that point (a fuller org name, a fuller title) from elsewhere in the story. This rule doesn't add specificity from outside the text — it identifies *who* an already-present generic role phrase ("a spokesperson") refers to, once the same story later removes the ambiguity. Closer in kind to the Sutskever coreference carve-out (item 17) than to the org-name/title inference bans.
    - **Open question for the user:** does this generalize beyond "spokesperson" to Note 8's other generic-designation words ("officials," "lawyers")? The access-norm reasoning is specific to a single company spokesperson; it's less obviously true for, say, a government agency's "officials" (often genuinely plural, several people who could plausibly each be quoted). Flagging rather than assuming either way.

## LLM-benchmark-2026 project checklist

Higher-level roadmap for after the GT-2026 (stories 1-43) migration work above is
finished. Story-number boundaries below are rough/provisional -- the final
cutoffs will be known once the expanded GT set is finalized. Not started yet;
capturing now so the plan is written down before it's needed.

1. **[DONE] Prompt revisions pass.** Worked through the accumulated Prompt
   Updates and Prompt Development checklist items (10-19 in the punchlist
   above) and drafted `system_prompt_v60`/`user_prompt_v60_csv`, using the
   user's own Word-doc track-changes workflow (system_prompt_v60.docx /
   user_prompt_v60.docx), then renumbered all Notes across both documents,
   fixed cross-references, and committed the finalized `.txt`/`.md` files to
   `new_prompts/`. Sanity-tested via a full manual test annotation (RAI-004,
   a CNBC distillation-policy article not in GT-2026) — see items 10 and 21
   above for the two real edge cases that surfaced. Items 20 (secondary-source
   signal) and 21 (Document vs. Named Organization for a company's own
   research paper) remain open design questions, deliberately deferred, not
   part of this pass.
2. **[DONE] Build the few-shot JSON payload for the OpenRouter API completions
   call, shipped alongside `system_prompt_v60`/`user_prompt_v60_csv`.** Built
   as `few_shot_examples/examples_bank.json` (13 real, correct-only examples,
   HYPOTHETICAL Example 4 excluded per decision), `few_shot_examples/
   build_payload.py` (reference request-builder, verified end-to-end against
   a real story), and `few_shot_examples/README.md` (design reasoning:
   OpenRouter's unified multi-turn API, message ordering, correct-only vs.
   contrastive tradeoff, and per-provider prompt-caching mechanics/support).
   Sent to the student for API testing 2026-09-08. `v59-worked-examples.md`
   now has 14 examples total (13 real + 1 hypothetical, unconverted); keep
   appending new ones there as they surface, converting into the bank
   afterward the same way.
3. **Bring benchmark code up to v59 schema compliance.** A student fixes the
   benchmark project's code -- both the LLM annotation extraction script and
   the evaluation script -- to handle the v59 6-field schema (the new Source
   Descriptors column, the narrower Anonymous Source/Unnamed Person boundary,
   etc.) instead of the older 5-field schema they currently assume.
4. **Re-run the benchmark on GT 1-43 only, on current models.** Once the code
   is v59-compliant, re-run annotation + evaluation against just the existing
   GT-2026 (stories 1-43) and reproduce/compare scores using current-generation
   models. This is a smoke test that the annotation and eval code both work
   correctly end-to-end on v59 -- deliberately done *before* bringing in the
   much larger expanded GT set, so any code bugs surface on a small, familiar
   dataset first.
5. **[IN PROGRESS] Migrate the GT-II batch (82 files / 71 distinct stories,
   roughly the story 50-161 range) to v59/60 schema.** This is the "50-161
   GT batch" referenced in earlier drafts of this roadmap, now concretely
   scoped: 82 XLSx files delivered by student annotators
   (`~/Documents/GT-II-finishedfiles-XLSx/`, not yet in this repo), 71
   distinct stories once de-duplicated, originally annotated under v55.
   Needs the same kind of migration work done on GT-2026 -- Anonymous Source
   reclassification (item 1), Source Descriptors population (item 2),
   schema-violation cleanup -- applied to all 82 files individually (not
   just 71), since 9 stories are double-coded by two independent annotators
   (AV and SZ) and both copies of each need migrating separately and blind
   to each other, not merged first. Explicitly watch for recurrence of items
   7 (SS segmentation), 8 (SJ methodology-boilerplate false positives), 9
   (unnamed org rep role), and 20/21/22 (secondary-source signal, Document
   vs. Named Org self-reference, spokesperson backward-resolution) while
   migrating -- use the same conservative manifest-text defaults already
   established for these rather than blocking on them; log real recurring
   instances, don't re-litigate the open design questions mid-migration.
   - **[DONE] ICR-coverage sanity check, done before migration starts.**
     Counted Sourced-Statement rows across the combined GT-I (43 stories,
     648 rows) + GT-II (71 stories, ~1,205 rows) corpus against the rows
     already double-coded for inter-coder reliability: 6 GT-I stories (#1
     Harris Poll, #4 SFO Labor Day, #11 Nebraska felons/voting, #31 Vermont
     bill, #36 Apple AI, #37 DogWalker) plus 9 GT-II stories (#52, #55, #76,
     #94, #108, #109, #112, #122, #125 -- confirmed via GT-II's file-naming
     convention: annotator AV always prefixes her filename with "#", SZ
     never does, so a story number appearing in both naming forms is a
     confirmed AV/SZ double-coded pair). Result: 15 of 114 combined stories
     (13.2%) and 220 of 1,853 combined rows (11.9%) are ICR-covered --
     comfortably inside the 10-20% academic coverage requirement, using
     either measure. User can expand GT-III's dual-coding deliberately if a
     future check comes in lower, rather than by default.
   - Convert all 82 XLSx files to CSV first (fixing the known
     `Type of source` -> `Type of Source` header-casing gotcha along the way
     -- see `inter_coder_reliability/CLAUDE.md`'s Data Files section for the
     same issue in the existing ICR CSVs).
   - One of us (not necessarily Claude Code) handles the substantive
     migration judgment calls, same as GT-2026.
6. **[PENDING, after step 5] Update the ICR code for the v59/60 6-field
   schema.** `v13all-icrclaude.py` and `icr_prep_proto.py`'s `REQUIRED_COLS`
   both currently hard-code the 5-field v55 schema. Per user decision:
   score all 5 existing columns properly against v59/60-*migrated* files
   (not the raw v55 annotator files) -- rerunning ICR on the new schema was
   the explicit reason for updating the code at all, so migration must
   happen first (step 5), not after. Source Descriptors scoring is
   explicitly deferred -- not urgent this month since it's a new field; the
   existing 5-column approach can in principle be adapted for it later.
7. **[PENDING, after step 6] Run ICR on the 15 migrated overlap pairs** (6
   GT-I + 9 GT-II stories identified in step 5's coverage check) --
   `icr_prep_proto.py` for row alignment first, then `v13all-icrclaude.py`.
   These become the first ICR numbers computed under the v59/60 schema
   rather than v55.
8. **[PENDING, after step 7, do not conflate with step 5] Adjudicate the 9
   GT-II double-coded stories into one canonical row set each** for the
   actual benchmark GT, informed by where step 7's ICR results show
   agreement vs. disagreement. This is downstream of and distinct from
   migration itself (step 5 migrates both annotators' copies independently;
   this step decides the single official version for each of those 9
   stories) -- keep them as separate tracked steps so "migration is done"
   doesn't get conflated with "the benchmark-ready canonical GT is done."
9. **Expand the benchmark to the full GT set and get final scores.** Once the
   v59-compliant GT set (roughly stories 1-180 to 1-200, wherever the final
   count lands) is ready: run LLM annotation extraction only for the new
   stories (roughly 50 through the end of the set -- GT-2026's 1-43 already
   has annotation runs from prior work), but run evaluation across the *whole*
   set (1 through the end) to get final aggregate scores.

## Where to go for more

The canonical, sign-off'd prompt text is `system_prompt_v59.txt`/`.md` and `user_prompt_v59_csv.txt`/`.md`. If a migration script needs reasoning not covered here — a specific boundary case, why a particular Note is worded the way it is, or the fuller story behind any item on the punchlist — the full design discussion (including real worked examples from GT11, the OpenAI board story, the Vermont hair-discrimination bill story, and the Apple story) lives in a Claude.ai Project conversation and can be retrieved from there on request.
