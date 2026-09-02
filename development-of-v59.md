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

## Migration punchlist (script development pending)

1. **Reclassify GT Anonymous Source rows → Unnamed Person or Unnamed Group of People** wherever no qualifying disclosure (explicit or blanket-earlier-in-story) exists for that specific source. This is the majority of the migration effort — GT was annotated under the old, looser reading. Applies to both the 645-row legacy GT and the newer batch.
2. **Populate the new Source Descriptors column** by re-deriving values from existing Title of Source / Name of Source / Source Justification text, applying the credentialing test and the atomic-word rule above (including the resolved "employees and advisers" split pattern). Expect the largest single source of new values to come from Unnamed Group of People rows where a non-credentialing word was previously dumped into Name of Source.
3. **GT housekeeping fixes** (lower priority, human review needed, do not auto-migrate):
   - Multi-source-per-sentence rows crediting two organizations jointly (e.g., "Bloomberg and The Information") — open question whether Note 12's one-row-per-source split rule applies to organizations or only individuals.
   - Rows attributing a description of a company presentation/demo to the company as a Named Organization source — likely miscategorized; probably reporter first-hand observation (not attributable to a source at all) rather than a sourced statement.
4. **Edge case, revisit later: UGOP row with no atomic descriptor word in its Source Justification.** `GT-2026/40-SeattleProtest.csv` row 1 (Type: Unnamed Group of People) has Source Justification "The group that organized a takeover protest of a building on the University of Washington campus earlier this week" — this clearly describes the source, but there's no bare atomic word in it the way "witnesses" or "protesters" would be (no single token like "organizers"). Left Source Descriptors null rather than coining a word not actually in the text. Needs a human call: leave null, or is inferring "organizers" from "the group that organized..." within the spirit of the atomic-word rule?
5. **Draft, and eventually add to system_prompt_v59, a general principle for why a single source (individual or group) may carry multiple simultaneous descriptors across fields** — currently only demonstrated through worked examples ("AI researcher and activist," "employees and advisers"), not stated as a principle. Needs careful wording: do NOT frame it as "reporters routinely give sources rich, multi-faceted portrayal" — that asserts good practice as the norm when it isn't. Journalism has a real, live contest here: some reporters reduce sources to flattening, reductive labels (e.g., "far-left activist"), others give fuller, multi-faceted treatment. The schema's job is to capture *all* usages that map into each column so that downstream analysis can distinguish reductive/flattening source treatment from richer, more democratic portrayal — that's the actual reason multi-field capture matters, not an assumption that reporters already do it well. Revisit after the students' test results come back.

## Where to go for more

The canonical, sign-off'd prompt text is `system_prompt_v59.txt`/`.md` and `user_prompt_v59_csv.txt`/`.md`. If a migration script needs reasoning not covered here — a specific boundary case, why a particular Note is worded the way it is, or the fuller story behind any item on the punchlist — the full design discussion (including real worked examples from GT11, the OpenAI board story, the Vermont hair-discrimination bill story, and the Apple story) lives in a Claude.ai Project conversation and can be retrieved from there on request.
