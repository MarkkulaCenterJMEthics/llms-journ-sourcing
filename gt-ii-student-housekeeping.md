# GT Expansion (Summer 2026) Student Housekeeping List

Running list of items for the student annotators to resolve. Not part of
`development-of-v59.md` (that's the internal migration/schema punchlist) —
this is action items to hand to the students directly, starting with the
2026-09-11 meeting.

## What we need from you right now (as of 2026-09-16)

**1. Stories needing a PDF or story text** — doesn't matter why (fetch
failed, extraction came back empty, or the fetch technically succeeded but
only returned a paywall teaser), split by annotator (story 98 is the only
one both of you logged counts for in the Expansion List — putting it on
SZ's side for now):

- **AV:** 72, 73, 80, 82, 96, 100, 114, 128
- **SZ:** 98, 106, 123, 127, 129, 130, 150, 151, 153, 157, 158, 161, 162,
  165, 181

**2. Stories where we already have the text, but the annotation is still
due:**

- **AV:** 67, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 185,
  186, 187, 188, 189, 190
- **SZ:** none

(67 is AV's still-missing file for the whale-collision story, see item 3
below. The rest are all from AV's new "solidarity reporting initiative"
batch — texts arrived 2026-09-16, annotation XLSx files haven't yet.)

See item 5 below for how the first list was derived, and the "Fixes
applied" section for a data-integrity issue caught and fixed in this same
update.

1. **[RESOLVED 2026-09-16] Missing double-annotation for story 60 — story
   was removed, this is a non-issue.** [UPDATE 2026-09-14: AV's files for
   74 and 98 were delivered and have been converted/migrated — both are
   now resolved. Only 60 remained open at that point.] The 2026-09-16 GT
   Expansion List confirms story 60 as "invalidated - removed story" on
   both AV's and SZ's sides — no second annotator's file is needed, the
   entry is validly empty.
2. **[RESOLVED 2026-09-16] No-URL story 96 now has a URL.** [UPDATE
   2026-09-14: the 2026-09-14 Expansion List update added URLs for 18 of
   the 19 other previously no-URL stories (53, 61, 65, 66, 91, 95, 101–107,
   110, 111, 123, 124, 135, 136) — thank you. Only 96 remained.] The
   2026-09-16 Expansion List adds a hyperlink to an AP News article for
   96. However, the fetch attempt against that URL still fails (same
   blocked-domain class as the other AP stories) — so 96 has moved onto
   the "needs a PDF or story text" list at the top of this file rather
   than being fully done.
3. **Stories 67 and 70 are the same underlying article** ("California Asks
   Ships to Hit the Brakes for Whales," KQED) — AV annotated it as story 67
   (Table II, 10 rows), SZ annotated it as story 70 (Table I, 11 rows), but
   they were never paired under one shared story number the way the other
   double-coded stories were (52/55, 76/94, etc.). Needs a decision: merge
   under one number, or confirm they should stay separate and explain why.
   - **AV's story 67 annotation file was never delivered.** The Expansion
     List records AV_ss = 10 for story 67, but no XLSx for it exists in the
     downloaded finished-files folder — only SZ's story 70 copy has actually
     been received. Need AV's file for story 67.
   - **Orphaned blank placeholder row for story 67.** Table I, row 20 of the
     Expansion List has StoryNumber = 67 but both AV and SZ headline/count
     columns blank on that row — no data either side, just a leftover
     placeholder. Should be cleaned up/removed once the 67/70 merge decision
     above is settled.
   - **Story 61 has the same disconnected-pairing issue, just under one
     shared number instead of two.** AV's entry (Table II, AV_ss = 11) and
     SZ's entry (Table I, SZ_ss = 8) both exist for story 61, but they sit in
     two separate, unconnected rows rather than one shared row the way
     properly-paired dual-coded stories (74, 52, etc.) do. Still present in
     the 2026-09-14 Expansion List — not yet fixed. AV's actual annotation
     file for story 61 has since been delivered and migrated, so this is
     purely a spreadsheet bookkeeping issue at this point, not a missing-data
     one — but worth cleaning up so future story numbers don't inherit the
     same pattern.
4. **[RESOLVED 2026-09-14] Story 167's URL copy-paste error is fixed.** Its
   headline was "Life under a Delhi flyover: how one homeless family endures
   the city's extreme heat," but the hyperlink attached to it pointed to the
   Austin Chronicle's "The Vicious Cycle of Homeless Camp Sweeps" — the same
   URL as story 166. The 2026-09-14 Expansion List now gives it its own
   distinct headline and URL ("You're swamped? Join the club.,"
   scalawagmagazine.org) — thank you, confirmed fixed.
5. **URLs fetched that failed or returned unusable content** (mostly
   NYT/Reuters/WaPo/AP/Politico paywalls or bot-blocking, a few
   empty/too-short extractions from MSN/Medium/YouTube/etc.) — full list
   below. These need PDFs.
   - fetch failed: 50, 53, 54, 57, 58, 60, 62, 68, 69, 71, 73, 84, 91, 98,
     114, 123, 128, 129, 130, 150, 151, 153, 157, 158, 162, 165, 169, 181
   - extraction empty/too short (URL loaded but no usable article text):
     72, 80, 100, 106, 127, 172, 175
   - **false success caught during review, actually paywalled — needs a PDF
     too:** 82 (its URL in the sheet was a Google redirect wrapper; the
     unwrapped NYT URL also fails to fetch, same as the other NYT stories
     above), 161 (Washington Examiner — two paragraphs then a subscription
     prompt, not the full article).
   - **[UPDATE 2026-09-14]** Following the 2026-09-14 Expansion List's new
     URLs (item 2), attempted the 19 newly-added URLs: 15 succeeded (61, 65,
     66, 95, 101, 102, 103, 104, 105, 107, 110, 111, 124, 135, 136), 4 failed
     and were added to the lists above (53, 91 — fetch failed; 106 —
     extraction empty; 123 — fetch failed).
   - **[UPDATE 2026-09-16]** Two new deliveries resolved a large chunk of
     this list without needing any further URL fetch: SZ sent PDFs for 11
     of the fetch-failed stories (50, 53, 54, 57, 58, 62, 68, 69, 71, 84,
     91), and AV's new "solidarity reporting initiative" text batch
     resolved 169 (fetch-failed) and 172, 175 (extraction-empty). Story 60
     also drops off the fetch-failed list — it's an invalidated/removed
     story (see item 1). Story 96 (previously on the separate no-URL list,
     item 2) joins the fetch-failed list — it now has a URL but the fetch
     itself still fails. Updated lists after all of the above:
     - fetch failed: 73, 96, 98, 114, 123, 128, 129, 130, 150, 151, 153,
       157, 158, 162, 165, 181
     - extraction empty/too short: 72, 80, 100, 106, 127
     - false success, still needs a PDF: 82, 161
   - **Net result: 63 stories successfully pulled as of 2026-09-14, now 91
     stories with real full-text content in
     `extracted_articles_boilerplate/`** (63 prior + 11 via SZ's PDFs + 17
     via AV's new text batch, covering stories 166–176 and 185–190 — see
     the "Fixes applied" section below for a data-integrity issue caught
     and fixed while bringing that last batch in).
6. **Real inter-annotator disagreement on how sharply to apply the Sourced Statement definition — story 74.** AV's copy (`74-AV_San_Ramon_Pride.csv` row 8) includes "Some speakers were concerned parents of LGBTQ youth and many described themselves as long-term San Ramon residents" as a Sourced Statement (Unnamed Group of People). SZ's copy of the same story left this sentence out entirely. This is a genuinely borderline case — the only attribution present is the source group "describ[ing] themselves" that way, sitting right on the line between reporter's own characterization and attributed content, similar in kind to the reporter-first-hand-observation carve-out in the core Sourced Statement definition. Worth discussing with AV and SZ together at the next review: how strictly to draw this line going forward, since it's exactly the kind of call where the schema currently leaves room for reasonable disagreement.
7. *(open — add items here as they surface during the GT-II migration prep)*

## Fixes applied during schema migration (for your awareness)

Different from the numbered list above — those are things I need *from* you (URLs, PDFs, missing files, decisions). This section is the reverse: fixes I'm making to your first-draft annotations as I convert them to CSV and apply the v59/60 schema, shared here so nothing is a surprise when the final CSVs come back to you for signoff.

1. **Thin Source Justification fixed — story 74, "One speaker, a student at Gale Ranch Middle School" row.** AV's original annotation had Source Justification as just the single word "Speaker" — accurate but too thin to explain the source's actual standing to be quoted (the reporter is signaling their connection to the story: they were one of the speakers, and a Gale Ranch Middle School student, at the meeting). Expanded to "speaker, a student at Gale Ranch Middle School." Flagging as a pattern worth watching for in future annotation, not just this one row.
2. **Thin Source Justification fixed — story 74, Bruce Hixon row.** AV's copy had Source Justification as just the single word "Speaker" for Bruce Hixon (introduced in the article as "one of the first speakers of the night"). SZ's copy of the same story already had the fuller "one of the first speakers of the night" for this same source — applied that same fuller text to AV's copy for consistency. Second thin-SJ instance caught in this same file (see item 1 above).
3. **Source Descriptors consistency fixes applied to story 74's SZ copy.** Same two trims already applied to AV's copy (see items 1-2): Shailaja Dixit's "longtime resident of San Ramon" -> "resident of San Ramon" (dropping the duration modifier, keeping the location), and the Gale Ranch Middle School speaker's "student at Gale Ranch Middle School" -> "student" (dropping the institution name, which belongs in Source Justification instead). Not annotation errors on SZ's part — these are schema-application decisions made during migration that just hadn't been applied consistently across both annotators' copies yet.
4. **Story 108 (Youth vs. Apocalypse) — Source Descriptors trimmed, Source Justification added.** The article's own description, "YVA is a youth-led, Bay Area-based collective of young climate justice activists," had been condensed entirely into Source Descriptors ("youth-led climate justice collective") with nothing captured in Source Justification at all, across all 6 rows where YVA is the source (both AV's and SZ's copies). Per the schema, only the bare category word belongs in Source Descriptors ("collective") and the fuller descriptive sentence belongs in Source Justification. Trimmed SD to "climate justice collective" and populated SJ with the full sentence on all 6 rows (combined via ";" on the one row that already had different context). Likely why SD had grown overloaded: that descriptive sentence sits between two quotes in the reporter's own voice, not directly attached to any single "YVA said/did X" row, so it's an easy thing to read past when annotating row-by-row.
5. **Corpus text mismatch found and fixed for 8 story numbers (166, 167,
   168, 170, 171, 173, 174, 176) while bringing in AV's new "solidarity
   reporting initiative" text batch (2026-09-16).** Our local
   `extracted_articles_boilerplate/` already had *something* saved under
   these 8 numbers, left over from an earlier fetch/numbering pass — but
   checking each one's actual headline against the current Expansion List
   showed none of them matched. Two (166, 167) both held a duplicate copy
   of the same wrong article (this is the same underlying bug already
   flagged and marked resolved in item 4 above for story 167's URL — the
   spreadsheet URL got fixed at the time, but the already-fetched text
   file never got refreshed to match). Four more (168, 170, 173, 174)
   turned out to be real articles, just sitting under the wrong number —
   each one's true story number (190, 189, 187, 170 respectively) already
   has its own correctly-numbered text in today's delivery, so these were
   simply redundant duplicates. The remaining two (171, 176) don't match
   any headline in the current 166–190 batch at all — moved aside to
   `stale-story-text-leftovers/` (not deleted) rather than guessed at,
   since forcing them onto a number without confirmation is exactly the
   kind of mistake this check exists to catch. **To be clear: nothing
   about the GT Expansion List spreadsheet itself was wrong here** — every
   one of today's freshly delivered texts matches its assigned number's
   headline exactly. This was purely a stale local-copy issue on our side,
   now fixed: all 17 numbers in today's batch (166–176, 185–190) have
   verified, correctly-matched text in `extracted_articles_boilerplate/`.

## Flagged for annotator review (not just FYI — defend or veto)

Different from both sections above: these are judgment-call fixes made
during migration, not mechanical ones, so the annotator gets a real say —
confirm the call or override it, and I'll do a final fix afresh if
needed. Organized by annotator.

### SZ

- **Stories 178 (AFD_Win_Europe) and 179 (AFD_Majority_Fail).** Both
  articles report the same AfD state-election outcome in back-to-back
  sentences with no separating attribution — a vote-percentage figure and
  a seat-allocation figure — reading as one pull from a single "final
  results" release rather than two different documents (179 explicitly
  calls it "Final results" in the sentence introducing the seat count).
  Originally, the vote-percentage rows had Source Descriptors "vote" and
  the seat-allocation rows had "results" in 179 but "vote" in 178 —
  inconsistent between the two articles for the same underlying fact.
  Standardized all 3 affected rows (178 row 28, 178 row 29, 179 row 7) to
  "results" in both articles, on the reasoning that "results" is the
  broader, more accurate genre word covering both figures if they in fact
  come from one combined release. **SZ: if you know these actually came
  from two distinct sources/documents (e.g., a live vote-count feed
  separate from the official seat-allocation announcement), say so and
  we'll split them back apart instead.**
- **Stories 177 (AFD_Prediction_German_State_Election) and 179
  (AFD_Majority_Fail) — "Germany's domestic intelligence agency"
  reclassified from Named Organization to Unnamed Group of People.**
  Neither article ever states this agency's actual formal name (no "BfV"
  or similar anywhere in the text) — it's referred to only generically,
  the same way "police" or "the department" would be. Per the existing
  rule that a Named Organization row needs an actual name stated at that
  point in the text (not inferred from elsewhere), this doesn't qualify —
  and 177 already treats an identical case ("State authorities," row 20)
  as Unnamed Group of People, so this brings the file in line with its
  own existing precedent. Fixed: Type of Source → Unnamed Group of
  People, Name of Source blanked, Title of Source → "Germany's domestic
  intelligence agency" (kept whole — "Germany's" is a jurisdiction
  qualifier, not decorative). **SZ: if you know the actual name of this
  agency (likely the BfV, Germany's federal domestic intelligence
  service, or its Saxony-Anhalt state-level equivalent) and it just
  didn't make it into your draft, let me know and we'll restore this to
  Named Organization with the real name instead.**
- **Story 177 (AFD_Prediction_German_State_Election) row 19 — trimmed an
  overlapping Sourced Statement.** Row 19's Sourced Statement was
  "Independent election observers are active at polling booths amid
  speculation from the AfD that voter fraud has taken place, in
  particular in postal voting, with specific focus on a retirement home.
  State authorities have denied the claims." — but that second sentence
  is a separate attribution to a different source, and it's already
  correctly captured on its own in row 20 (Unnamed Group of People,
  Title "State authorities"). Row 19 had it duplicated in its own SS
  text on top of that. Trimmed row 19's SS to just the AfD-attributed
  portion, dropping the "State authorities have denied the claims."
  sentence since row 20 already owns it exclusively. Not a missing row
  (that part was already done correctly) — just an overlap to clean up.
- **Story 184 (Anthropic_Bio_Weapons) row 11 — recovered a missing second
  sentence from the same paragraph.** Row 11's Sourced Statement was only
  "In these cases, each operation was run by an actor working within or
  on behalf of an Iranian state propaganda institution, the company
  said." — but in the source article, that same paragraph has a second,
  unattributed sentence right after it ("Claude was used to build
  content, make posts seem like they were from independent news sources
  and to proliferate content across social media platforms like X,
  Instagram and TikTok.") that draws on the same implied attribution and
  was never captured anywhere. Compare rows 12 and 13 in this same file,
  which each correctly merged both sentences of their own paragraphs —
  row 11 is the one that was under-captured. Added the missing sentence
  to row 11's Sourced Statement.
- **Story 184 (Anthropic_Bio_Weapons), the article's last paragraph —
  left open for now, not yet a row.** The final paragraph ("In China,
  three accounts aligned with the PRC municipal security service used
  Claude for surveillance and transnational repression, including a
  municipal bureau that profiles overseas activists and organizations.")
  has no attribution tag of its own anywhere in it, and isn't captured
  as a Sourced Statement row at all. Not merged into the preceding
  paragraph's attribution either, per the rule that different paragraphs
  are separate instances. Flagging for awareness, not fixed yet — no
  action needed from you unless you want to weigh in on whether this
  should be added as its own row.

## status-2026-09-14

Full cross-reference of everything delivered so far (`~/Documents/GT-II-finishedfiles-XLSx/`), run before the next batch of XLSx files comes down.

**a) All XLSx → CSV conversion: ✅ Complete.**
80 XLSx files in the folder, minus 2 deliberately excluded (133, 134 — confirmed different experiment) = 78. All 78 have a corresponding CSV in `benchmarking/GT data/GT-II/`, 1:1, zero missing, zero count mismatches.

**b) No-URL stories itemized in housekeeping list: ✅ Confirmed.**
Of the 71 distinct story numbers with an XLSx delivered, 19 fall in the no-URL bucket (53, 61, 65, 66, 91, 95, 101–107, 110, 111, 123, 124, 135, 136) — all already itemized in item 2 above. (Story 96 is also on that no-URL list but has no XLSx delivered yet, so it's a non-issue for this batch.)

**c) Dual-coded files converted: ✅ Confirmed.**
9 story numbers have 2 XLSx files each (52, 55, 76, 94, 108, 109, 112, 122, 125) — all 9 have exactly 2 corresponding CSVs. All 9 also have real article text and were fully carried through Phase 1/2 of the migration.

**d) Maximum possible migration complete: ✅ Confirmed.**
Full accounting of all 71 distinct story numbers:

| Bucket | Count | Status |
|---|---|---|
| Successfully pulled (real article text) | 30 numbers → 39 files (incl. 9 dual-coded pairs) | Phase 0 **and** Phase 1/2 complete |
| Fetch failed | 15 | Phase 0 done, blocked on text (item 5 above) |
| Extraction empty/too short | 4 | Phase 0 done, blocked on text (item 5 above) |
| False success (paywall/redirect) | 1 (82) | Phase 0 done, blocked on text (item 5 above) |
| No URL at all | 19 | Phase 0 done, blocked on text (item 2 above) |
| Deliberately excluded | 2 (133, 134) | Not migrated, confirmed different experiment |
| **Total** | **71** | fully accounted for |

Every structural conversion that could be done is done; every deeper annotation-quality pass (Phase 1/2) that could be run against real article text has been run, against exactly the 39 files that have it — no story with real text was missed, and nothing without real text was silently skipped without being logged.
