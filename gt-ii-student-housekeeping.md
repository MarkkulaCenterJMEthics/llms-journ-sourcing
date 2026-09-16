# GT Expansion (Summer 2026) Student Housekeeping List

Running list of items for the student annotators to resolve. Not part of
`development-of-v59.md` (that's the internal migration/schema punchlist) —
this is action items to hand to the students directly, starting with the
2026-09-11 meeting.

1. **Missing double-annotation for story 60.** [UPDATE 2026-09-14: AV's files
   for 74 and 98 were delivered and have been converted/migrated — both are
   now resolved. Only 60 remains: the GT Expansion List marks it as
   dual-annotated, but only one annotator's file has actually been
   delivered.] Need the second annotator's file for story 60.
2. **1 story still has no URL anywhere in the Expansion List** (headline text
   present, but no hyperlink attached on either side): 96. [UPDATE
   2026-09-14: the 2026-09-14 Expansion List update added URLs for 18 of
   the 19 other previously no-URL stories (53, 61, 65, 66, 91, 95, 101–107,
   110, 111, 123, 124, 135, 136) — thank you. Only 96 still needs a URL or a
   PDF.]
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
   - **Net result: 63 stories successfully pulled into
     `extracted_articles_boilerplate/` with real full-text content.**
6. **Real inter-annotator disagreement on how sharply to apply the Sourced Statement definition — story 74.** AV's copy (`74-AV_San_Ramon_Pride.csv` row 8) includes "Some speakers were concerned parents of LGBTQ youth and many described themselves as long-term San Ramon residents" as a Sourced Statement (Unnamed Group of People). SZ's copy of the same story left this sentence out entirely. This is a genuinely borderline case — the only attribution present is the source group "describ[ing] themselves" that way, sitting right on the line between reporter's own characterization and attributed content, similar in kind to the reporter-first-hand-observation carve-out in the core Sourced Statement definition. Worth discussing with AV and SZ together at the next review: how strictly to draw this line going forward, since it's exactly the kind of call where the schema currently leaves room for reasonable disagreement.
7. *(open — add items here as they surface during the GT-II migration prep)*

## Fixes applied during schema migration (for your awareness)

Different from the numbered list above — those are things I need *from* you (URLs, PDFs, missing files, decisions). This section is the reverse: fixes I'm making to your first-draft annotations as I convert them to CSV and apply the v59/60 schema, shared here so nothing is a surprise when the final CSVs come back to you for signoff.

1. **Thin Source Justification fixed — story 74, "One speaker, a student at Gale Ranch Middle School" row.** AV's original annotation had Source Justification as just the single word "Speaker" — accurate but too thin to explain the source's actual standing to be quoted (the reporter is signaling their connection to the story: they were one of the speakers, and a Gale Ranch Middle School student, at the meeting). Expanded to "speaker, a student at Gale Ranch Middle School." Flagging as a pattern worth watching for in future annotation, not just this one row.
2. **Thin Source Justification fixed — story 74, Bruce Hixon row.** AV's copy had Source Justification as just the single word "Speaker" for Bruce Hixon (introduced in the article as "one of the first speakers of the night"). SZ's copy of the same story already had the fuller "one of the first speakers of the night" for this same source — applied that same fuller text to AV's copy for consistency. Second thin-SJ instance caught in this same file (see item 1 above).
3. **Source Descriptors consistency fixes applied to story 74's SZ copy.** Same two trims already applied to AV's copy (see items 1-2): Shailaja Dixit's "longtime resident of San Ramon" -> "resident of San Ramon" (dropping the duration modifier, keeping the location), and the Gale Ranch Middle School speaker's "student at Gale Ranch Middle School" -> "student" (dropping the institution name, which belongs in Source Justification instead). Not annotation errors on SZ's part — these are schema-application decisions made during migration that just hadn't been applied consistently across both annotators' copies yet.
4. **Story 108 (Youth vs. Apocalypse) — Source Descriptors trimmed, Source Justification added.** The article's own description, "YVA is a youth-led, Bay Area-based collective of young climate justice activists," had been condensed entirely into Source Descriptors ("youth-led climate justice collective") with nothing captured in Source Justification at all, across all 6 rows where YVA is the source (both AV's and SZ's copies). Per the schema, only the bare category word belongs in Source Descriptors ("collective") and the fuller descriptive sentence belongs in Source Justification. Trimmed SD to "climate justice collective" and populated SJ with the full sentence on all 6 rows (combined via ";" on the one row that already had different context). Likely why SD had grown overloaded: that descriptive sentence sits between two quotes in the reporter's own voice, not directly attached to any single "YVA said/did X" row, so it's an easy thing to read past when annotating row-by-row.

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
