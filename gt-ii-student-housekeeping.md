# GT Expansion (Summer 2026) Student Housekeeping List

Running list of items for the student annotators to resolve. Not part of
`development-of-v59.md` (that's the internal migration/schema punchlist) —
this is action items to hand to the students directly, starting with the
2026-09-11 meeting.

## Excluded story numbers (standing list — not migrated, don't re-flag)

- **133, 134** — confirmed a different experiment, not part of this GT
  corpus at all.
- **161, 162** — op-ed/opinion pieces ("Trump just slashed gas prices...",
  "Todd Monken has 2 versatile weapons..."). Annotation XLSx exist but no
  article text was ever sourced for them, and they sit on their own
  "Op-Ed pieces" sheet in the GT Expansion List, separate from the main
  sheet everything else lives on — same shape as 133/134's exclusion.
  Confirmed excluded 2026-09-17.
- **127** — the real story 127 is an influencer video ("Fetterman
  Secret Israel Handler EXPOSED", YouTube), listed on the GT Expansion
  List's "GT for InfluencerVideos" sheet alongside 133/134 — excluded
  for the same reason. Separately, the file delivered as
  `127 - Everest Climbers.xlsx` is **not** that video: it's a
  mislabeled copy of SZ's story-125 annotation (all 6 Sourced
  Statements identical to `125-SZ_Mount_Everest.csv`, and verbatim in
  125's article text). Its Phase-0 CSV (`GT-II/127-Everest_Climbers.csv`)
  was removed 2026-10-07 — nothing lost, since 125 is already fully
  migrated with both SZ's and AV's annotations. (Original XLSx
  untouched; CSV recoverable from git, commit 11aa5a8.) Confirmed
  excluded 2026-10-07.
- **59, 136** — not news stories. 59 ("The Robot Rat Race", Institute
  for Global Affairs) is a foundation's explainer, and 136 ("Scientists
  create supercharged vitamin K...", ScienceDaily) is a republished
  research summary. Their GT-II CSVs (`59-Robot_Rat_Race.csv`,
  `136-Vitamin_K.csv`) and article texts were removed 2026-10-21
  (recoverable from git, commit 5261112). Original XLSx untouched.

## What we need from you right now (as of 2026-09-17)

**Big update since the last version of this list (2026-09-16):** the
Sep17 delivery (new PDFs from both of you, plus AV's Sep11 batch that had
been missed) resolved almost everything that was on this list — including
story 67, AV's long-missing whale-collision file. Full detail on
everything that arrived and got matched up is in `development-of-v59.md`
if you want it; here's just what's still actually needed.

**1. Stories still needing a PDF or story text — 1 now, SZ's:**

- ~~**165**~~ — **RESOLVED 2026-09-17.** SZ supplied the correct PDF —
  Independent.co.uk's "Canada will move faster from US reliance as
  tariffs take effect and Trump response looms" (the earlier PDF under
  this number was a duplicate of 181's AP dairy-tariffs article). Verified
  real extractable text (4 pages, ~5,900 chars across the first 3), and
  confirmed the headline/lede matches the annotation's own Sourced
  Statements exactly. Converted and cleaned — two spots needed
  reconstruction rather than mechanical cleanup: a "Skip to content" nav
  artifact interleaved mid-sentence in two places, one of which garbled a
  phrase character-by-character ("tShkripe taot sc oanntedn atttacks").
  Reconstructed as "threats and attacks" by cross-referencing 181's
  near-identical AP wire sentence (same underlying reporting team —
  Rob Gillies/Paul Wiseman — republished by both outlets), not guessed
  from scratch. Phase 0 done (17 rows, straightforward — already
  v59/60-native schema, no legacy-v55 conversion needed); Phase 1 trivially
  clean (zero Anonymous Source rows in this file).
- ~~**71**~~ — **RESOLVED 2026-09-17.** AV supplied a fresh, non-paywalled
  NYT PDF for "Global Deforestation Slows, Analysis Finds. But Fires
  Remain a Major Threat." (the original delivery was a paywalled browser
  print with no real article body). Verified real extractable text (3
  pages, ~7,200 characters), converted and cleaned. Ready for Phase 0.
- **100** — the delivered PDF has zero extractable text on any of its 6
  pages (checked directly) — it appears to be a full-page-image render
  rather than a real text export, so no amount of cleanup can recover it;
  it would need OCR or a different source entirely. The story's URL (from
  the Expansion List) is
  https://www.msn.com/en-us/money/companies/openai-chief-altman-has-over-2-billion-stake-in-companies-that-dealt-with-openai-court-filing-shows/ar-AA237jES —
  an MSN link, which is likely why the PDF rendered as images (MSN pages
  often render heavy client-side content that "print to PDF" captures as
  a picture rather than selectable text). Needed: either a cleaner PDF
  save from that same URL, or a link to wherever the underlying wire
  story (this looks like an AP/Reuters-style business story MSN is
  syndicating) is hosted directly.
- **212 (SZ)** — ✅ **RESOLVED 2026-10-06**: fresh capture received
  (`SZ-212 - Iran awaits US move after WSJ report says Trump rejects
  peace plan _ Reuters.pdf`); all 15 annotated rows now found in the
  text. Original note kept below for the record. — the supplied PDF converts cleanly, but its text is
  missing content the annotation (rows 12-15) actually cites: Araqchi
  briefing Prince Faisal bin Farhan, the Saudi-led coalition
  intercepting Houthi drones/missiles, a Turkey/Pakistan defense
  meeting in Riyadh, and a UN children's-agency statement on Yemen.
  This looks like a Reuters "rolling update" story (the page itself is
  marked "Updated September 26, 2026") — the supplied PDF is likely an
  earlier snapshot captured before a regional-roundup section was
  appended later that day. **Needed: a fresher PDF capture of the same
  URL**, taken after that update:
  https://www.reuters.com/world/middle-east/iran-awaits-us-move-after-wsj-report-says-trump-rejects-peace-plan-2026-09-26/
- **53 (SZ)** — ✅ **RESOLVED 2026-10-08**: fresh PDF received (`53 - Iranian sea trade blocked as six ships forced to turn back.pdf`) and swapped in; it's the version SZ annotated (headline matches the Expansion List). Row 1 turned out to be this version's subtitle — see the removal entry below. Original note kept for the record: sent to SZ 2026-10-08. Our copy of the Washington Post
  article (converted 2026-10-07 from the PDF in SZ's folder) seems to be
  a different version from the one annotated: its headline reads "U.S.
  blockade has turned back 6 merchant ships leaving Strait of Hormuz,"
  not the Expansion List's, and row 1 ("More than a dozen American
  warships positioned in the Gulf of Oman and Arabian Sea are acting as
  a 'net,' officials said.") isn't in the text. **Needed: a fresh PDF,
  or the version SZ worked from**:
  https://www.washingtonpost.com/national-security/2026/04/14/us-blockade-hormuz/
- **193 (SZ)** — row 23's sentence ("Ireland's national broadcaster RTÉ
  said it would not drop Sheeran's music from its radio stations
  following the backlash...") does not appear anywhere in our source
  text for this story, and the source here is a plain `.txt` file, not
  a PDF extraction, so this isn't an extraction artifact. Likely the
  same pattern as 212: the live Newsweek article was probably updated
  with this reaction after whatever snapshot we have was taken.
  **Needed: a fresh full copy of the current live article text**
  (https://www.newsweek.com/ireland-disowns-ed-sheeran-over-macklemore-12454999),
  so we can confirm row 23 and re-check nothing else was added or
  changed since our copy was captured.

**2. Stories where we already have the text, but the annotation is still
due: none right now** — the last item here (67) arrived in AV's Sep11
batch and is ready to migrate.

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
     the "corpus text mismatch" item in AV's list under "Flagged for
     annotator review" below for a data-integrity issue caught and fixed
     while bringing that last batch in).
6. **Real inter-annotator disagreement on how sharply to apply the Sourced Statement definition — story 74.** AV's copy (`74-AV_San_Ramon_Pride.csv` row 8) includes "Some speakers were concerned parents of LGBTQ youth and many described themselves as long-term San Ramon residents" as a Sourced Statement (Unnamed Group of People). SZ's copy of the same story left this sentence out entirely. This is a genuinely borderline case — the only attribution present is the source group "describ[ing] themselves" that way, sitting right on the line between reporter's own characterization and attributed content, similar in kind to the reporter-first-hand-observation carve-out in the core Sourced Statement definition. Worth discussing with AV and SZ together at the next review: how strictly to draw this line going forward, since it's exactly the kind of call where the schema currently leaves room for reasonable disagreement.
7. *(open — add items here as they surface during the GT-II migration prep)*

## Flagged for annotator review

Different from the numbered "what we need from you" list above (that's
things I need *from* you — URLs, PDFs, missing files, decisions). This
section is fixes made to your annotations during schema migration,
organized by annotator, one combined list each — some are judgment calls
where you get a real say (confirm or override, and I'll do a final fix
afresh if needed), others are more mechanical (shared so nothing is a
surprise when the final CSVs come back to you for signoff). Not
distinguished by subsection since it's easier to review as one list per
person; judgment calls are generally phrased as "flagged"/"resolved" and
mechanical ones as "fixed," if you want to tell them apart at a glance.

### SZ

[Also see `development-of-v59.md`, Prompt Updates checklist items 20, 21,
and 22 — candidate prompt additions (drafted, not yet merged) worth
discussing at the next meeting before they're formally added to
system_prompt_v60: item 20 covers the "editorializing qualifier stacked
on a credentialing title" pattern (stories 178/180's Salvini/Wilders/
Ben-Gvir/Merz rows below); item 21 covers the missing counterpart to the
org-affiliation-stripping rule for Source Descriptors -- when an attached
object names a generic cause/category rather than an organization, it
stays instead of getting stripped (story 180's Huckabee row below); item
22 covers Source Descriptors not capturing a source's own self-description
inside a quote, only the reporter's characterization (story 180's
Miliband rows -- no fix applied there, just a documented no-change call).]

- **Stories 178 (AFD_Win_Europe) and 179 (AFD_Majority_Fail) — Document
  Source Descriptors for the vote-count/seat-count rows, corrected to
  match each article's own manifest text.** Both articles report the
  AfD state-election vote-percentage and seat-count figures. An earlier
  pass standardized these to "results" in both articles, reasoning that
  the two figures likely come from one combined release — but on a closer
  manifest-text check, that reasoning doesn't hold the same way in both
  articles, because each article establishes its *own* term for this data
  at its first citation, and the two are different words:
  - **179** explicitly says "**Final results** showed AfD falling three
    seats short..." (the sentence right before the vote/seat figures, same
    paragraph) — "results" is genuinely this article's own term. Rows 6
    and 7 (both the vote-percentage and CDU's vote/seat figures) are
    correctly "results."
  - **178** never uses "results" as its own citation term anywhere —
    its first citation of this same data is "...the AfD surged into first
    place... on Sunday, **exit polls showed**..." (row 4, already SD
    "polls"). The few uses of the word "result" elsewhere in 178 are all
    inside quotes from reacting foreign politicians calling the outcome
    "the result" colloquially, not the reporter's own citation of a data
    source. Reverted rows 28 and 29 back to "polls," matching this
    article's own established term (row 4) instead of borrowing 179's
    word for a structurally similar but textually distinct citation.
  **Net effect:** 178 rows 28-29 → "polls" (reverted from "results"); 179
  row 6 stays "results"; 179 row 7 (a 4th affected row found during a
  later full re-sweep, same paragraph/pattern as row 6) → "results."
  **SZ: if either article's own data-citation language differs from what
  I've described here, let me know and we'll revisit.**
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
- **Story 177 (AFD_Prediction_German_State_Election) rows 7, 9, 10, 11,
  12 (Ulrich Siegmund) — "AfD's lead candidate" moved from Title of
  Source to Source Descriptors/Source Justification.** "Candidate" is a
  non-credentialing word per the schema (explicitly listed as an SD
  example, alongside "activist," "organizer," etc.), so it shouldn't sit
  in Title of Source at all — and this file's own comparison files,
  178-AFD_Win_Europe and 179-AFD_Majority_Fail, already treat this exact
  same person's "candidate" role correctly (bare "candidate" in Source
  Descriptors, fuller phrase in Source Justification). Brought these 5
  rows in line: Title of Source cleared, Source Descriptors → "candidate",
  and "AfD's lead candidate" appended to the existing Source Justification
  text via ";" (which already held "Ulrich Siegmund, 35, a former
  fragrance salesman with a large following on TikTok who has been
  celebrated like a pop star." on all 5 rows). **Update:** the
  systematic Source Descriptors recovery pass has now run, and as
  flagged above, "fragrance salesman" from that same Source
  Justification text is a genuine second Source Descriptors facet for
  this source. Added: Source Descriptors → "candidate, former fragrance
  salesman" (comma-separated, both facets manifest in the article's own
  text) on all 5 rows.
- **Story 179 (AFD_Majority_Fail) — removed a duplicate row, and every
  row number in this file shifted down by 1 as a result.** The old row 2
  ("AfD's candidate for governor, Ulrich Siegmund, said 'we have written
  history'...") was a partial, redundant capture of the same single
  paragraph that the old row 3 already captured in full (all three
  sentences of that paragraph, correctly merged per the
  same-paragraph/implied-attribution rule). Deleted the old row 2 and
  renumbered the "No." column sequentially. **Important for looking up
  any other note in this file by row number: every row after the old
  row 2 is now one row earlier than it used to be** (old row 4 →
  current row 3, old row 12 → current row 11, old row 26 → current row
  25, old row 32 → current row 31, etc. — subtract 1 from any row number
  referenced elsewhere for this file before this fix). The vote/results
  note above has already been corrected to the current row number (6,
  not the original 7).
- **Story 182 (Missouri_Map) row 10 (Catherine Hanaway) — split "Republican"
  out of Title of Source into Source Descriptors, matching this file's
  own row 3 (Denny Hoskins).** Title of Source was "Missouri Republican
  Attorney General" — but row 3, for a different source in this same
  file, already correctly splits this exact pattern into Title "Missouri's
  secretary of state" + Source Descriptors "Republican" (party affiliation
  is non-credentialing, formal office is credentialing). Hanaway's row
  didn't follow that split. Fixed: Title of Source → "Missouri Attorney
  General", Source Descriptors → "Republican".
- **Story 179 (AFD_Majority_Fail) row 20 (Friedrich Merz) — completed
  Title of Source with the jurisdiction already stated in the text.**
  Title of Source was bare "chancellor," but the article's own text says
  "Germany's unpopular chancellor Friedrich Merz" — the country name is
  manifestly present in the sentence, just not captured. Every other
  government-official Title in this batch (178's prime ministers/foreign
  ministers, 180's Foreign Secretary/Israeli Foreign Minister/Israeli
  President, etc.) keeps its stated country/jurisdiction prefix when the
  article gives one; this row was the one exception. Fixed: Title of
  Source → "Germany's chancellor" (dropping only "unpopular," which is
  the reporter's own editorial characterization, not a jurisdiction
  qualifier — Source Justification already correctly holds "unpopular
  chancellor Friedrich Merz" separately).
- **Story 180 (UK_Israel_Ban) row 23 (Itamar Ben-Gvir) — split a
  two-role Title of Source into a comma-separated list, dropping one
  editorializing qualifier.** Title of Source was "Israel's hard-line
  public security minister and a settler leader" — both "public security
  minister" (a formal government office) and "settler leader" (leadership
  is itself a credentialing category per the schema) qualify as Title
  material, so no need to demote either one to Source Descriptors. Fixed:
  Title of Source → "Israel's public security minister, settler leader"
  — comma-separated since both are credentialing, and "hard-line" dropped
  as the reporter's own editorial characterization rather than part of
  either formal role.
- **Story 163 (Jill_Stein_Court) rows with Name of Source "Jill Stein"
  (No. 2 and No. 14) — recovered a missed Title of Source.** The
  article's opening sentence introduces her as "the Green Party's former
  presidential candidate **Dr.** Jill Stein" — "Dr." is a credentialing
  title that was never captured anywhere (only Source Descriptors
  "candidate" and Source Justification existed). Added Title of Source →
  "Dr." to both rows.
- **Story 178 (AFD_Win_Europe) rows 18, 19 (Santiago Abascal, André
  Ventura) — recovered a missed Title of Source, and rows 20, 21 (Matteo
  Salvini) — trimmed an editorializing qualifier out of Title of
  Source.** All four rows are connected: the sentence just before
  Abascal/Ventura reads "Far-right leaders in Spain and Portugal also
  hailed the AfD's win," directly naming the two countries these two
  people represent (Abascal/Spain, Ventura/Portugal) — a direct
  antecedent, not an inference. Added Title of Source → "leader" to both
  (bare, not "far-right leader" — see below for why). Separately,
  Salvini's existing Title of Source, "Italy's far-right deputy prime
  minister," has the same "far-right" qualifier attached to a genuinely
  credentialing title, and applying the same test already used elsewhere
  in this batch (does removing the qualifier still leave an accurate
  description of the same position? yes) means it should have been
  trimmed too — "far-right" is editorializing, not part of the formal
  job title. Fixed: Title of Source → "Italy's deputy prime minister"
  on both Salvini rows. In every case, "far-right" is preserved, not
  lost — it's either already present in Source Justification (Abascal/
  Ventura, via the existing "Far-right leaders in Spain and Portugal..."
  text) or added there via ";" (Salvini, both rows).
- **Story 180 (UK_Israel_Ban) row 26 (Mike Huckabee) — recovered both
  Source Justification and Source Descriptors from the same
  uncaptured phrase.** The article introduces him as "the U.S.
  ambassador to Israel, Mike Huckabee, **a longtime supporter of the
  settlements**" — that phrase was never captured anywhere (both fields
  were blank). It independently satisfies Source Justification's own
  definition (context on his stake/vantage point in the issue, which is
  exactly why the reporter includes his criticism), so it goes there in
  full, untrimmed: "a longtime supporter of the settlements." Separately,
  Source Descriptors gets the same phrase with only the duration qualifier
  "longtime" dropped: "supporter of the settlements" — checked against 6
  other "supporter"/"advocate" precedents across both corpora first
  (Paul Boden's "homeless advocate," "Tenant advocates," "Union
  advocates," "Border Advocates," etc.) to confirm the object stays
  attached when it names a generic cause/category rather than an
  organization's proper name (which would instead get stripped per the
  org-affiliation rule, as it did for a different source's bare
  "supporter" elsewhere in the corpus).

- **Story 181 (US_Canada_Dairy) row 18 — reclassified Anonymous Source to
  Unnamed Person; the earlier disclosure doesn't clearly cover it.** The
  file has two separate "Canadian official" attributions: rows 8-9 (EU
  relations) are introduced with "a Canadian official familiar with the
  discussions said," then explicitly disclosed two sentences later ("The
  official spoke on condition of anonymity because they were not
  authorized to discuss the talks publicly."). Row 18 (a different topic —
  Ottawa's response strategy to Trump) comes ~25 lines and two
  subheadings/Carney-quote blocks later, and is introduced the same
  way — "a Canadian official said" — using the indefinite article again
  rather than "the official," which is the phrasing you'd expect if the
  reporter meant to keep referring to the already-disclosed source from
  rows 8-9. Nothing in the text confirms it's the same official (no
  "another official" or similar to mark a *different* one either) — it's
  genuinely ambiguous, and treating rows 8-9's disclosure as covering row
  18 requires inferring same-source continuity the text doesn't actually
  state. Per Phase 1's rule (an explicit or blanket disclosure must
  actually be present, not inferred), reclassified to Unnamed Person.
  Title of Source kept as "Canadian official" (matches rows 8-9's own
  precedent in this file); Source Justification blanked, since the copied
  text ("a Canadian official familiar with the discussions") was rows
  8-9's EU-specific phrase and doesn't substantively describe row 18's
  Ottawa-strategy quote.
- **Story 181 row 20 — reclassified Unnamed Group of People to Named
  Organization, matching story 165's precedent for the identical
  sentence.** Same Sourced Statement as row 19 ("Since Canada-U.S. trade
  talks collapsed... Trump and his administration have imposed
  additional tariffs..."), split per Note 12 into a Named Person row
  (Trump, row 19) and an org-half row (row 20). Your draft had the
  org-half typed Unnamed Group of People with "Trump Administration"
  sitting in Name of Source — but "Trump Administration" isn't a common
  noun, it's the specific, de facto name of a specific organization (the
  executive branch under a given president), the same way `165`'s
  identical sentence already correctly typed it Named Organization.
  Fixed: Type of Source → Named Organization, Name of Source → "Trump
  administration" (lowercase, matching `165`'s casing).
- **Story 181 rows 1, 2, 5 — confirmed Named Organization is correct for
  bare country/region names used as geopolitical shorthand.** "The
  United States is banning..." (White House), "Canada had responded to
  tariff moves..." (Canada), "the U.S. imposed 50% tariffs..." (United
  States) — all three were already correctly typed Named Organization.
  Checked the actual prompt directly and confirmed this exact case
  (a bare sovereign-region name standing in for its government) isn't
  explicitly addressed anywhere yet — logged as a real prompt gap
  (`development-of-v59.md` item 26), not just a local judgment call, and
  drafted as a generalization of the existing "San Mateo County
  officials" pattern (Note 8) to every level of regional granularity.
  No CSV change needed here, these three rows were already right.
- **Story 181 row 5 — fixed a transcription error in Sourced Statements.**
  Your copy had "n Aug. 22, the U.S. imposed 50% tariffs..." — missing
  the "O" from "On." Checked against the article text directly (which
  correctly reads "On Aug. 22, the U.S. imposed 50% tariffs..."),
  confirmed this was a transcription slip on your side, not an
  extraction artifact, and fixed the Sourced Statement text to match the
  article verbatim.
- **Source Descriptors consistency fixes applied to your copy of story
  74.** Same two trims already applied to AV's copy: Shailaja Dixit's
  "longtime resident of San Ramon" -> "resident of San Ramon" (dropping
  the duration modifier, keeping the location), and the Gale Ranch Middle
  School speaker's "student at Gale Ranch Middle School" -> "student"
  (dropping the institution name, which belongs in Source Justification
  instead). Not an annotation error on your part — these are
  schema-application decisions made during migration that just hadn't
  been applied consistently across both annotators' copies yet.
- **Story 108 (Youth vs. Apocalypse) — Source Descriptors trimmed, Source
  Justification added.** Applies to both your copy and AV's — see AV's
  list below for the full write-up (same fix, same reasoning, both
  copies).
- **Story 181 (US_Canada_Dairy) rows 8, 9 — added the missing anonymity
  disclosure sentence to Source Justification, found during Phase 4a.**
  Both rows were already correctly typed Anonymous Source with SJ "a
  Canadian official familiar with the discussions," but the article's
  own disclosure sentence — "The official spoke on condition of
  anonymity because they were not authorized to discuss the talks
  publicly" — sits several lines further down (past a stray extraction
  artifact) and had never been captured anywhere. This is the actual
  textual basis for the Anonymous Source Type assignment on these rows,
  not optional color. Added via the existing ";"-concatenation
  convention: SJ now reads "a Canadian official familiar with the
  discussions; spoke on condition of anonymity because they were not
  authorized to discuss the talks publicly" on both rows.
- **Story 181 — new row added, found during the same Phase 4a pass.**
  "Canadians have sharply cut travel to the United States and
  boycotted U.S. goods, moves Carney praised as signs of national
  resolve." had no row — a genuine attribution (Carney's own
  characterization of the boycotts), sitting a couple of sentences
  after the last Trump-administration row. Added as row 21 (Named
  Person, Mark Carney, Title "Canadian Prime Minister," matching all
  his other rows in this file); everything renumbered sequentially
  (final: 1-27). A neighboring sentence — "Trump's trade war and
  repeated talk of making Canada the 51st state have fueled anger
  across the country" — was considered and rejected: "repeated talk"
  there is the reporter's own causal framing referencing an
  already-captured pattern (row 6), not a new attributed statement.
- **New combined batch, Phase 0 intake, 2026-10-02: stories 191-211 (21
  files, 335 rows), converted from the Sep24 + Sep29 xlsx deliveries.**
  These were already authored in the full v59/60 7-column schema
  (Source Descriptors column present natively), so Phase 0 was pure
  mechanical conversion to CSV in `benchmarking/GT data/GT-II/` — no
  column-adding needed. Two filename-only typos corrected (no
  annotation content touched): `205-Morgage_Rates` -> `205-Mortgage_
  Rates.csv`; `211-Taisa_Fortune` -> `211-Tasia_Fortune.csv` (matches
  the subject's real name, "Tasia Fortune," per the source PDF).
  **Story 212 deliberately excluded from this batch** — its supplied
  PDF is missing content the annotation cites (see `development-of-
  v59.md`'s item 42-adjacent story-212 log); held until a fresher PDF
  capture is available. Flagged 4 non-canonical Type of Source values
  for the next pass (canonicalization): `191` row 3 "Documentary
  Source / Unnamed Source"; `203` row 9 "Named Publication", row 10
  "Anonymous Group"; `205` rows 1/8/16 "Unnamed Group of people"
  (casing).
- **Type of Source canonicalization pass on those 4 rows, done
  2026-10-02.** `203` row 9: "Named Publication" -> Named Organization
  (Name "Wall Street Journal" already correct). `203` row 10:
  "Anonymous Group" -> Unnamed Group of People (no anonymity-disclosure
  phrase present for "administration officials" anywhere in the
  article, so this follows the standing Phase 1 rule directly); also
  fixed a schema violation along the way — "administration officials"
  was sitting in Name of Source, moved to Title of Source (UGOP rows
  can never have Name of Source populated). `205` rows 1/8/16: pure
  casing fix, "Unnamed Group of people" -> "Unnamed Group of People" —
  nothing else needed touching. `191` row 3 needed real article
  context, not just a label swap: it had a Type-of-Source value
  ("Named Person") sitting in the Name field — an annotation-entry
  glitch. The sentence ("...Musk, Zuckerberg and Huang all reportedly
  pushed back...") is a tight implied continuation of an explicit "The
  Wall Street Journal reported" two sentences earlier in the same
  paragraph cluster (row 2) — reclassified to Named Organization / The
  Wall Street Journal, SD and SJ both blanked (neither held anything
  substantive once the Type/Name fix made them moot). **Flagged but not
  fixed, out of scope for this pass**: `191` rows 8/9's Source
  Justification says "who runs SpaceXAI" while the row's own Sourced
  Statement text correctly says "who runs xAI" — a typo merging two
  company names, worth a quick fix whenever this file comes up again.
- **Phase 1 (Anonymous Source reclassification), run 2026-10-02 —
  clean, no changes.** Zero rows across all 21 files (335 total rows)
  are currently typed Anonymous Source, so there was nothing to check
  for a disclosure statement. Type breakdown: Document 39, Named
  Organization 86, Named Person 166, Unnamed Group of People 36,
  Unnamed Person 8.
- **Phase 2.1 (Unnamed Group of People), run 2026-10-02 — 3 files
  edited.** `192` row 2: SD "tech bosses" moved to Title as "U.S. tech
  bosses" (credentialing role, same bucket as row 4's "industry
  leaders," already correctly in Title). `196` rows 12/13: casing fix,
  "U.S. Officials"/"Danish Officials" -> lowercase "officials," to
  match the article's own capture-exactly-as-written text. `207` row
  8: reclassified Unnamed Group of People -> Named Organization /
  Monster — this row cites the same Monster recruitment-platform study
  as the file's own rows 2, 3, 9, and 10 (all already correctly Named
  Organization/Monster); row 8 alone had been left inconsistent. SD
  ("recruitment platform") kept, SJ left untouched pending the
  separate SJ review pass. Two candidates from this pass were raised
  and then retracted after corpus-wide checks: a proposed Title
  backfill for "people familiar with..." phrasing (`206` row 16, `208`
  row 26) turned out to contradict a clean 20-instance precedent that
  this phrasing is SJ-only, never Title; and a proposed Title expansion
  for `208` row 2 ("officials" -> "current and former officials") was
  reconsidered and dropped — for a *group* Title, "current and former
  officials" is semantically just "officials from both groups," so the
  bare role noun already carries the full credentialing content (no
  precedent existed either way; this was a fresh call, not a lookup).
- **Phase 2.4 (Document), run 2026-10-02 — 1 file edited (3 casing
  fixes).** `204` rows 7/9: SD "Proclamation" -> lowercase
  "proclamation," matching the article's own mid-sentence "The
  proclamation..." text. Row 8: SD "studies" -> capitalized "Studies,"
  matching the article's sentence-initial "Studies have found...".
  Everything else (schema violations, missed-document-name recovery,
  genre-word recovery) came back clean across all 39 Document rows.
  One adjacent finding (`210` row 14, AdImpact/Wall Street Journal)
  was raised and retracted — see `development-of-v59.md` item 20 for
  the secondary-sourcing design discussion; no CSV change there.
- **Phase 2.2 (Unnamed Person) and Phase 2.3 (Anonymous Source), both
  run 2026-10-02 — clean, no changes on either.** 2.2: re-checked the
  same 8 rows from Phase 1's reverse check for Title/SD placement, all
  already correctly structured. 2.3: zero Anonymous Source rows exist
  in the batch (confirmed in Phase 1) — explicitly logged rather than
  silently skipped, since this phase has been missed by accident twice
  before when row counts looked small.
- **Phase 2.5 (Named Organization), run 2026-10-02 — 7 files edited
  across Pass 1/2 (schema/naming) and Pass 3 (category-word SD).**
  Pass 1/2: `193` row 12 (SD "Rap Trio" -> lowercase "rap trio"), row
  19 (SD "Broadcaster" -> lowercase "broadcaster"); `200` row 2 (Name
  "US Central Bank" -> "Federal Reserve" — a genuine naming-
  inconsistency fix, not an inference: the article's very next
  sentence after naming "the Federal Reserve" describes the same
  entity generically as "the US central bank," and there's only one
  US central bank); `198` row 2 (SD "notes" blanked — it described the
  medium the Pentagon spoke through, not what kind of entity the
  Pentagon is, so it didn't fit the category-word test). Pass 3: `192`
  row 1 (SD -> "company," already the established value elsewhere in
  this file, just missed on row 1); `200` row 2 again (SD -> "central
  bank," newly possible after the naming fix above supplied a genuine
  category word from the row's own text); `203` row 7 (SD ->
  "association," matching `197`'s already-accepted "center" precedent
  for a category word that's also a substring of the entity's full
  name); `195` row 1 (SD -> "search giant" — a genuinely novel call
  with no corpus precedent either way, applied per the user's
  reasoning that "search" functions as the category and "giant" as
  size, not an editorializing qualifier — flagged for a second look
  during the later corpus-wide Pass 3 review, not fully settled as
  precedent yet). **Two items flagged, not CSV changes**: `193` row 23
  (RTÉ) and `212`, both added to the "What we need from you right now"
  list above requesting a fresh article capture from SZ — see that
  section for detail.
- **Phase 2.6 Pass 0 (Named Person, general annotation-error audit),
  started 2026-10-02 — `200` rows 9/10 fixed, a tooling gap found and
  closed along the way.** Row 9 was truncated ("...painted by White
  [House]." — missing "economic officials"); row 10 was genuinely
  garbled from an overlapping PDF text layer (a "Read Next" sidebar
  widget rendered at the same position as the body paragraph, so plain
  text extraction interleaved both: "There Ris enao dev Nideenxcte
  here" instead of "There is no evidence here"). Both fixed with
  verified text — row 9 from a clean re-extraction, row 10 by
  separating the two overlapping fonts directly in the PDF's character
  data. The underlying boilerplate text file was also rewritten with
  the correct text (no longer garbled). **Process change**:
  `storypdf_to_text.py` now auto-detects overlapping text layers and
  warns about them on every conversion; see `development-of-v59.md`
  item 50 and `gt-migration-methodology.md`'s Phase 0 section for the
  new guardrail. **Open question, not yet started**: whether a
  retroactive sweep of already-migrated stories' PDFs is needed to
  catch any pre-existing garbling from before this guardrail existed —
  flagged by the user, not yet scoped.
- **`212` article text replaced, 2026-10-06 — fresh-capture gap
  resolved.** SZ supplied a newer PDF of the same Reuters URL (captured
  after the regional-roundup update; headline changed to "Iran awaits US
  move after WSJ report says Trump rejects peace plan"). Converted and
  saved as `212-iran_awaits_us_move_after_wsj_report_says_trump_re.txt`,
  replacing the old `212-trump_rejects_iranian_proposal_to_open_hormuz_and_.txt`.
  Checked every annotated Sourced Statement against the new text: all
  15 rows present (rows 12-15, the original gap, now exact matches;
  rows 1/9/10/11, also weak against the old text, now exact too). Row
  4 is the only non-exact match, and only because the annotation drops
  the quotation marks around Araqchi's direct quote; the wording is
  identical. The 3 overlapping-text-layer warnings were all
  bold-next-to-regular cases (byline, photo-caption "[1/3]", "Our
  Standards" footer), none touching annotated text. Trimmed:
  video-player chrome at top, site chrome after "Our Standards." No
  annotation content touched; `212` is now ready for Phase 0.
- **Phase 2.6 Pass 0 (Named Person) completed across 191-211,
  2026-10-07 — 166 rows screened, fixes applied:**
  - `202` row 4: Trump's Title "Mayor" -> "President" (copy-paste slip
    from the Mamdani half of the joint-row split on row 3).
  - `208` row 7: Title typo "Pesident" -> "President".
  - `199` rows 3/4: Title "GOP Senate nominee/Dr." -> "Dr." The article's
    "GOP Senate nominee" describes Mike Rogers, not El-Sayed ("His
    opponent Dr. Abdul El-Sayed"); El-Sayed's party is never stated, so
    nothing inferred in its place.
  - `191` row 8: Sourced Statement "who runs xAI" -> "who runs SpaceXAI"
    to match the article verbatim (your own SJ on this row already had
    it right).
  - `204` row 5: trailing period removed from Title; Sourced Statement
    dashes restored to the article's "El Niño — from six inches to a
    foot — would".
  - `192` row 14: Sourced Statement dashes restored to the article's
    "— that it's entitled ... of its own —".
  - `209` (11 Heidi Overton rows): double space in Title removed.
  - Batch-wide: 23 invisible zero-width characters (copy-paste
    carryover from Reuters pages) stripped from `200`, `203`, `206`,
    `208`, `209`, across all field types. No visible text changed.
  No changes needed: stage names (Macklemore, CMAT, JMSN), in-article
  brackets (`194` "[by]", `207` "annihilat[ing]"), `191` "Nvidia"
  (article misspells it "Nividia").
- **`124-Ebola_Dulles.csv` row 10 removed, 2026-10-08 (23 -> 22 rows,
  renumbered).** New standing rule, adopted 2026-10-08: annotate the article body only -- photo captions and scrolling-graphic text panels count as body; subtitles (the summary line under the headline) and summary boxes don't, since they recap the body. Row 10 ("A draft DHS memo directs commercial
  airline passengers who traveled to the Democratic Republic of the
  Congo, South Sudan or Uganda in the past 21 days to land at Dulles.")
  annotated the article's summary line, not the body. Nothing lost: the
  memo is already annotated from the body in rows 4 and 8.

- **Version-mix and rewording fixes, 2026-10-08 (from the corpus text
  sweep's version check).** `135-Laos_Divers.csv`: rows 5, 12 and 21
  were annotated from an earlier version of the CBS story (the live
  article was updated during the rescue; the Internet Archive's May 29
  snapshot matches those three rows exactly), while the other 18 rows
  match the later version we have. Rewritten to the later version's
  wording so the story is annotated against one version: row 5 ends
  "...'ticking clock,' Paasi had previously told CBS News." (was "lead
  rescue diver Mikko Paasi told CBS News chief correspondent Matt
  Gutman"); row 12 "the rescued miner" / "the person's name" (was "a
  person" / "their name"), and "It was not immediately clear how the
  miner was rescued" dropped (not in this version); row 21 "who also
  helped lead the 2018 Thai cave rescue" in both Sourced Statement and
  Source Justification (was "who helped lead the rescue of a youth
  soccer team from a flooded cave in Thailand in 2018"), plus
  "well-thought-out". `124-Ebola_Dulles.csv` row 3: "who have been in
  the Democratic Republic of Congo" (the row had shortened it to
  "Congo"). `150-Mark_Walter_Insurance.csv` row 3: restored the
  article's own sentence order ("U.S. federal prosecutors and the SEC
  are investigating Walter's business empire..." -- the row had
  rewritten it in the passive). Please quote sentences exactly as they
  appear in the article, even when shortening or rephrasing seems
  harmless.

- **`164-Lonnie_Bunch.csv` row 1 Sourced Statement replaced, 2026-10-08.**
  The original ("Lonnie Bunch III said Tuesday that he will retire as the
  head of the Smithsonian Institution, a role that placed him at the
  center of President Donald Trump's efforts to reshape how federal
  cultural and historical institutions present American history.")
  matches neither our copy of the CNN article nor today's live page, and
  no archived copy exists -- likely an earlier version of the story; the
  annotator couldn't locate the version used. Replaced with our copy's
  opening sentence, word for word: "Lonnie Bunch III said he will retire
  as the head of the Smithsonian Institution, a role that placed him at
  the center of President Donald Trump's efforts to reshape the world's
  largest museum organization." Other fields unchanged.

- **`53-Iran_Sea_Trade_Blocked.csv` row 1 removed, 2026-10-08 (14 -> 13
  rows, renumbered).** With the fresh PDF of the version you annotated,
  row 1 ("More than a dozen American warships positioned in the Gulf of
  Oman and Arabian Sea are acting as a 'net,' officials said") turned
  out to be the article's subtitle. Under the body-only rule it isn't
  annotatable, and the body already covers it (now rows 2, 8 and 10:
  the dozen-plus Navy ships, "Our net is the Gulf of Oman," and no
  escalation).

- **Graphic-derived rows moved to the graphics side file, 2026-10-08.**
  New standing rule, adopted 2026-10-08: we only annotate text the models actually receive -- article body, photo captions, and scrolling-graphic text panels. Text that belongs to a chart, map, table, or an image of a document isn't part of that, so rows annotating it were moved (not deleted) to `GT-II/graphics-only-rows.csv`, kept in case we ever evaluate models on page images. `65-Housing_Tech_and_Taxes.csv` row 15 (the description of an
  embedded chart: "This chart shows Bay Area tech salaries...") -- 48 ->
  47 rows, renumbered; the same Census comparison is already annotated
  from the body (now row 17). `112-SZ_Hantavirus.csv` row 6 (the intro
  text of CNN's evacuation-flights map: "122 passengers and crew of the
  MV Hondius have been evacuated... Here's where they went:") -- 13 ->
  12 rows, renumbered.

- **`122-SZ_SJ9_Climate_Policy.csv` row 1: name typo fixed, 2026-10-08.**
  Name of Source "Sott Hughes" -> "Scott Hughes" (spelled correctly
  everywhere else in the file). The typo came from the original row
  that jointly credited all five candidates; when that row was split
  into one row per candidate, it carried over into row 1.

- **Copy errors fixed, 2026-10-08 (corpus text sweep, Tier 4).**
  `159-Iran_Deal.csv` rows 14 and 15: garbled characters from copying
  ("Iranâ€™s" -> "Iran's", "UAEâ€™s" -> "UAE's"). `101-Cyclone_Freddy.csv`
  row 5: the Sourced Statement began "DoDMA Spokesperson Chipiliro
  Khamula says...", but the article's sentence says only "Spokesperson
  Chipiliro Khamula says..." -- "DoDMA" removed so the quote matches
  the article (DoDMA is named in the sentence before). Related, left
  for the Title of Source pass: all of Khamula's rows (5-8, 20, 21, 31)
  have Title "DoDMA Spokesperson", but the article states only
  "Spokesperson". `112-SZ_Hantavirus.csv` row 1: "he told CNN's Erin
  Burnett on Tuesday" -> the article's own wording, "he said on CNN's
  'Erin Burnett OutFront' on Tuesday." Please paste sentences exactly
  as they appear; don't add words, even when they seem helpful.

- **Invisible/odd characters cleaned up, 2026-10-08 (no visible text changed; makes rows match the article text exactly): non-breaking spaces -> normal spaces, non-breaking hyphens -> normal hyphens, the "ʼ" modifier apostrophe -> "’", zero-width characters removed, and the "ﬁ" ligature -> "fi". These usually come from copying text out of web pages or PDFs.** Files: `101-Cyclone_Freddy.csv`, `102-Detroit_Tenants.csv`,
  `123-Iran_Uranium.csv` (5 rows had "ﬁ" for "fi" -- they now match the
  article), `150-Mark_Walter_Insurance.csv`,
  `156-Trump_Immigration_Collapse.csv`, `180-UK_Israel_Ban.csv`.

- **`212-Trump_Rejects_Peace_Iran.csv` converted to CSV (Phase 0),
  2026-10-10.** Your XLSX was already in the full 7-column format, so this
  was format conversion only: 15 rows, whitespace trimmed, one hidden
  zero-width character removed from row 15's Source Justification. All
  15 Sourced Statements match the article text you supplied. Flagged
  for a later pass: row 13 is a Named Organization with Name of Source
  left blank -- the name ("Yemen's Saudi-led coalition") is in its
  Source Justification instead.

- **Unnamed Group of People rows tidied, 2026-10-10 (Phase 2.1).**
  `84-Paradise_Lake_Fire.csv` row 1: "Fire officials" moved from Name
  of Source to Title of Source; row 3: "Residents" moved from Name of
  Source to Source Descriptors. For Unnamed Group of People, Name of
  Source stays blank; a credentialed role (officials) goes in Title, a
  general description (residents) in Source Descriptors.

- **`53-Iran_Sea_Trade_Blocked.csv` rows 3 and 4 merged, 2026-10-10
  (Phase 2.2).** Row 3 had "An estimated 90 percent of Iran's economy is
  fueled by international sea trade." with the start of the sentence
  ("Adm. Brad Cooper said late Tuesday") cut off; row 4 had the next
  quote typed as an Unnamed Person ("Centcom commander"). Both are from
  Cooper -- "the Centcom commander" refers back to him in the very next
  sentence -- and both are in the same paragraph, so under the
  paragraph rule they form one row: "Adm. Brad Cooper said late Tuesday
  an estimated 90 percent of Iran's economy is fueled by international
  sea trade. "In less than 36 hours ... by sea," the Centcom commander
  said in a separate post." Named Person, Brad Cooper (13 -> 12 rows).
  **Open for discussion:** the two statements reached the reporter by
  different routes ("said late Tuesday" vs. "said in a separate post"),
  so we want to talk through whether that should ever keep them as
  separate rows. For now the paragraph rule applies.

- **Named Organization rows tidied, 2026-10-15 (Phase 2.5 Pass 1).**
  `53-Iran_Sea_Trade_Blocked.csv`: names now match the article -- row 1
  "US Military" -> "U.S. military"; rows 2 and 12 "US Central Command"
  -> "U.S. Central Command" (row 12's sentence says "Centcom", the same
  organization). Rows 1 and 12 also had the attribution cut from their
  sentences; both are now the full sentences ("...by sea, the U.S.
  military said, hours after military officials said U.S. warships
  confronted six merchant vessels departing an Iranian port and forced
  them to turn back." / "...non-Iranian ports," Centcom said in its
  statement."). `212-Trump_Rejects_Peace_Iran.csv` row 13: Name of
  Source was blank -- now "Yemen's Saudi-led coalition" (that text had
  been put in Source Justification, which is now blank).

- **Organization category words, 2026-10-15 (Phase 2.5 Pass 3).**
  For a Named Organization, Source Descriptors holds a bare word for what
  kind of organization it is, when the article uses that word for it.
  `53-Iran_Sea_Trade_Blocked.csv` row 11: added "network" (the sentence
  refers to Iranian Press TV as "the network").
  `212-Trump_Rejects_Peace_Iran.csv` row 11: removed "report" -- that's
  the Journal's article, the medium, not what kind of organization The
  Wall Street Journal is.

- **Name of Source "Donald Trump" -> "Trump", 2026-10-15.** Name of Source is the person's full name *as stated in the article*. Where the article only ever says "Mr. Trump", "President Trump" or "Trump" -- never "Donald Trump" -- the Name is "Trump".
  `69-Trump_Assassination_Suspect.csv` row 16;
  `202-Trump_Mamdani_Affordable.csv` rows 4, 8;
  `207-US_Iran_Unexpected_Problems.csv` rows 4, 7.

- **Titles, 2026-10-16 (Phase 2.6 Pass 1).** A title must be stated
  for the person in the article, word for word where possible.
  `53-Iran_Sea_Trade_Blocked.csv` row 3: "Admiral" -> "Adm.; Centcom
  commander" (both stated: "Adm. Brad Cooper" and "the Centcom
  commander"); row 5: "Chinese Foreign Ministry spokesperson" ->
  "Foreign Ministry spokesperson" ("Chinese" isn't in the article).
  `69-Trump_Assassination_Suspect.csv` row 16: "President of the United
  States of America" -> "President" (the article says "President
  Trump").

- **Descriptors and titles from the candidate list, 2026-10-16.** The
  article introduces the District 9 candidates in one sentence: "...serial
  entrepreneur Mike Hennessy, longtime city employee Gordon Chester,
  licensed marriage and family therapist Genny Altwer and tech worker
  Rick Ator." These apply to every row for that person, not only the rows
  near that sentence. Owner/founder/entrepreneur and licensed professions
  are titles; jobs without stated authority go in Source Descriptors, and
  "longtime" is left out (it only says how long).
  `122-SZ_SJ9_Climate_Policy.csv`: row 2 Title "serial entrepreneur"
  (Hennessy); row 4 Title "licensed marriage and family therapist"
  (Altwer); rows 3, 16 Source Descriptors "District 9 candidate, city
  employee" and row 16 Title "longtime city employee" removed (Chester);
  rows 5, 11 Source Descriptors "District 9 candidate, tech worker" and
  row 11 Title "tech worker" removed (Ator).

- **Source Justification, 2026-10-16 (Phase 3).** Source Justification
  says why the source matters to the story, not how the reporter got the
  statement, and not the statement repeated.
  `212-Trump_Rejects_Peace_Iran.csv` row 15: blanked (it repeated the
  Sourced Statement's own attribution, "United Nations children's agency
  said on Friday, reporting a rise in acute malnutrition among
  children.").

- **Missed statement and cut-off attributions, 2026-10-16 (Phase 4a).**
  `53-Iran_Sea_Trade_Blocked.csv`: new row 12, Anonymous Source (Title
  "US officials"): "U.S. warships are not escorting the intercepted
  vessels back to the Iranian ports, both officials said." -- no row had
  it; old row 12 is now row 13. Rows 2 and 5: the attribution at the end
  of the sentence had been cut off; a Sourced Statement keeps the whole
  sentence, including "..., U.S. Central Command said in a post on social
  media." (row 2) and "...," Foreign Ministry spokesperson Guo Jiakun
  said." (row 5).

- **Name of Source cleared for unnamed sources, 2026-10-23.** Unnamed Person and Unnamed Group of People sources have no Name of
  Source; the words that had been put there now go in Title of Source if
  they name a role with authority or expertise (officials, regulators,
  economists, officers, leaders), and in Source Descriptors otherwise
  (residents, advocates, organizers, investors).
  `123-Iran_Uranium.csv` rows 14-15: Name "Iranian Officials" -> Title
  "Iranian officials". `151-US_Housing_Pressure.csv` rows 3, 5: Name
  "Economists" -> Title "economists" (row 3's Title "Economists polled by
  Reuters" trimmed: how the estimate was gathered isn't part of the
  title). `106-Palestine_Protest_Arrests.csv` row 1: Name "Officer" ->
  Title "officer".

- **One row per paragraph, 2026-11-02 (Phase 4a).** `84-Paradise_Lake_Fire.csv`
  (now 5 rows): James Lico's old rows 4 and 5 are now row 4 ("Lico noted
  the complexity of this portion, stating, '...' He added, '...'"), and
  old rows 6 and 7 are now row 5 ("'We appreciate the patience...' Lico
  said. 'There is urgency...'"). Each pair is a single paragraph on the
  KTVN page (one `<p>` in the page's HTML), so it is one row. The page
  shows each quote in italics with "Lico said." in bold, which makes the
  quotes look like separate pieces -- an easy split to make.

- **Sourced Statement completed, 2026-11-02 (Phase 4a).** `91-El_Nino.csv`
  row 10 (Wenju Cai): added the next sentence of the same paragraph, "The
  rapid switch caused a high-pressure system over the western Pacific to
  channel warm, wet air from the tropics and ram it toward China." -- it
  continues Dr. Cai's explanation. Rows 12 and 13 (also Dr. Cai, same
  paragraph) stay separate rows: row 12 is what he said, row 13 is what
  his team's study found.

- **Phase 4a fixes, 2026-11-02.** `98-Trump_China.csv` (5 rows):
  - Row 1: comma restored ("...some of the executives, according to the
    White House official...").
  - Row 2 (Elon Musk): old rows 2 and 3 merged -- both are in one
    paragraph ("The billionaire... feuded with Trump... Musk eventually
    said that he regretted some of his posts on X about Trump.").
  - New row 4, Named Organization "Commerce Department's Bureau of
    Industry and Security": "Nvidia must ensure that there is an adequate
    supply in the U.S., and the H200 chips must undergo a third-party
    review before being exported to China, according to new rules set by
    the Commerce Department's Bureau of Industry and Security. But the new
    rules lower the bar for exports." -- "according to new rules set
    by..." is an explicit attribution that had no row. (The organization
    that issues rules in its own name is the source.)

- **Sourced Statement copied exactly, 2026-11-02 (Phase 4a).**
  `212-Trump_Rejects_Peace_Iran.csv` row 4: quotation marks restored around
  "If there is seriousness on the US side to come to a deal and reopen the
  Strait of Hormuz, everything is now prepared,".

- **Photo captions no longer annotated, 2026-11-05.** Photo captions
  (and their photo credits) are not part of the annotated text any more;
  they now sit in each story's header as a `Caption:` field, so the models
  don't see them either. Rows that came from a caption were moved to
  `GT-II/caption-only-rows.csv` (kept, not deleted).
  `55-Chicago_Pentagon.csv` (now 24 rows): old row 1 (Andrew Torrence,
  "...said he moved to the neighborhood with the hopes of setting down
  roots...") and old row 20 (Jayna McGruder, "...said she first realized
  the impact of inequality...") -- both from photo captions; both people
  still have their other rows from the article body.
  `122-SZ_SJ9_Climate_Policy.csv` (now 20 rows): old row 21 (organizers
  behind the candidate forum) -- from a photo caption.
  `135-Laos_Divers.csv` (now 20 rows): old row 13 ("This video grab
  provided by the Association Of Volunteers For Lao People, shows
  rescuers evacuating...") -- from a video-still caption.

- **Source Descriptors filled on every row for the same source, 2026-11-05.**
  A source's Source Descriptors now apply to all of that source's rows in
  a story, not only the row whose sentence states the descriptor.
  `122-SZ_SJ9_Climate_Policy.csv` rows 13 and 14 (Scott Hughes): blank ->
  "District 9 candidate" (as on row 1). `163-Jill_Stein_Court.csv` rows 5
  and 6 (Washington University): blank -> "university" (as on row 11).
- **Organization category words filled from the article, 2026-11-05.**
  When the article calls an organization by a category word ("the
  nonprofit," "a think tank"), that word goes in Source Descriptors on all
  of the organization's rows. `65-Housing_Tech_and_Taxes.csv` rows 20 and
  28 (Joint Venture Silicon Valley): "nonprofit" -> "nonprofit, think tank"
  ("Joint Venture Silicon Valley, a think tank that studies the region").
  `76-Ghiradelli_Square.csv` rows 2 and 8 (1823 Partners): blank -> "asset
  management firm, company" ("the asset management firm has ties to...";
  "1823 Partners is an independent company"). `95-China_Death_Penalty.csv`
  rows 6 and 7 (PLA Daily): blank -> "newspaper" ("the official PLA Daily
  newspaper"). *(95's annotator isn't recorded in the repo; filed here
  under SZ -- move to AV if that's wrong.)*

- **Source Descriptors must be the reporter's own word, 2026-11-05.** A
  Source Descriptor has to be a word the article actually uses for that
  source; a label worked out from other facts in the story doesn't count.
  `105-Texas_Plantation_Prison.csv`: Robert Hockley (rows 4-11, 13, 22,
  35, 36, 43, 46, 47) "formerly incarcerated" -> blank (the article never
  calls him that; his release is in his Source Justification and row 43);
  Craig Anderson (rows 12, 20, 23, 24) "formerly incarcerated" -> "recently
  released" ("as the recently released Anderson told us"); Mary McConnell
  (row 25) "wife of an incarcerated person" -> blank; Tona Southards
  Naranjo (row 29) "mother" -> blank (the article says "her son Jon" but
  never calls her his mother). `102-Detroit_Tenants.csv` rows 23-25
  (Sammie Lewis): "organizer" -> blank, and Source Justification "an
  organizer with Detroit Eviction Defense;who has been with the group for
  a couple years" -> "who has been with the group for a couple years"
  (the first part isn't in the article). `103-Durham_Tenants.csv` rows
  7-10, 19, 26-28, 33 (Leslie St Dre): "tenant organizer" -> "neighbor"
  ("her neighbors, Leslie... and Jadelynn St Dre"; the article also says
  "The St Dres weren't ERA tenants"). `61-Atlantic_Current.csv` row 6
  (Science Advances): "study" -> "research" ("The new research, published
  in the journal Science Advances"). `78-BART_Closures.csv` rows 1, 7, 8,
  19 (Emily Yarrington): "resident" -> blank. `55-Chicago_Pentagon.csv`
  rows 2, 11, 23 (Jerry Whirley): "South Shore homeowner" -> blank (the
  article says "a few blocks from his South Shore home", not homeowner).
  *(103, 105 and 78: annotator not recorded in the repo; filed here
  under SZ -- move to AV if that's wrong.)*

- **Organization Source Descriptors trimmed to the category word, 2026-11-05.**
  For an organization, the Source Descriptor is the bare word for what kind
  of group it is, copied from the article; descriptive words around it
  stay in Source Justification. `104-Irish_Centre.csv` row 2 (Frick Park
  Friends): "collective of neighbors" -> "collective", and Source
  Justification blank -> "a collective of disgruntled neighbors" (the
  article's words; it's why they're in the story -- they live next to the
  proposed building). `108-SZ_YVA.csv` rows 5 and 9 (Youth vs Apocalypse):
  "climate justice collective" -> "collective" (the article says "a
  youth-led, Bay Area-based collective of young climate justice
  activists", which is already the Source Justification). *(104:
  annotator not recorded in the repo; filed here under SZ.)*

- **Source Descriptors: kind-words kept, describing words left out, 2026-11-05.**
  A Source Descriptor keeps the words directly in front of it that say
  what kind of person or group it is (a topic word like "affordable
  housing," a word like "presidential" or "full-time," an identity or
  political word like "unhoused" or "far-right," and for people, a place
  like "Florida"). It leaves out words that rate or describe this one
  source ("longtime," "top," "disgruntled") and anything that comes after
  it ("of four," "of the settlements"); those stay in Source Justification.
  `178-AFD_Win_Europe.csv`: row 11 Elon Musk "billionaire" -> "nativist US
  tech billionaire"; rows 12, 13 Ulrich Siegmund "candidate" -> "lead
  candidate"; row 15 AfD "party" -> "far-right party"; row 2 (group)
  "figures" -> "nationalist figures". `163-Jill_Stein_Court.csv`: rows 2,
  14 Jill Stein "candidate" -> "former presidential candidate"; rows 7-10
  KSDK "news station" -> "local news station".
  `177-AFD_Prediction_German_State_Election.csv` rows 6, 8-11 (Siegmund):
  "candidate, former fragrance salesman" -> "lead candidate, former
  fragrance salesman". `180-UK_Israel_Ban.csv`: row 26 Randy Fine
  "Republican" -> "Florida Republican"; row 25 Mike Huckabee "supporter of
  the settlements" -> "supporter". `183-Trump_Promise.csv` row 16 (Bernie
  Moreno): "Republican" -> "Ohio Republican". `107-Jackson_Water.csv` rows
  12-14 (Shenetha James): "mother of four" -> "mother".
  `152-Socialists_Florida.csv` row 3 (Oliver Larkin): "activist and union
  organizer" -> "activist, union organizer". `103-Durham_Tenants.csv` rows
  3, 17, 20, 32 (Eno River Association): "land trust and conservation
  nonprofit" -> "land trust, conservation nonprofit".
  `105-Texas_Plantation_Prison.csv` row 21 (group): "current inmates and
  their families" -> "current inmates, families". *(107 and 152: annotator
  not recorded in the repo; filed here under SZ.)*

- **Source Descriptors lowercased, 2026-11-06.**
  A Source Descriptor is copied in the article's own case. A capital
  letter that is there only because the word starts a sentence ("Critics
  have also warned...") isn't part of it, so the descriptor is lowercase
  unless it's a proper noun.
  `84-Paradise_Lake_Fire.csv` row 3 (group): "Residents" -> "residents".
  `177-AFD_Prediction_German_State_Election.csv` row 15 and
  `179-AFD_Majority_Fail.csv` row 29 (groups): "Critics" -> "critics".

- **Title copied as written, 2026-11-06.** `53-Iran_Sea_Trade_Blocked.csv`
  rows 6-10 and 12 (Anonymous Source): Title of Source "US officials" ->
  "U.S. officials", the way the article writes it ("two U.S. officials
  said").

- **Anonymous Source: the two sources counted as one disclosed pair, 2026-11-09.**
  `123-Iran_Uranium.csv` row 1 ("...two senior Iranian sources said,
  hardening Tehran's stance..."): Unnamed Group of People -> Anonymous
  Source. The article says "one of the two Iranian sources, who spoke on
  condition of anonymity," and credits the two sources together in rows 1,
  8, 10, 13 and 16, so the disclosure covers the pair; row 10 (same pair,
  same wording) was already Anonymous Source.

- **Phase 2.1 (groups of people), late-text stories, 2026-11-09.**
  `157-Pentagon_Iran_Military_Presence.csv` row 20 ("More hawkish voices
  in Washington have sought to influence the Pentagon's planning..."):
  Source Descriptors blank -> "voices in Washington" ("hawkish" describes
  them and is left out; the place is kept). *(157: annotator not recorded
  in the repo; filed here under SZ.)*

- **Phase 2.3 (Anonymous Source), late-text stories, 2026-11-09.**
  `123-Iran_Uranium.csv` row 1: Title of Source "two senior Iranian
  Sources" -> blank ("sources" names the source's role in the reporting,
  not a position; the story's other "sources" rows are already blank).
  `130-Cuban_Troops.csv` row 5: Title of Source blank -> "defense official"
  ("said a defense official, granted anonymity...").
  `157-Pentagon_Iran_Military_Presence.csv` row 8: Title of Source blank ->
  "U.S. official" ("said a U.S. official"). *(130 and 157: annotator not
  recorded in the repo; filed here under SZ.)*

- **Phase 2.4 (documents), late-text stories, 2026-11-09.**
  For a document, Name of Source is only the document's own title, and the
  kind of document ("report," "polls") goes in Source Descriptors. When the
  article names the organization the document comes from, that
  organization is the source (Named Organization).
  `129-Blue_Texas.csv`: rows 1 and 3 Name "National Polls" -> blank,
  Source Descriptors "national polls"; row 2 Name "Presidential Election
  Results" -> blank, Source Descriptors "presidential election results";
  rows 6 and 7 Name "Polls" -> blank, Source Descriptors "polls".
  `130-Cuban_Troops.csv` row 2: Name "Flight Tracking Sites" -> blank,
  Source Descriptors "flight tracking sites".
  `157-Pentagon_Iran_Military_Presence.csv`: row 19 Document "policy
  proposal from Defense Priorities" -> Named Organization / Defense
  Priorities, Source Descriptors "think tank"; row 21 Document "report by
  Jewish Institute for National Security of America" -> Named Organization
  / Jewish Institute for National Security of America, Source Descriptors
  "pro-Israel group". *(129, 130 and 157: annotator not recorded in the
  repo; filed here under SZ.)*

- **Phase 2.5 (organizations), late-text stories, 2026-11-10.**
  `123-Iran_Uranium.csv` rows 4 and 17 ("Israeli officials have told
  Reuters..."): Named Organization / Israel -> Unnamed Group of People,
  Title of Source "Israeli officials" -- "Israeli" describes the officials
  but doesn't name an organization; matches rows 14 and 15 ("Iranian
  officials"). `129-Blue_Texas.csv` rows 4 and 5: Name of Source "New York
  Times/Sienna" -> "New York Times/Siena", as the article spells it.

- **Phase 2.5 (organizations' Source Descriptors), late-text stories, 2026-11-10.**
  The article's word for what kind of organization it is goes in Source
  Descriptors on all of that organization's rows.
  `150-Mark_Walter_Insurance.csv` rows 1, 2, 4, 5 (TWG Global): blank ->
  "holding company" ("TWG Global, Walter's holding company"). *(150:
  annotator not recorded in the repo; filed here under SZ.)*

- **Phase 2.6 Pass 0 (named people), late-text stories, 2026-11-11.**
  `106-Palestine_Protest_Arrests.csv` rows 5, 16, 17 (George Lobb): Title
  "Lawyer with..." / "a lawyer with..." -> "lawyer with the Austin Lawyers
  Guild." `123-Iran_Uranium.csv` row 11 (Mohammad Baqer Qalibaf): Title
  "Iran's peace negotiator" -> "Iran's top peace negotiator" (a rank word
  is part of a rank title, like "senior vice president").
  `130-Cuban_Troops.csv` rows 3, 4 (Mark Cancian): "Senior analyst at
  Center for..." -> "senior analyst at the Center for Strategic and
  International Studies." *(130: annotator not recorded in the repo; filed
  here under SZ.)*

- **Phase 2.6 Pass 1 (titles of named people), late-text stories, 2026-11-11.**
  `130-Cuban_Troops.csv` rows 3, 4 (Mark Cancian): Title -> "senior analyst
  at the Center for Strategic and International Studies; former Pentagon
  official" (both positions the article gives). `157-Pentagon_Iran_Military_Presence.csv`
  rows 11, 12 (Michael Ratney): Title "Deputy Chief of Mission in Qatar" ->
  "former diplomat" (the article says he is "a former diplomat who served
  as" ambassador and deputy chief of mission; the old Title read as a
  current post). `158-Better_Home_Finance_CEO_Fired.csv`, all 22 Vishal Garg
  rows: Title "the founder and former chief of mortgage lender Better Home
  & Finance" -> "founder and former chief, Better Home & Finance" ("mortgage
  lender" describes the company, not his position). *(130, 157 and 158:
  annotator not recorded in the repo; filed here under SZ.)*

- **Phase 2.6 Pass 2 (missed titles), late-text stories, 2026-11-12.**
  `130-Cuban_Troops.csv` row 7 (Joe Plenzler): Title of Source blank ->
  "retired Marine Corps officer" ("said Joe Plenzler, a retired Marine
  Corps officer"). *(130: annotator not recorded in the repo; filed here
  under SZ.)*

- **Phase 2.6 Pass 3 (named people's Source Descriptors), late-text stories, 2026-11-12.**
  `106-Palestine_Protest_Arrests.csv`: rows 2, 3, 4, 7 (Anne-Marie Jardine)
  -> "international relations and global studies senior"; rows 9-12, 14
  (Benjamin Kern) -> "demonstrator" ("one of the demonstrators"); rows 15,
  18 (Ibraheem) -> "protester" ("another protester"); rows 19, 21, 22
  (Citlalli Soto-Ferate) -> "health and society senior."
  `153-Guinea_Pig_Sanctuary.csv` row 2 (Janet Woodman) -> "volunteer" ("a
  volunteer at the nonprofit"). *(153: annotator not recorded in the repo;
  filed here under SZ.)*

- **Phase 3 checks, late-text stories, 2026-11-12.**
  Source Justification should say why the source is in the story. A word
  "null," the attribution alone ("polls show"), or a repeat of the Title
  isn't that, so those were cleared.
  `157-Pentagon_Iran_Military_Presence.csv`: row 8 "U.S. official" (same as
  its Title) -> blank; rows 9 and 10 "person familiar with the posture
  adjustments said" -> "person familiar with the posture adjustments"
  (the word "said" is the attribution). `130-Cuban_Troops.csv` row 7 (Joe
  Plenzler) "Retired Marine Corps Officer" (same as his Title) -> blank.
  `123-Iran_Uranium.csv` row 11 (Mohammad Baqer Qalibaf) "Iran's top peace
  negotiator" (same as his Title) -> blank.

- **Phase 4 (rows checked against the article), `106-Palestine_Protest_Arrests.csv`, 2026-11-13.**
  New row 22 (group, Source Descriptors "protesters"): "About 150
  protesters stood outside the jail all night calling for those inside to
  be released." -- a call made by the group, so a sourced statement. 106
  now has 23 rows (the last Soto-Ferate row moved from 22 to 23). The
  reporter's own sentences about what happened to each person in custody
  were left out of their rows, whether they came before or after what the
  person said: row 7 (Anne-Marie Jardine) without "At about midnight,
  Jardine was put in a solitary cell." and "She was released at around 8
  a.m. on Thursday, April 25 with other protesters."; row 18 (Ibraheem)
  without "At 11:50 p.m., officers moved Ibraheem to a solitary cell, where
  he remained for 12 hours before his release."; row 19 (Citlalli
  Soto-Ferate) without "When Soto-Ferate arrived at the jail, she and her
  fellow demonstrators were processed and changed into prison clothes.";
  row 21 (Soto-Ferate) without "Officers also moved Soto-Ferate to a
  solitary cell at 11 p.m." and "Soto-Ferate was released at
  approximately 8 a.m."

- **Phase 4a, `123-Iran_Uranium.csv`, 2026-11-13.** Four sourced
  statements had no row; added (now 27 rows). Rows 5 and 6: "Israel, the
  United States and other Western states have long accused Iran of seeking
  nuclear weapons..." -- one Named Organization row each for Israel and
  United States (a statement credited to two sources is repeated, one row
  per source; "other Western states" names no source and gets no row).
  Row 18: "Before the war, Iran signalled willingness to ship out half of
  its stockpile..." -> Named Organization / Iran. Row 21: "Tehran has vowed
  a crushing response if attacked." -> Named Organization / Iran (the
  annotator's name for Tehran in rows 7 and 27).

- **Phase 4b, `123-Iran_Uranium.csv`, 2026-11-13.** Sourced Statements
  made to match the article text (words run together in the PDF
  conversion): row 3 "reportersat" -> "reporters at", row 4 "willbe" ->
  "will be", row 8 "itssupport" -> "its support", row 11 "redlines" ->
  "red lines", row 12 "asense" -> "a sense", row 15 "itsenriched" -> "its
  enriched"; row 13: opening quote mark restored before "obvious and
  hidden moves by the enemy".

- **Anonymity disclosure in Source Justification, 2026-11-13.** The
  reporter's disclosure that a source asked for or was granted anonymity
  is Source Justification on every row for that source (the basis for the
  Anonymous Source type; now v61 Note 22a). Rows missing it were filled;
  existing valid SJ text kept, with the disclosure after "; ".
  `123-Iran_Uranium.csv` rows 1, 9, 10, 12, 15, 19, 22, 23: -> "who spoke
  on condition of anonymity because of the sensitivity of the matter"
  (replaces the labels "one of the two Iranian sources" in row 9 and "two
  senior Iranian sources" in row 12, which are attribution wording only).
  `53-Iran_Sea_Trade_Blocked.csv` rows 6, 7, 8, 9, 10, 12: blank -> "The
  officials spoke on the condition of anonymity to disclose details about
  the operation that have not been made public."
  `157-Pentagon_Iran_Military_Presence.csv` rows 1, 2, 4, 5, 6, 7, 8, 9,
  10, 18: + "who like others spoke on the condition of anonymity to
  discuss the sensitive analysis" (a blanket disclosure covering all of
  the story's anonymous sources); the "...familiar with..." SJs in rows 1,
  2, 7, 9, 10, 18 kept in front of it.

- **Phase 4a, `129-Blue_Texas.csv`, 2026-11-13.** New row 8: "Back in
  2018, Beto O’Rourke never led a poll collected by RealClearPolitics
  against Ted Cruz." -> Named Organization / RealClearPolitics (the named
  organization the polls come from). Now 8 rows. *(129: annotator not
  recorded in the repo; filed here under SZ.)*

- **Phase 4b, `129-Blue_Texas.csv`, 2026-11-13.** Source Justifications
  that weren't about the source: row 1 (national polls) "But there’s
  another reason Democrats might pull off a statewide win... demographics."
  (the columnist's thesis) -> blank; row 3 (national polls) "Latest
  National Polls" (a label only) -> blank; row 4 (New York Times/Siena)
  "and the Republican grip on Texas is in danger as a result. The latest
  New York Times/Siena poll is representative" -> "The latest New York
  Times/Siena poll is representative" (the first half is row 3's
  statement); row 6 (polls) "He’ll be facing Mr. Paxton, who enters the
  general election campaign already damaged..." (describes Paxton, the
  poll's subject) -> blank.

- **Phase 4a, `130-Cuban_Troops.csv`, 2026-11-13.** New row 1: "The
  president has floated an invasion of the island after economic and
  political pressure failed to topple the Communist government." -> Named
  Person / Donald Trump, Title "president" (as written; named in the
  paragraph before). Old rows 1-7 -> 2-8. *(130: annotator not recorded in
  the repo; filed here under SZ.)*

- **Phase 4b, `130-Cuban_Troops.csv`, 2026-11-13.** Row 4 (Mark
  Cancian): Sourced Statement's opening straight quote -> curly, as in the
  text. Rows 4 and 5 (Cancian): Source Justification "The Nimitz arrived in
  the region on the same day as the U.S. indicted former president Raul
  Castro, in what appeared a public show of force;  a former Pentagon
  official" -> "a former Pentagon official" (the dropped sentence is about
  the ship, not Cancian). Row 6 (anonymous defense official): Source
  Justification "a defense official granted anonymity..." -> "a defense
  official, granted anonymity to speak candidly about military operations"
  (comma restored, verbatim).

- **Phase 4a, `150-Mark_Walter_Insurance.csv`, 2026-11-13.** New row 5:
  "Ratings agency S&P Global revised its outlook on Delaware Life to
  negative last month..." -> Named Organization / S&P Global, Source
  Descriptors "ratings agency". Old row 5 -> 6. *(150: annotator not
  recorded in the repo; filed here under SZ.)*

- **Phase 4b, `150-Mark_Walter_Insurance.csv`, 2026-11-13.** Row 4
  (TWG Global spokesperson): Sourced Statement's curly quote marks ->
  straight, as in the text.

- **Phase 4a, `151-US_Housing_Pressure.csv`, 2026-11-13.** New row 1:
  "The manufacturing sector, meanwhile, continues to ride the coattails of
  the artificial intelligence build out, with a Federal Reserve measure of
  factory output hitting its highest in more than four years last
  month..." -> Named Organization / Federal Reserve (as written). Sentences
  that carry on a release's own figures appended: row 5 (National
  Association of Realtors) + "They were down 2.3% on the month to the
  lowest level since January."; row 8 (Mortgage Bankers Association) +
  "Still, at 6.77% it remains near the highest in more than a year." Old
  rows 1-11 -> 2-12.

- **Phase 4b, `151-US_Housing_Pressure.csv`, 2026-11-13.** Rows 3 (Ben
  Ayers), 11 and 12 (Bernard Yaros): Sourced Statements' curly quote marks
  -> straight, as in the text.

- **Phase 4a, `153-Guinea_Pig_Sanctuary.csv`, 2026-11-13.** New row 2:
  "Maui is friendly and gives kisses. Gritty is older, but shy. Rio has
  spunk and likes to run around." -> Unnamed Group of People, Source
  Descriptors "volunteers" (the personalities the volunteers "tick off," as
  the paragraph before says). Old rows 2-9 -> 3-10. *(153: annotator not
  recorded in the repo; filed here under SZ.)*

- **Phase 4b, `153-Guinea_Pig_Sanctuary.csv`, 2026-11-13.** Sourced
  Statements made to match the text: rows 3 (Janet Woodman), 6, 7, 10
  (Kimie Smothermon) -- straight apostrophes and row 10's straight closing
  quote -> curly. Smothermon's Source Justification (rows 1, 5, 6, 7, 8,
  9, 10): second sentence dropped ("On any given day, 350 to 400 guinea
  pigs are housed in a modest workspace, split into cages stacked from
  the floor to ceiling." describes the workspace, not her); now
  "Smothermon is the owner of the Guinea Pig Sanctuary, in Salisbury,
  Massachusetts, roughly 40 miles (64 kilometers) north of Boston."

- **Phase 4a, `157-Pentagon_Iran_Military_Presence.csv`, 2026-11-13.**
  New rows 11 and 12, Named Organization / The Washington Post (the
  outlet credits its own earlier reporting): "Six U.S. service members
  were killed in a March strike on Port Shuaiba... The Washington Post
  reported in May." and "While defending its personnel in the region, the
  U.S. military has also expended more than 1,000 sophisticated
  air-defense interceptors... The Post reported in March." Old rows 11-21
  -> 13-23. *(157: annotator not recorded in the repo; filed here under
  SZ.)*

- **Phase 4b, `157-Pentagon_Iran_Military_Presence.csv`, 2026-11-13.**
  The article text uses straight quote marks and apostrophes; Sourced
  Statements made to match it in rows 1, 2, 4, 5, 6, 7, 10, 13, 14, 16,
  17, 18, 20, 21, 22, 23 (curly -> straight), and the Source
  Justifications of rows 7 and 20 ("Pentagon's"). Row 23 (Jewish Institute
  for National Security of America): "Ovda Airbase" -> "Ovda base", as the
  text has it. Rows 13 and 14 (Michael Ratney): Source Justification
  completed -- "a former diplomat who served as the U.S. ambassador to
  Saudi Arabia" + "and the deputy chief of mission in Qatar."

- **Phase 4a, `158-Better_Home_Finance_CEO_Fired.csv`, 2026-11-13.** Two
  rows added (now 25). New row 6, Named Person / Vishal Garg: "He
  attributed that success to the company's decision to train AI models...
  which Garg said doubled productivity and slashed loan origination costs
  in half." New row 23, Named Organization / MPAMAG: "It noted that he was
  removed by a unanimous board vote taken without his participation..."
  (continues the Better Home & Finance statement that MPAMAG reports, row
  22). Old rows 6-21 -> 7-22, old 22-23 -> 24-25. *(158: annotator not
  recorded in the repo; filed here under SZ.)*

- **Phase 4b, `158-Better_Home_Finance_CEO_Fired.csv`, 2026-11-13.** Now
  22 rows (numbers below are before renumbering). Row 1 ("...told CNN he
  had been ousted...") -> Named Organization / CNN, Title blank (relayed
  speech: the source is the outlet the statement was given to). Rows 11,
  12, 13 merged into one row (one paragraph, one source: "Garg told how
  Lewis... He then bemoaned..."); rows 14 and 15 merged likewise ("'It's
  not about me,'... a seat on the board."). Row 19 ("Spiro then sent the
  board a letter last week demanding it reinstate Garg...") -> Named
  Person / Alex Spiro, Title "attorney" (he speaks through the letter).
  Rows 22 and 23 ("a representative for Garg said...") -> Unnamed Person,
  Title "representative", Name blank. Missing opening quote marks restored
  in rows 2, 7, 8, 16, 21; final period restored in row 20.

- **Phase 2.6 Pass 1 (Title credentialing), SZ batch 191-211, batch A
  (191-199), 2026-11-13.** Titles as the article writes them:
  `192-OpenAI_Concerning_Behavior.csv` row 3 (Sam Altman) "OpenAI CEO" ->
  "OpenAI chief executive"; rows 13, 14 (Mustafa Suleyman) "Chief Executive
  of Microsoft AI" -> "chief executive of Microsoft AI".
  `194-Mice_Part_Human_Brain.csv` rows 1, 2, 3, 6, 9 (Sergiu Pașca)
  "Professor of Psychiatry" -> "professor of psychiatry"; rows 10, 11
  (Madeline Lancaster) Title -> "group leader at the MRC Laboratory of
  Molecular Biology in Cambridge" on both, and row 10's Source Descriptors
  (the same title text) -> blank. `195-Gemini_Hack.csv` rows 4, 14 (Heather
  Adkins) -> "vice president of security engineering at Google".
  `197-Humpback_Whale.csv` rows 5, 6 (Kathi George) -> "director of
  cetacean conservation biology". `193-Ireland_Ed_Sheeran.csv` rows 22, 27
  (Andrew E. Quinn): Title "journalist" on both (a profession), row 27's
  Source Descriptors "journalist" -> blank. `199-Trump_Tech_Issues.csv` row
  5 (Ken Paxton): Title "Republican Senate candidate, attorney general" ->
  "the state's attorney general", Source Descriptors "candidate" ->
  "Republican Senate candidate" (a candidacy is not an office; the party
  word goes with it); row 19 (Josh Shapiro): Title "Gov." -> "Democratic
  Gov." (the party designation the reporter attaches to the office).
  Rows 2, 11 (Mike Rogers): Title "GOP Senate nominee" -> blank, Source
  Descriptors -> "GOP Senate nominee" (a nominee has won the party's
  primary but holds no office; the story gives him no other title).

- **Phase 2.6 Pass 1 (Title credentialing), SZ batch 191-211, batch B
  (200-211, plus 194 row 5), 2026-11-13.** `200-US_Factory_Fail.csv`: the
  firm each economist works at added to the Title, as written -- row 3
  "chief US economist at Pantheon Macroeconomics", row 6 "lead U.S.
  economist at Oxford Economics", rows 8, 9 "chief economist at
  FWDBONDS", row 10 "chief economist at High Frequency Economics".
  `204-El_Nino_Emergency.csv` rows 3, 4, 5 (Patrick Barnard): "Research
  director at... Santa Cruz." -> "research director at the Center for
  Coastal Climate Resilience at the University of California, Santa Cruz"
  (case as written, stray period dropped). `206-Trump_Welcomes_Xi.csv` row
  12 (Edgard Kagan): Source Descriptors "Center for Strategic and
  International Studies" -> blank (an affiliation, already in the Source
  Justification). `209-FDA_Nominee.csv`: the state the reporter attaches
  added to each senator's Title -- rows 2, 3, 5, 6 (Bill Cassidy) "US
  Senator" -> "US Senator, Louisiana; physician" ("Cassidy, himself a
  physician" -- a licensed profession); rows 9, 10 (Ed Markey) "Democratic
  Senator, Massachusetts"; rows 14, 16, 19 (Patty Murray) "Democratic
  Senator, Washington"; row 20 (Susan Collins) "Republican Senator,
  Maine"; row 22 (Chris Murphy) "Democratic Senator, Connecticut".
  `210-Fox_Trump_Ad.csv` rows 12, 13 (Jamie Raskin): Title "Representative"
  -> "representative, a Maryland Democrat; constitutional scholar" (the
  party and state given for an elected official, plus his stated
  expertise), Source Descriptors "Democrat" -> blank.
  `211-Tasia_Fortune.csv` rows 2, 4 (Samuel Dukes): "Det." -> "Jackson
  police Det." (as written). `194-Mice_Part_Human_Brain.csv` row 5 (Emily
  Jackson): Title -> "professor of law at London School of Economics";
  "chair of a recent report on neural organoids for the Nuffield Council
  on Bioethics" moved to the Source Justification (her role on a specific
  work, beside a position title).

- **Phase 2.6 Pass 3 (Source Descriptors, story-level), SZ batch 191-211,
  Named Person rows, 2026-11-14.** Source Descriptors are the reporter's
  own words from anywhere in the story, in the article's case.
  `193-Ireland_Ed_Sheeran.csv`: rows 1, 6, 24, 25, 26 (Ed Sheeran)
  "Songwriter"/"Singer" -> "songwriter, singer, artist" ("Grammy-award
  winning songwriter," "The singer," "the two artists"); rows 2, 3, 4
  (Macklemore) "Rapper" -> "artist" (the text never calls him a rapper;
  "the two artists"); row 10 (Aaron Rowe) "Singer" -> "Irish singer,
  Dublin singer"; rows 13-18 lowercased as written ("actor," "singer,"
  "broadcaster," "singer," "actor," "musician"). `199-Trump_Tech_Issues.csv`:
  rows 6, 7 (James Talarico) "opponent" -> "Democratic opponent"; row 12
  (Jon Husted) "candidate" -> blank (not a word the text uses for him);
  row 17 (Tom Tiffany) "candidate" -> "Republican candidate"; row 18
  (David Crowley) "opponent" -> "democratic opponent"; row 20 (Stacy
  Garrity) "challenger" -> "Republican challenger". `210-Fox_Trump_Ad.csv`
  row 18 (JMSN) "artist" -> "Detroit-born artist, singer".
  `211-Tasia_Fortune.csv` row 6 (Christy Spivey) "Fortune's mother" ->
  "mother".
  `209-FDA_Nominee.csv` rows 1, 7, 8, 11, 12, 13, 15, 17, 18, 21, 23 (Heidi
  Overton): blank -> "America First ally" (from "turning to an "America
  First" ally"; "FDA nominee" appears only in the headline, which is not
  annotated).
  `193-Ireland_Ed_Sheeran.csv` row 8 (Ruth Coppinger): Title blank ->
  "politician" ("Among politicians, Alan Kelly, a Labour TD, ... while Ruth
  Coppinger of People Before Profit..."; she is named as one of the group,
  and "politician" is treated as a title). Row 7 (Alan Kelly) keeps the
  more specific "Labour TD."

- **Phase 2.6 Pass 3 (Source Descriptors, story-level), SZ batch 191-211,
  organization rows, 2026-11-14.** An organization's Source Descriptor
  goes on every row for it. `192-OpenAI_Concerning_Behavior.csv` rows 5, 8,
  10, 11, 15, 16, 17, 19 (OpenAI): blank -> "company" ("the company").
  `195-Gemini_Hack.csv` rows 2, 3, 12, 13 (Google): blank -> "search giant"
  ("the search giant"). `200-US_Factory_Fail.csv` row 1 (Federal Reserve):
  blank -> "central bank" ("The US central bank").
  `203-Cancelled_Obamacare.csv` rows 3, 4, 8, 12 (CMS): blank -> "health
  agency" ("The health agency").

- **Phase 3 (corpus checks a-e), SZ batch 191-211, 2026-11-14.** (a) Source
  Justifications that were only the attribution or how the reporter got
  the statement: `208-Review_Voter_Rolls.csv` row 1 "according to an
  internal email seen by Reuters" -> blank, row 19 "DHS Secretary Markwayne
  Mullin told Fox News on Saturday" -> blank; `198-Pentagon_UFO.csv` row 2
  "In notes accompanying the documents" -> blank; `210-Fox_Trump_Ad.csv`
  row 16 (AdImpact) "a firm that tracks political ads, told the Wall Street
  Journal" -> "a firm that tracks political ads", row 14 (Wall Street
  Journal) the same text -> blank (it describes AdImpact). (d) Source
  Descriptors: sentence-start capitals lowercased -- `194` row 7 "tests",
  `196` row 9 "allies" and row 14 "polling", `199` row 16 "advocates",
  `204` row 8 "studies"; `193` row 23 (RTÉ) "Broadcaster" -> "national
  broadcaster" ("Ireland's national broadcaster"; the place is left out
  for an organization); `203` row 7 (National Association of Benefits and
  Insurance Professionals) "association" -> "trade group, association".
  Titles of unnamed groups in the article's case: `194` row 8
  "researchers," `196` row 10 "analysts," `200` row 5 "economists" (the
  text has them lowercase); and sentence-start capitals lowercased --
  `194` row 4 "experts," `200` row 4 "economists," `202` row 13
  "officials," `205` row 8 "experts," `206` row 14 "analysts," `211` rows
  1 "investigators" and 3 "authorities."
  `193-Ireland_Ed_Sheeran.csv` row 11 (Beoga): Source Descriptors "Irish
  folk band" -> "Irish folk band, Irish folk group" (both phrases the
  story uses; "Irish folk" names the kind of music, so "Irish" stays).

- **Phase 4 (text fidelity), SZ batch 191-211, 2026-11-15.** Sourced
  Statements made to match the article text exactly (61 rows; mechanical
  only). Curly vs straight quote marks and apostrophes, as the text has
  them: `191` rows 2, 8, 13, 14; `196` rows 1, 6; `198` rows 1, 2, 5, 7, 8,
  10, 12, 14, 16; `199` rows 13, 15; `206` row 1; `207` rows 6, 7; `208`
  rows 11, 15, 16, 17, 23, 29, 31; `210` row 18; `211` rows 2, 6.
  Quotation marks the annotator had left out, restored: `199` rows 4, 23;
  `200` rows 3, 8, 10; `201` rows 1, 3, 7, 9, 13; `202` rows 8, 9, 11;
  `203` row 6; `205` rows 15, 17, 19; `206` row 12; `208` row 25; `209`
  rows 6, 7; `210` row 13. Other characters: `191` rows 10, 11, 12
  "Nvidia's" -> "Nividia’s" (the article's own spelling); `194` row 9
  "Pașca" -> "Paşca" (as spelled in that paragraph; Name unchanged); `196`
  row 7 and `206` row 4 dash spacing/type; `207` row 4 em dashes; `207`
  row 11 and `209` row 9 double space -> single.

- **Phase 4, `192-OpenAI_Concerning_Behavior.csv` and
  `194-Mice_Part_Human_Brain.csv`, 2026-11-15.** 192 row 2 (U.S. tech
  bosses): Source Justification "U.S tech bosses" -> "U.S. tech bosses"
  (as written). 194 row 6 (Sergiu Pașca): the rest of the paragraph
  appended -- "The newborn mice received several injections, each
  containing about 100,000 human brain cells... In total, the rodents
  lacked about 14m mouse brain cells and ended up with about 4m human ones
  – half the brain by volume." (carries on the account he "describes" in
  Nature); row 1: Source Justification "a professor of psychiatry who led
  the research at Stanford University." -> "professor of psychiatry who
  led the research at Stanford University" (as on his other rows).

- **Phase 4, `193-Ireland_Ed_Sheeran.csv`, 2026-11-16.** The article text
  was updated to the Newsweek version of Sep 21, 2026 (user's PDF); it now
  includes the RTÉ paragraph that row 23 (now 27) was annotated from, and a
  new section, "Why the Macklemore Dispute Resonates in Ireland." Rows added
  for that section (now rows 7-10; old rows 7-27 -> 11-31): row 7, Named
  Organization / Ireland’s Department of Foreign Affairs ("Ireland’s
  Department of Foreign Affairs says the country was the first member...
  That position appeared in a joint declaration by the foreign ministers of
  Ireland and Bahrain in 1980."); row 8, Named Organization / Irish
  government ("The Irish government presented recognition as support for
  Palestinian self-determination..."); row 9, Named Person / Simon Harris,
  Title "then-Taoiseach" ("Announcing the decision, then-Taoiseach Simon
  Harris compared it... "Today we use the same language..."");
  row 10, Document, Source Descriptors "survey" ("A nationally
  representative survey of 1,547 people in Ireland last month found...").
  Rows 2, 3, 4 (Macklemore): Source Descriptors "artist" -> "rapper,
  artist" ("The rapper went on to perform...").

- **Phase 4, `195-Gemini_Hack.csv`, `196-Greenland_Agreement.csv`,
  `197-Humpback_Whale.csv`, 2026-11-16.** 195 rows 10 and 11 (Irregular)
  merged into one row -- one statement in one paragraph ("“This is the same
  issue... separate incident,” an Irregular spokesperson said in a
  statement. “All relevant labs were notified in late July...”"); later
  rows renumbered (now 14). 196 row 3 (Greenland): Source Justification
  "The governments of Greenland and Denmark, which has sovereignty over the
  island," -> blank (the attribution phrase); row 4 (Denmark): -> "which has
  sovereignty over the island." 197 row 2 removed (Unnamed Group of People,
  scientists: "The rare inland visit from a whale has attracted the
  attention of scientists who want to know what kind of condition it’s in
  and how it wound up so far from the sea." -- credits the group with no
  claim); later rows renumbered (now 9).
  195 row 12 removed (Named Organization, Google: "A Google spokesperson
  declined to identify the exact Gemini model involved." -- a refusal to
  the reporter with nothing said); row 13 -> 12 (now 13 rows).

- **Phase 4, `198-Pentagon_UFO.csv` and `199-Trump_Tech_Issues.csv`,
  2026-11-16.** 198 row 4 (researchers): Title "Researchers" ->
  "researchers" (sentence-start capital). 198 row 14: Unnamed Person,
  Title "reporting officer" -> Named Organization / The Pentagon ("“The
  reporting officer described the phenomenon...,” the military said in notes
  accompanying one of the clips" -- the military's statement is the primary
  attribution; the notes are the Pentagon's, as in row 2). 199 rows 8, 14,
  21: Document, Source Descriptors "poll" -> Named Organization / Marquette
  ("a Marquette poll," "The Marquette poll"), Source Descriptors blank;
  row 9: Document -> Named Organization / Quinnipiac University ("A March
  Quinnipiac University poll"). 199 new row 1, Named Person / Donald Trump,
  Title "President": "In the final months before the high-stakes midterm
  election, President Donald Trump has come out against regulation on
  controversial tech issues..." (old rows 1-23 -> 2-24).

- **Phase 4, `200-US_Factory_Fail.csv`, `201-Trump_Backs_Off_Greenland.csv`,
  `202-Trump_Mamdani_Affordable.csv`, and `196-Greenland_Agreement.csv`,
  2026-11-16.** 200 rows 1 and 2 (Federal Reserve) merged -- same source,
  one paragraph ("The decline in output, reported by the Federal Reserve on
  Friday, followed seven straight months of increases. The US central bank
  on Wednesday raised interest rates..."); row 9 (Christopher Rupkey, now
  row 8): quotation marks around the statement restored, as in the text;
  now 9 rows. 202 row 1: Document, Source Descriptors "mock-up" -> Named
  Person / Zohran Mamdani, Title "Mayor" ("At that meeting, the mayor
  pitched the president a massive housing project..."); rows 2 and 3
  (Zohran Mamdani) merged -- same source, one paragraph; now 12 rows. 196
  row 7: Unnamed Person, Title "State Department official" -> Named
  Organization / State Department, Title blank (an unnamed official of a
  named organization, as in 201 row 8). 201: no changes.

- **Phase 4, `203-Cancelled_Obamacare.csv`, `204-El_Nino_Emergency.csv`,
  `205-Mortgage_Rates.csv`, 2026-11-16.** 203 row 2: Document, Source
  Descriptors "documents" -> Named Organization / CMS, Source Descriptors
  "health agency" ("...comes as the agency says it will bar some 569
  brokers who it says..." -- the claim is the agency's; the rule-making
  documents stay in the Source Justification); row 10 removed (Unnamed
  Group of People, administration officials -- their estimate is reported
  through the Wall Street Journal, row 9, which is the primary
  attribution); row 11 (now 10): Document, Source Descriptors "report" ->
  Named Organization / US Department of Health and Human Services ("A
  report by the US Department of Health and Human Services..."); now 11
  rows. 205 row 18 (Jake Krimmel): the reporter's lead-in "Home prices are
  already showing signs of softening." taken out of the Sourced Statement.
  204: no changes.

- **Phase 4, `206-Trump_Welcomes_Xi.csv` and
  `207-US_Iran_Unexpected_Problems.csv`, 2026-11-16.** 206: new row 1,
  Unnamed Group of People, Source Descriptors "protesters" ("Outside the
  gate, a group of loud protesters shouted slogans like "China out of Hong
  Kong"..."); new row 12, Named Person / Edgard Kagan, Title "former senior
  White House official for East Asia," Source Justification "now with the
  Center for Strategic and International Studies" ("Edgard Kagan, a former
  senior White House official for East Asia, said the contrasting rhetoric
  demonstrates the different strategy of each leader."); old row 6 removed
  (Unnamed Group of People: "That move left some US allies in Asia and even
  some of Trump's fellow Republicans concerned that..." -- a reported
  concern, not a claim); now 17 rows. 207 row 5 removed (Unnamed Group of
  People, analysts: "With U.S.-Iran talks in New York this week, analysts
  had hoped for some details on when the action may de-escalate." -- a
  reported hope, not a claim); now 11 rows.
  206 row 5: Named Organization, Name "official Chinese news agency" ->
  Document, Source Descriptors "readout," Source Justification "from the
  official Chinese news agency" ("...according to a readout from the
  official Chinese news agency" -- the organization is described, not
  named).

- **Phase 4, `208-Review_Voter_Rolls.csv`, 2026-11-16.** The outlet's own
  reporting credited by name is Named Organization / Reuters (v62 item 10):
  row 10 ("Multiple studies and audits - including a Reuters analysis in
  July - have found...") kept as Document with Source Descriptors
  "studies, audits" (was "studies, audits, analysis"), plus a new row 11,
  Named Organization / Reuters, same Sourced Statement; new row 12, Named
  Organization / Reuters, for "Reuters found this week that states may have
  added more than 30,000 noncitizens..." (same Sourced Statement as the
  experts' row, now 13); old row 12 (now 14), "A Reuters review in July
  found...": Document -> Named Organization / Reuters, Source Descriptors
  blank, and the trailing "The Trump administration has charged at least
  20 more under the statute in the months since." taken out (a later
  update, not from the review). Now 34 rows.

- **Phase 4, `209-FDA_Nominee.csv`, 2026-11-16.** New row 20, Unnamed Group
  of People, Source Descriptors "abortion-rights groups": "If confirmed,
  Overton would inherit an FDA review of mifepristone that abortion-rights
  groups call politically motivated and warn could tighten restrictions on
  a drug used by millions since its US approval 26 years ago." Old rows
  20-23 -> 21-24 (now 24 rows).

- **Phase 4, `210-Fox_Trump_Ad.csv` and `211-Tasia_Fortune.csv`,
  2026-11-16.** 210: new row 2, Unnamed Group of People, Source Descriptors
  "Democrats," same Sourced Statement as row 1 ("...a White House-produced,
  pro-Trump ad that some Democrats said could be illegal government
  propaganda..." -- two sources joined in one sentence, one row each); old
  row 11 removed (an exact duplicate of row 10, the ad's closing tribute);
  old row 17 (now 17): closing quotation mark restored ("...golden age of
  America.”"); now 18 rows. 211 row 1 ("Investigators believe Tasia
  Fortune was already dead..., a detective testified Friday."): Unnamed
  Group of People, Title "investigators" -> Named Person / Samuel Dukes,
  Title "Jackson police Det.," Source Justification as on his other rows
  (the detective who testified, named in the next paragraph; the
  investigators' belief is part of his testimony).

- **Phase 4 Step 1 (mechanical fixes), original GT-II batch, 2026-11-18.**
  Settled rules only; changes approved as one batch table (Part A).
  Sourced Statements made to match the article text exactly -- curly vs
  straight quote marks and apostrophes as the text has them (and a double
  space -> single in `152` rows 12, 13): `65` rows 1, 2, 3, 5, 8, 9, 10,
  13, 16, 18, 20, 22, 23, 24, 26, 27, 29, 30, 31, 32, 33, 36, 38, 40, 41,
  43, 45, 46, 47; `66` rows 5, 6, 7, 8, 9, 11; `70` rows 2, 3, 5, 6, 9;
  `74-San_Ramon_Pride.csv` rows 2, 4, 6, 9; `78` rows 1, 4, 8, 10, 11, 13,
  14, 15, 16, 19; `101` rows 3, 4, 6, 7, 10, 11, 13, 17, 18, 21, 23, 24,
  25, 27, 28, 30; `102` rows 1, 5, 6, 8, 10, 13, 15, 16, 17, 21, 22, 23,
  25; `103` rows 1, 2, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 22,
  23, 24, 26, 27, 28, 29, 31; `104` rows 2, 3, 4, 5, 6, 8, 9, 10, 11, 14,
  15, 16, 19; `105` rows 1, 3, 4, 6, 8, 10, 12, 15, 19, 20, 21, 22, 23,
  24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 38, 40, 41, 42, 44, 45, 46, 47;
  `107` rows 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17;
  `108-SZ_YVA.csv` rows 1, 2, 3, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18,
  19, 20, 21; `109-SZ_Waymo.csv` rows 1, 2, 4, 7, 8, 10, 12, 14, 15; `110`
  rows 2, 3, 4, 7, 8, 9; `111` rows 1, 3, 5, 6; `112-SZ_Hantavirus.csv`
  rows 1, 2, 4, 7, 9, 12; `122-SZ_SJ9_Climate_Policy.csv` rows 7, 14, 18,
  19; `124` rows 4, 8; `152` rows 12, 13; `159` rows 8, 9. Title of
  Source: words the article doesn't capitalize lowercased (capitals the
  reporter uses, including organization names, kept), and a leading
  article dropped: `51` rows 9, 10, 11 "Astronaut" -> "astronaut";
  `61-Atlantic_Current.csv` row 1 "Scientists" -> "scientists"; `70` row 8
  "Federal scientists" -> "federal scientists"; `78` rows 9, 10
  "Transportation Engineer at Government Agency" -> "transportation
  engineer at government agency"; `78` rows 11, 12, 13, 14, 16 "Building
  engineer at the San Francisco International Airport" -> "building
  engineer at the San Francisco International Airport";
  `94-SZ_Met_Gala.csv` row 6 "the museum’s director and CEO" -> "museum’s
  director and CEO"; `94-SZ_Met_Gala.csv` rows 7, 8, 9, 10, 11 "the
  curator in charge of the Costume Institute" -> "curator in charge of the
  Costume Institute"; `101` rows 2, 3, 4 "Group village headman" -> "group
  village headman"; `101` row 30 "Group village headman Nkhulambe" ->
  "group village headman Nkhulambe"; `105` rows 15, 17 "Spokesperson,
  Texas Department of Criminal Justice" -> "spokesperson, Texas Department
  of Criminal Justice"; `105` row 19 "Founder of Worth Rises" -> "founder
  of Worth Rises"; `108-SZ_YVA.csv` rows 6, 16, 17, 19 "YVA Organizer" ->
  "YVA organizer"; `108-SZ_YVA.csv` row 14 "YVA Cofounder, YVA Program
  director" -> "YVA cofounder, YVA program director"; `109-SZ_Waymo.csv`
  rows 1, 2 "Spokesperson" -> "spokesperson"; `111` row 1 "Spokesperson,
  National Homelessness Law Center" -> "spokesperson, National
  Homelessness Law Center"; `111` row 3 "Spokesperson for Gov. Gavin
  Newsom" -> "spokesperson for Gov. Gavin Newsom"; `112-SZ_Hantavirus.csv`
  rows 7, 9, 12 "Incident Manager for the CDC's hantavirus response" ->
  "incident manager for the CDC's hantavirus response";
  `122-SZ_SJ9_Climate_Policy.csv` rows 19, 20 "Event Moderator" -> "event
  moderator"; `154` rows 3, 4, 17, 18 "Senior fellow at the
  Washington-based Foundation for Defense of Democracies" -> "senior
  fellow at the Washington-based Foundation for Defense of Democracies".
  Row numbers: `74-San_Ramon_Pride.csv` ("1.0, 2.0..." -> "1, 2...");
  `104-Irish_Centre.csv` ("1.0, 2.0..." -> "1, 2..."); `109-SZ_Waymo.csv`
  ("1.0, 2.0..." -> "1, 2..."); `111-CA_Homelessness.csv` (duplicate row
  number 1 -> 1, 2, 3...).
- **Phase 4 Step 1 Part B (text fidelity), original GT-II batch,
  2026-11-19.** Approved item by item with full before-rows. Sourced
  Statements only; row numbers are before the splits unless shown with an
  arrow. Fragment -> the full sentence(s) from the article, lead-in included
  (v61 Note 12a): `61-Atlantic_Current.csv` rows 2, 3, 4; `78` rows 5, 18. A
  row crossing a paragraph break -> one row per paragraph, other fields
  copied (Note 12a; breaks checked against the live pages): `65` row 21 ->
  rows 21, 22; `66` row 10 -> rows 10, 11; `70` row 4 -> rows 4, 5; `101`
  row 1 -> rows 1, 2, row 12 -> rows 13, 14, row 14 -> rows 16, 17, row 15
  -> rows 18, 19, row 29 -> rows 33, 34, row 31 -> rows 36, 37; `103` row 3
  -> rows 3, 4, row 5 -> rows 6, 7; `105` row 7 -> rows 7, 8, row 17 -> rows
  18, 19; `112-SZ_Hantavirus.csv` row 3 -> rows 3, 4. Wording made to match
  the article text (typos, dropped/changed words, dashes, ellipses, quote
  marks): `65` rows 4, 6, 39; `122-SZ_SJ9_Climate_Policy.csv` rows 11, 16,
  20; `124` row 1; `135` row 3; `159` row 5. Rows renumbered after the
  splits: `65` (47 -> 48 rows); `66` (11 -> 12 rows); `70` (9 -> 10 rows);
  `101` (31 -> 37 rows); `103` (33 -> 35 rows); `105` (47 -> 49 rows);
  `112-SZ_Hantavirus.csv` (12 -> 13 rows). Same-source merges deferred until
  after 4b (Step 3).
- **Story-text fix and matching GT fix, 2026-11-19 (Phase 4 Step 1 Part B).**
  The article text for `78` had "return-to-workorder"; corrected to
  "return-to-work order" as on the live page (logged in
  `extracted_articles_boilerplate/REFLOW_2026-11.md`). `78` row 9 Sourced
  Statement: straight apostrophes -> curly as the text has them
  ("state's" -> "state’s", "she's" -> "she’s").
- **Story-text re-flow and matching GT fix, 2026-11-19 (Phase 4 Step 1 Part B).**
  The `110` article text (one run-on line from the web fetch) was re-flowed to
  the PDF's 15 paragraphs; headline/byline, promo and footer removed; photo
  captions moved to a `Caption:` header line (see
  `extracted_articles_boilerplate/REFLOW_2026-11.md`). `110` row 10 Sourced
  Statement: straight quote marks -> curly and the space before the comma
  removed, as the text has it ("...ever 😂 ," -> "...ever 😂,”").
- **Phase 4 Step 1 (text fidelity) on story 52/55 SZ files, 2026-11-19.**
  Approved with full before-rows. Sourced Statements made to match the
  article text: outer quote marks the text does not have removed, and the
  text's double quotes restored where the row had turned them into single
  quotes -- `52-East_Bay_Voters.csv` rows 11, 12, 13;
  `55-Chicago_Pentagon.csv` rows 6, 10, 11, 12, 13, 14, 15, 16, 18, 22;
  `55-Chicago_Pentagon.csv` row 21 final period added.
  `55-Chicago_Pentagon.csv` row 5 (Eliza Glezer) crossed a paragraph break
  and had mixed quote marks -> two rows, one per paragraph, with the text's
  punctuation (other fields copied); rows after it renumbered (24 -> 25
  rows; old rows 6-24 are now 7-25). Row numbers above are before the split.
- **Phase 4 Step 2 batch 1 (4a recall, 4b precision, SJ, Title), original
  GT-II batch, 2026-11-19.** Approved item by item with full before-rows.
  `155`: 4 rows added -- State prosecutors "have objected to the defense's
  claim" (Unnamed Group of People, Title "state prosecutors"); Jamie
  McDonald, U.S. attorney for the Southern District of New York, "said his
  office would seek the maximum sentence"; Mangione "admitted ambushing
  Thompson... He also told the court..." (plea hearing); authorities on the
  arrest ("Mangione was arrested five days later... authorities said.");
  rows put in text order (11 rows; old 1, 2, 4, 3, 5, 6, 7 are now 1, 2, 3,
  7, 8, 10, 11; new rows 4, 5, 6, 9). `95` rows 3, 4, 5 (the CSIS report):
  Document -> Named Organization / Center for Strategic and International
  Studies, Source Descriptors "report" -> blank (v61 Minor 22: the reporter
  names the report's producer). `111`: row added for "The Trump
  administration wants to end housing first, which it says isn’t working..."
  (Named Organization / Trump administration; now row 6, old rows 6-8 ->
  7-9); Title of Source as written: row 2 "spokesperson, National
  Homelessness Law Center" -> "spokesperson for the National Homelessness
  Law Center", row 5 "Senior California Policy Fellow, National Alliance to
  End Homelessness" -> "senior California policy fellow for the National
  Alliance to End Homelessness". `61-Atlantic_Current.csv`: Name of Source
  "Professor Stefan Rahmstorm" -> "Stefan Rahmstorf" (rows 3, 4, 5, 7, 8)
  and "Dr. Valentin Portmann" -> "Valentin Portmann" (row 2), as the text
  spells them; row 2 SJ "led the new research on Atlantic Current" -> "led
  the new research" (the text's words); row 7 Sourced Statement trimmed to
  "This makes the work “very credible”, said Rahmstorf." -- the two
  sentences before it are the reporter's (Note 12a).
- **Phase 4 Step 2 batch 1, held item, 2026-11-19.** `111` row 8
  ("...according to the most recent HUD report."): Document -> Named
  Organization / Department of Housing and Urban Development (as the article
  names it), Source Descriptors "report" -> blank -- a HUD report is the
  department's own product (Minor 22).
- **Phase 4 Step 2 batch 2b, original GT-II batch, 2026-11-19.** Approved
  item by item with full before-rows. `70`: 3 rows added -- "Researchers
  estimate it has reduced the risk of death by ship strike for whales in the
  area by 40%." (Unnamed Group of People, Title "researchers"); Sam
  Liccardo's bill as a Document (SD "legislation"): "His Save Willy Act
  would establish a “whale desk”..." and "The legislation introduced
  Wednesday by Sam Liccardo would seek to help whales..." (user: what a bill
  says is the bill's content). Now 13 rows: old 1-2 -> 2-3, 3-9 -> 5-11, 10
  -> 13; new rows 1, 4, 12. Not changed: the Sonoma State University /
  Marine Mammal Center research row stays Document, and the NOAA scientists
  row stays Unnamed Group of People (user: scientists at an agency do not
  speak for it, item 41). `51`: old row 11 ("“That,” says Aschbacher, “would
  create a very different life on planet Earth.”") Name "Alexander Gerst" ->
  "Josef Aschbacher", Title "astronaut" -> "Director General of European
  Space Agency (ESA)" -- the speaker is Aschbacher; old row 7 final period
  added; Barber rows (old 5, 6, 7) Name "Dr. Simeon Barber" -> "Simeon
  Barber", Title "Space Scientist from Open Univeristy" -> "Dr., space
  scientist from the Open University" (the "Dr., <position>" form); row
  added for China's "stated aim of landing an astronaut on the Moon by
  around 2030" (Named Organization / China; now row 8, old rows 8-11 ->
  9-12).
- **Phase 4 Step 2 batch 2c (dual pair; technical fixes only), original
  GT-II batch, 2026-11-19.** `94-SZ_Met_Gala.csv`: Bolton row 7 extended
  with the lead-in sentence of his paragraph ("Andrew Bolton, the curator in
  charge of the Costume Institute, weighed in on why...", Note 12a); old row
  4 ("From that homemade tennis skirt... Williams has come to...") removed
  -- the reporter's characterization; old row 9 trimmed to "Or, as Bolton
  noted: “Even in its absence...”" -- the sentence before it ("In that way,
  the clothed human form is a reflection...") is the reporter's, between
  Bolton's statements (item 55). Now 10 rows (old 5-11 -> 4-10). Row 1
  ("Venus Williams remembers...") kept (user).
- **Phase 4 Step 2 batch 2b follow-up, 2026-11-19.** `51` rows 10, 11 ("ESA
  astronaut Alexander Gerst once told Aschbacher...", "Gerst told the ESA
  boss that he wishes...") Named Person "Alexander Gerst" / Title
  "astronaut" -> "Josef Aschbacher" / "Director General of European Space
  Agency (ESA)" -- relayed speech: the reporter heard it from Aschbacher in
  an interview (dev doc item 73).
- **Minor 22 Document-name fixes after a re-read of the v61/v62 rules,
  2026-11-19.** A Document's Name of Source is the document's own title only
  if the reporter gives it. `155` row 1 ("...court records show."): Name
  "Court Records" -> blank, Source Descriptors -> "court records". `70` row
  1 ("His Save Willy Act would establish..."): Name blank -> "Save Willy
  Act" (the reporter gives the bill's title). `61-Atlantic_Current.csv` row
  6 (the Science Advances research): Name "Science Advances" -> blank (the
  journal published affiliated researchers' work; it is not the research's
  title or its producer); Source Justification -> "published in the journal
  Science Advances; explored four different ways of using real-world
  observations to assess the models" (user: publication signals the
  research's scientific standing).
- **Phase 4 Step 2 batch 3a (dual pair 52), original GT-II batch,
  2026-11-19.** `52-East_Bay_Voters.csv`: Source Descriptors "Livermore
  resident" (Julie Valentine) and "Hayward resident" (Melissa Estepa) ->
  blank (item 70: the text says "of Livermore", "of Hayward"; a Name of
  Place is not a descriptor). Title of Source for elected officials, as
  written: Eric Swalwell -> "Rep.", Ro Khanna -> "Rep. D-Santa Clara", Sam
  Liccardo -> "Rep. D-San José", Jared Huffman -> "Rep. D-California".
  Source Justification on the public-letter rows "More than 50 former
  staffers of Swalwell issued in a public letter" -> "...issued a public
  letter" (the text's words). 4 rows added: Julie Valentine "questioned the
  timing of the allegations" (her lead-in paragraph; the same sentence is
  still her quote row's SJ -- for the item 58 sweep); Swalwell "announced
  Sunday that he was dropping out..."; Named Organization / San Francisco
  Chronicle for the accuser's allegations from its published interviews
  (relayed through an outlet, item 73); Named Organization / CNN ("reported
  that three additional women..."). Now 17 rows: new rows 1, 5, 6, 7; old
  1-3 -> 2-4, 4-13 -> 8-17.
- **Phase 4 Step 2 batch 3b (66, 152), original GT-II batch, 2026-11-19.**
  Approved item by item with full before-rows. `66` (now 17 rows, put in
  text order): 4 rows added -- Musk "alleges that in the early days of
  OpenAI..." (Named Person, speaking through his lawsuit); "(OpenAI alleges
  that Musk knew back in 2017...)" (Named Organization / OpenAI); Musk "is
  asking the court for a number of different remedies..." (Named Person);
  "Former OpenAI researchers and AI safety nonprofits... say they
  believe..." (Unnamed Group of People, Title "former OpenAI researchers";
  the unnamed nonprofits go to item 75); old row 5 ("...legal experts and
  people close to the case say it’s unlikely.") split into one row per group
  (Title "legal experts"; SD "people close to the case"), SJ "people close
  to the case" -> blank; old row 10 (Gómez) trimmed to start at "Luís
  Calderón Gómez..." -- "Almost certainly not, but..." is the reporter's;
  Brockman Title "President and cofounder of OpenAI" -> "OpenAI’s President
  and cofounder" (as written); Calvin Title "general counsel for Encode, a
  nonprofit that has supported several AI safety bills and objected to the
  OpenAI restructuring" -> "general counsel for Encode", the clause -> his
  SJ. Rows now: legal experts 1, people close to the case 2, Musk 3, OpenAI
  4, defendants (old 1-4) 5-8, Musk 9, OpenAI spokesperson (old 9) 10,
  former researchers 11, Hilton (old 6) 12, Calvin (old 7) 13, Horwitz (old
  8) 14, Gómez (old 10, 11) 15, 16, Hilton (old 12) 17. `152` (now 17 rows):
  statements the reporter took from other outlets' interviews credited to
  the outlet (dev doc item 73): old row 3 Larkin ("...Larkin said during an
  Aug. 11 interview with Florida Today...") -> Named Organization / Florida
  Today (Larkin's SD and SJ cleared), and its last sentence "Larkin has been
  endorsed by the DSA." trimmed (reporter's); old row 4 Moskowitz ("In South
  Florida, that means...") -> Named Organization / Florida Politics, SD
  "online news site", Title and SJ cleared; row added for "Moskowitz, when
  asked by Florida Politics, an online news site, ... said they need to stop
  arguing..." (Florida Politics). Rows added: Tara Newsom "Newsom of St.
  Petersburg agrees."; Document / SD "polls" ("...polls indicate GOP voters
  in the Aug. 18 primary will likely choose him..."); "A federal judge ruled
  2 Live Crew's album... obscene in 1990." (Unnamed Person, Title "federal
  judge"). Kyle Kondik Title: trailing period removed (old rows 5-7). Rows
  now: new 4, 6, 10, 12; old 4 -> 5, 5-7 -> 7-9, 8 -> 11, 9-13 -> 13-17.
- **Phase 4 Step 2 batch 3b, held item, 2026-11-19.** `66`: 2 Document rows
  added for what the lawsuit itself says (v61 Minor 22: the reporter credits
  the suit), Source Descriptors "lawsuit": "Musk’s suit essentially accuses
  OpenAI of straying from its founding nonprofit mission..." (now row 3) and
  "The second core claim is fraud... The third claim is unjust enrichment,
  which argues..." (now row 6). Now 19 rows; the rows after each new row
  move down by one or two.
- **Phase 4 Step 2 batch 3c (dual pairs; technical fixes), original GT-II
  batch, 2026-11-19.** `112-SZ_Hantavirus.csv`: row added for "Kornfeld said
  they all seem to be planning to stay in quarantine for the full
  recommended 42 days..." (now row 5, old 5-13 -> 6-14); old row 7 (p10,
  CDC) Sourced Statement restored to the text's exact wording ("officials
  with the CDC said", "the officials said", curly quotes -- the row had
  "officials with the US Centers for Disease Control and Prevention" and
  "the CDC officials"); old row 10 (Fitter) trimmed to "But the “CDC
  currently recommends testing only for people with symptoms,” Fitter said."
  (the sentence before it is the reporter's); old rows 11, 12 "...WHO and
  CDC say." -> "...WHO and the CDC say." (as the text); Title of Source:
  Fitter "incident manager for the CDC's hantavirus response" -> "Dr.,
  incident manager for the CDC’s hantavirus response", Kornfeld "Oregon
  oncologist" -> "Dr., Oregon oncologist". `109-SZ_Waymo.csv`: row added for
  "Some residents have expressed concerns..." (Unnamed Group of People, SD
  "residents"; now row 8, old 8-15 -> 9-16); row 5 Title "elected leaders in
  West Valley" -> "elected leaders in the West Valley".
- **Phase 4 Step 2 batch 3c, held items, 2026-11-19.** `109-SZ_Waymo.csv`:
  Rob Moore row 7 (p8) -> "Moore said Waymo previously told him... Still, he
  said he sees potential safety benefits in autonomous vehicle technology."
  -- his two sentences in the paragraph, joined; the residents' sentence
  between them is its own row (Note 12a). (The residents' sentence was taken
  out of his row.)
- **Phase 4 Step 2 batch 3d (156), original GT-II batch, 2026-11-19.**
  Approved with full before-rows. `156` (now 20 rows): old row 5
  ("...according to Ipsos polling.") Name "Reuters/Ipsos" -> "Ipsos" (as
  credited there); old row 7 (Newsom) extended with the first sentence of
  its paragraph ("The move was widely criticized at the time, including by
  California Governor Gavin Newsom, who sued Trump over the
  federalization."). Rows added: White House "has consistently rejected the
  idea..." (Named Organization); Trump "promised to crack down... what he
  described as the largest deportation operation..." (Named Person,
  President); "Despite the drop in support, Trump and his allies have
  defended the crackdown..." -- one row each (Named Person / Donald Trump;
  Unnamed Group of People, SD "allies"); "...with 38 percent of respondents
  saying that he was doing a good job on immigration." (Named Organization /
  Reuters/Ipsos, the poll series of the surrounding paragraphs; SJ as on the
  other Reuters/Ipsos rows). New rows 3, 7, 10, 11, 12; old 3-5 -> 4-6, 6-7
  -> 8-9, 8-15 -> 13-20.
- **Phase 4 Step 2 batch 4a (107), original GT-II batch, 2026-11-20.**
  Approved item by item with full before-rows. `107` (now 16 rows, in text
  order): old row 1 removed (the headline, "Jackson’s Running Water Is
  ‘High-Pressure Poison,’ Residents Say" -- body-only rule); old row 16
  removed (Kynndle Robinson "doesn’t understand why..., but as far as she
  knows, government officials don’t care..." -- a reported belief, v62 item
  21); row added for Kansas Gray "Growing up on the city’s south side,
  Kansas Gray, 37, remembers having brown water most times. He recalls his
  mom telling him that water from the faucet was poisonous." (SD "community
  activist"). Residents row SD "Jackson residents" -> "residents" (item 70).
  Leaders row Title "Jackson leaders" -> "leaders"; Tate Reeves Title
  "Republican Governor" -> "Republican Gov." (as written). Titles not in the
  article blanked: Tim Finch "outreach manager for Operation Good", Kansas
  Gray "owner of South Side Barber & Beauty Shop"; Maisie Brown Title
  "organizer with the Mississippi Student Water Crisis Advocacy Team" ->
  blank, SD "organizer" (non-credentialing). Finch SJ quote marks curled as
  in the text; Gray SJ -> his two p11 sentences only (dropped "Kansas Gray,
  owner of South Side Barber & Beauty Shop; a community activist", not in
  the text). Finch rows trimmed to his own sentences (item 55): old row 6
  drops "Finch, who now works with youth..., is conflicted about..."; old
  row 7 drops "Last week, he worked with a small team of youth
  volunteers..." between his statements. Row numbers now: leaders 1, Finch
  2, Reeves 3, residents 4, Finch 5-7, Gray 8-11 (8 new), James 12-14,
  Robinson 15, Brown 16.
- **Phase 4 Step 2 batch 4a, held item, 2026-11-20.** `107`: row added for
  Kynndle Robinson, "Living with discolored water and frequent boil water
  notices is “everyday life.”" (p23, implied attribution with her quoted
  words; SD "youth participant", SJ as on her other row); now row 15, her
  quote row 15 -> 16, Brown 16 -> 17.
- **Phase 4 Step 2 batch 4b (104), original GT-II batch, 2026-11-20.** `104`
  (now 18 rows): old row 1 removed (the headline, "“No Frick’n way”:
  Neighbors oppose transforming former Irish Centre into high-end
  apartments" -- body-only rule); Ray Baum row (old 12) trimmed to end at
  "...so requested a delay." -- the sentence after it ("Coalition members
  met earlier this week and agreed to support two of the three variance
  requests...") is the reporter's (v62 item 9); David Vatz rows (old 13-16)
  Source Descriptors blank -> "Squirrel Hill resident" (the text: "David
  Vatz, a Squirrel Hill resident and founder of Pro-Housing Pittsburgh").
  All rows move up by one.
- **Follow-up, 2026-11-21.** `66` row 2 ("...legal experts and people close
  to the case say it’s unlikely.", the people-close-to-the-case row): Source
  Descriptors "people close to the case" -> blank, Source Justification ->
  "people close to the case" -- the phrase names their relation to the
  story, not a category (v62 item 23).
- **Phase 4 Step 2 batch 4c (dual pair 122; technical fixes), original GT-II
  batch, 2026-11-21.** `122-SZ_SJ9_Climate_Policy.csv` (now 26 rows, put in
  text order): four Named Person rows added for "But Altwer and her fellow
  candidates also voiced concern..." (Hughes, Hennessy, Chester, Ator -- v61
  Minor 23; Altwer already had a row); row 1 Scott Hughes Title blank ->
  "chief of staff for terming out incumbent Vice Mayor Pam Foley" (as on his
  other rows); old row 10 split -- "But more recently, leaders have
  acknowledged the city is off course." (leaders) and "San Jose’s most
  recent greenhouse gas inventory found emissions increased slightly between
  2021 and 2023." (new row, Named Organization / San Jose, Minor 22); old
  row 15 ("Candidates also weighed in on the city’s plans... This effort has
  faced pushback from residents who are concerned...") -> the residents'
  sentence only, Source Descriptors "candidates" -> "residents" (user: the
  candidates' "weighed in" carries no content); Anika Khanna: Title "event
  moderator" -> "event’s moderator", Source Descriptors "member" ->
  "sophomore, member", Source Justification -> "Anika Khanna, the event’s
  moderator and a sophomore at Lynbrook High School" (the text's phrase);
  row added for "Still, Khanna, who is also a Silicon Valley Youth Climate
  Action member, said she is glad the forum... put climate concerns firmly
  on the candidates’ radars." (p22). Rows now: forum (all candidates) 1-5,
  Altwer 6-8, fellow candidates 9-12, leaders 13, San Jose 14, "most
  District 9 candidates" 15, Ator 16, Altwer 17, Hughes 18-19, residents 20,
  Chester 21, Hennessy 22-23, Khanna 24-26.
- **Phase 4 Step 2 batch 4d (78), original GT-II batch, 2026-11-22.**
  Approved item by item with full before-rows. The article text had "BART
  lists15 completed TODs" -- corrected to "BART lists 15" (logged in
  `extracted_articles_boilerplate/REFLOW_2026-11.md`). `78` (now 21 rows):
  old row 16 ("“It would be really difficult for my commute,” he said.")
  Tanner Firlotte -> Nathan Arza (Title and SJ as on his other row) -- the
  quote ends the Arza section; old row 13 (Firlotte) extended with the
  second sentence of its paragraph ("Riders losing their stations wouldn’t
  be the only ones pushed back behind the wheel, he worries: If fares rise
  by 30%, as BART has warned, even commuters... could start driving...");
  old row 4 ("BART’s latest “alternative service plan” doesn’t name which
  stations would shutter, but agency staff have presented the board with a
  list...") Document / Name "alternative service plan" -> Named Organization
  / BART (v61 Minor 22, Note 8); old row 6 ("A 2018 law goes further...")
  Name "2018 law" -> blank, Source Descriptors -> "law" (not the law's
  title); Melissa Lopez Title "transportation engineer at government agency"
  -> "transportation engineer" (as written). Rows added (Named Organization
  / BART): "But, as BART warns of potential service cuts in January and
  closing 15 stations as soon as next July, she stands to give up more than
  convenience..." (now row 2) and "BART lists 15 completed TODs accounting
  for some 4,200 residential units..." (now row 8). Old rows 2-6 -> 3-7,
  7-19 -> 9-21.
- **Phase 4 Step 2 batch 4e (135, 159), original GT-II batch, 2026-11-22.**
  Approved item by item with full before-rows. `135` (now 19 rows): Mikko
  Paasi rows 3, 5, 7 trimmed to start at his attributed sentence -- the
  unattributed opening sentences ("The miner was scuba diving for the first
  time...", "The round trip to where the group is trapped takes a trained
  team of divers about five hours.", "The rescuers carried five oxygen
  tanks...") could be the reporter's narration (user); old row 13 removed
  ("The rescue team had hoped to be able to pump the water out of the
  cave..." -- a reported hope, v62 item 21); old rows 14-20 -> 13-19. Row 1
  (unnamed diver) kept as Unnamed Person (user). `159` (now 23 rows): the
  article text had two all-caps related-story headlines inside the body
  ("TRUMP WARNS IRAN FIGHTING ISN'T OVER..." and "GULF COUNTRIES STRONGLY
  CONDEMN IRAN'S DRONE ATTACK ON BAHRAIN...") -- removed from the text
  (REFLOW_2026-11.md); no GT row used them. Rows added: Named Organization /
  United States for "Iran insists it must retain authority..., while the
  United States has demanded unrestricted commercial navigation." (two
  sources in one sentence, v62 item 18; now row 3); Named Organization /
  Iran and / United States for "Instead, the framework broke down as Iran
  and the United States accused each other of violating its terms and
  resumed military action." (now rows 11, 12). Old row 16 ("...as reports
  indicate the war has strained U.S. munitions inventories.") Name "Reports"
  -> blank, Source Descriptors -> "reports" (v61 Minor 22: a Document's Name
  is its own title only). Old rows 3-9 -> 4-10, 10-20 -> 13-23.
- **Phase 4 Step 2 batch 5a (dual pair 108; technical fixes), original GT-II
  batch, 2026-11-22.** `108-SZ_YVA.csv`: Title of Source as written: Aniya
  Butler "YVA organizer" -> "YVA’s hip-hop and climate justice initiative
  coordinator" with Source Descriptors "organizer" (p8 "YVA organizer Aniya
  Butler", p29 "Aniya is YVA’s hip-hop and climate justice initiative
  coordinator"); Hannah Estrada "YVA education and organizing coordinator"
  -> "YVA’s education and organizing coordinator"; Sim Bilal Title -> blank,
  Source Descriptors "organizer" (non-credentialing; "an organizer with
  Youth Climate Strike in Los Angeles"). Youth vs. Apocalypse's p7
  suggestion row -> "One suggestion: Use this upcoming Earth Day, April 22,
  to plan activities at your school. YVA suggests young people offer to do
  presentations..." (sentences 2-3 of the paragraph). (The reporter's
  first-person question "I ask YVA leaders what lessons they would share..."
  taken out, item 55; Bilal Title was "Youth Climate Strike Organizer".)
  Cory Jong row 12 trimmed to start at "She cared about energy efficiency
  and recycling in college..." -- "Cory Jong teaches eighth-grade ethnic
  studies at Urban Promise Academy in Oakland, CA." is the reporter's
  introduction (item 55).
- **Phase 4 Step 2 batch 5d (124, 160), original GT-II batch, 2026-11-22.**
  Approved item by item with full before-rows. `124` (now 21 rows): the
  article text had station promo lines in the body ("Stream NBC4 newscasts
  for free right here, right now.", "Transportation", "Reporter Adam Tuss
  and the News4 team are covering you down on the roads and in transit.") --
  removed from the text (REFLOW_2026-11.md); no GT row used them. Row 8
  ("...,” the document said.") Source Descriptors "memo" -> "order" (the
  document is the CDC order of the paragraph before, not the DHS memo);
  Maria Reppas Title "Communications Director for VDH" -> "communications
  director for VDH" (as written); old row 16 removed ("From the few
  outbreaks health experts have seen, Bundibugyo might be slightly less
  deadly..." -- the reporter's claim, no speech act). `160` (now 29 rows):
  old row 10 ("Putin’s arrival sparked an outcry from Japanese
  officials...") Named Organization / Japan -> Unnamed Group of People,
  Title "Japanese officials" (a nationality adjective is not a name, as
  123's "Israeli officials"). Rows added: Named Organization / China for
  "China went on to conduct what it called a joint maritime drill... though
  Jakarta downplayed..." (two sources in one sentence); Named Organization /
  White House for "The White House has repeatedly reaffirmed its support for
  Taiwan, though Trump stated..."; Named Person / Donald Trump, President,
  for "...as well as Trump’s announcement Sunday that he would order the
  Pentagon to scale back joint military exercises with South Korea.";
  Document / SD "reports" and Unnamed Group of People / SD "sailors'
  families" for "Reports first emerged earlier this month that a new
  aircraft carrier would be sent..., amid concerns raised by sailors'
  families over deteriorating conditions..." (one row per source).
- **Title of Source format: titles separated by "; ", "Dr." as its own title
  (v62 item 24), 2026-11-22.** `124` Celine Gounder (rows 12, 15, 16, 17)
  "infectious disease specialist and epidemiologist, editor-at-large for
  public health at KFF Health News" -> "Dr.; infectious disease specialist
  and epidemiologist; editor-at-large for public health at KFF Health News";
  Vasee Moorthy (row 13) -> "Dr.; special adviser in the office of the WHO
  chief scientist". "Dr., <position>" -> "Dr.; <position>": `209` Heidi
  Overton (rows 1, 7, 8, 11, 12, 13, 15, 17, 18, 22, 24); `51` Simeon Barber
  (rows 5, 6, 7); `112-SZ_Hantavirus.csv` Stephen Kornfeld (rows 1-5) and
  David Fitter (rows 9, 11, 14).
- **Phase 4 Step 2 batch 5e (dual pair 55; technical fixes), original GT-II
  batch, 2026-11-22.** `55-Chicago_Pentagon.csv` (now 30 rows): old row 8
  ("The state’s quantum bet is also about jobs, and supporters say the
  federal subsidies are necessary. Residents said the numbers are small —
  and lopsided.") split one row per source -- residents keep the second
  sentence; new row for the first (Unnamed Group of People, SD
  "supporters"). Rows added: "Since then, they have committed hundreds of
  millions in taxpayer money... to help build what they say will be the
  country’s first large-scale quantum computer." (Unnamed Group of People,
  Title "state and city officials"); "The petition met the signature
  threshold, but in January the Chicago Board of Elections voted to strike
  it from the March ballot, agreeing with attorneys who argued..." (Named
  Organization / Chicago Board of Elections -- its own ruling; the
  attorneys' argument is nested). Old row 13 split -- "Scholars note that
  quantum tools could theoretically help..." (Title "scholars, experts" ->
  "scholars") and new row "Simply giving communities access to advanced
  computing does not deliver justice, experts said." (Title "experts"). Row
  added for Andrew Torrence: "Torrence also spoke to what he said were
  larger politics at play." (as the AV copy).
- **Phase 4 Step 2 batch 5e, held item 5e.3, 2026-11-22.**
  `55-Chicago_Pentagon.csv`: Title of Source as written -- JB Pritzker
  "Governor" -> "Gov." ("Gov. JB Pritzker"); Brandon Johnson "Mayor" ->
  "Chicago Mayor" (first formal introduction, p10: "The office of Chicago
  Mayor Brandon Johnson..."; v61 elected-office rule, constituency as
  written); Eliza Glezer -> "assistant deputy director for the Illinois
  Department of Commerce and Economic Opportunity; state official" (p9 and
  p18; titles joined with "; ", v62 item 24).
- **Phase 4 Step 2 batch 6a (102), original GT-II batch, 2026-11-22.**
  `102`: Title of Source "co-founder of the Sherwood Heights Tenants
  Association" (Deborah Patrick rows 1, 2, 8; Roger Turner rows 3-6) and
  "co-founder of the Tenants Association of New Center Plaza and Marlenor"
  (Steven Rimmer rows 9-11, 14, 16, 17, 19) -> blank: the article says they
  "formed" the associations and never uses "co-founder"; their founding
  stays in their Source Justification.
- **Phase 4 Step 2 batch 6a (102), continued, 2026-11-22.** Source
  Justification made the text's words: "Patrick and Turner formed the
  Sherwood Heights Tenants Association to make tenant's voices heard that
  last June." -> "...to do just that last June." (Deborah Patrick and Roger
  Turner rows); "Rimmer helped form the citywide Detroit Tenants Association
  DTA last August." -> "...(DTA) last August." (Steven Rimmer, the "DTA’s
  first priority..." row). Row added: "The association has a list of demands
  around fixing up the buildings, especially the plumbing and electrical
  systems. Members also won’t agree to any rent increases until the biggest
  issues are resolved." (Named Organization / Sherwood Heights Tenants
  Association; now row 6, old rows 6-25 -> 7-26).
- **Phase 4 Step 2 batch 6b (154), original GT-II batch, 2026-11-22.**
  Approved item by item with full before-rows. `154` (now 32 rows): rows
  added -- Donald Trump, President: "WASHINGTON – President Donald Trump is
  putting another U.S. ally he says was insufficiently supportive of the
  Iran war on notice..." (p2), "Over the past several days, he has lobbed
  similar allegations at South Korea and made public overtures to Kim Jong
  Un..." (p5), "Then, on Aug. 16, Trump said the United States was dialing
  down military exercises..." (p18), "Asked about Kim in the Oval Office on
  Aug. 17, Trump brought up... his earlier allegation..." (p27); Named
  Person / Lee Jae Myung, "South Korean President", "Over the weekend, South
  Korean President Lee Jae Myung called for talks to end the war."; Named
  Organization / South Korea, "Last year, South Korea also committed to
  spending $25 billion... and pledged to spend $33 billion..." (a specific
  claim, user). Center for International Policy study row: Document / SD
  "study" -> Named Organization / Center for International Policy (Minor
  22). Name "Kim Min-Seok" -> "Kim Min-seok". Title of Source as written:
  Sydney Seiler -> "senior adviser with the Center for Strategic and
  International Studies focused on Korea"; Alexander Gray -> "former chief
  of staff on the National Security Council" (p33); Mark Montgomery ->
  "Retired Rear Adm.; senior fellow at the Washington-based Foundation for
  Defense of Democracies" (rank as a title, v62 item 24), SJ "Retired Rear
  Adm." -> blank. New rows 1, 4, 13, 15, 23, 32; old rows 1-2 -> 2-3, 3-10
  -> 5-12, 11 -> 14, 12-18 -> 16-22, 19-26 -> 24-31.
- **Phase 4 Step 2 batch 6b, held item, 2026-11-22.** `154` row 3 (Trump,
  p4) extended with the first sentence of its paragraph: "South Korea is the
  latest in a string of close U.S. allies Trump has threatened to penalize
  with reduced military support since the United States and Israel went to
  war with Iran earlier this year." (a threat credited to him; Note 12a).
- **Phase 4 Step 2 batch 7a (103), original GT-II batch, 2026-11-22.**
  Approved item by item with full before-rows. `103` (now 40 rows). Rows
  added: Named Organization / Eno River Association (SD "land trust,
  conservation nonprofit") for "Five days after the meeting... the ERA
  announced it would push back eviction dates..."; Named Organization / Eno
  River Tenants Association for "Tenants’ shared goals were recast as a list
  of demands for the ERA..." (petitions test) and for "Soon after, in early
  May, the ERTA emailed petition supporters with major news..." (split out
  of Leslie St Dre's row); two Unnamed Group of People rows for "Many ERA
  members and several newer board members stated they were hearing about the
  land transfer’s human impact...": one SD "members", one Title "board
  members" (user: a board member is a formal governance title, so a separate
  source from members). Leslie St Dre rows trimmed to her own sentence:
  "Tenants were still in limbo after the first meeting. But at the next,
  Leslie says..." -> "But at the next, Leslie says the ERA came prepared to
  make decisions." (reporter opener dropped; ERTA email to its own row);
  "For the mediator role, the ERTA recommended Manju Rajendran...
  Rajendran’s “liberatory lens” was crucial, Leslie says..." -> from
  "Rajendran’s “liberatory lens”..." (item 55). "The ERA has also waived
  tenants’ rental payments..., according to the insert.": Named Organization
  / Eno River Association / SD "land trust, conservation nonprofit" ->
  Document / Name blank / SD "insert" (Minor 22: the attribution credits the
  document). Sheba Everett (all rows): Title "chief spokesperson of the
  ERTA" -> "chief spokesperson of the ERTA; preschool teacher" (user), SD
  blank -> "tenant". Graham Jones (all rows): SD "tenant" -> "tenant,
  trailrunner".
- **Phase 4 Step 2 batch 7b (101), original GT-II batch, 2026-11-22.**
  Approved item by item with full before-rows. `101` (37 rows). Title of
  Source: Nkhulambe "group village headman Nkhulambe" -> "group village
  headman" ("We’re not beggars..." row); Chipiliro Khamula "DoDMA
  Spokesperson" -> "Spokesperson" (as written; all 8 rows); Sosten Chiotha
  -> "climatologist; director of the NGO Leadership for Environment and
  Development in Southern and Eastern Africa" (all 4 rows; v62 item 24). Row
  removed: "Based on the official average of one household comprising 4.3
  people, the DoDMA figures... would indicate..." (DoDMA; the reporter's own
  calculation). Row added: Named Organization / DoDMA for "The country’s
  Department of Disaster Management Affairs (DoDMA) has put the economic
  damage at over USD 500 million." Hyoung-Joon Lim row trimmed: "With fewer
  resources, since July 2023 WFP has cut food rations... “We’ve had to make
  some very tough choices,” Lim adds." -> the quote only (item 55).
  "Climatologist Friederike Otto told Table Briefings that...": Named Person
  / Friederike Otto / Title "Climatologist" -> Named Organization / Table
  Briefings / Title blank (item 73, relayed speech). Lucy Chikapa SD
  "resident of Chilobwe" -> "mother, resident of Chilobwe". Villagers row SD
  stays "villagers" (v62 item 25).
- **Phase 4 Step 2 batch 7b, 7b.8, 2026-11-22.** `101` Mary Likopa rows (11,
  12): Source Justification "returned to her home in a dangerously
  disaster-prone zone out of desperation" (a paraphrase) -> "This is not
  because everyone left for a better place, but because some, like Mary
  Likopa, returned to dangerously disaster-prone zones out of desperation."
  (the text's sentence).
- **Phase 4 Step 2 batch 7b, 7b.8 continued, 2026-11-22.** `101` Mary Likopa
  rows (11, 12): SD blank -> "widow" ("The 42-year-old widow perches
  on..."). Row 11 kept in full (user): the sentences after "she says" carry
  on her account (v62 item 9).
- **Phase 4 Step 2 batch 8a (65), original GT-II batch, 2026-11-23.**
  Approved item by item with full before-rows. `65` (now 52 rows). Step 1
  NOTFOUND row fixed: the story text's quote broken mid-sentence
  ("...filling up [my car]" / "wasn’t such an issue...") was rejoined
  (REFLOW_2026-11.md); Kassandra Gutierrez's row now matches that paragraph
  (curly quotes) and its last quote, "“It’s such a fast increase that no one
  can really catch up.”" (its own paragraph), is a new Gutierrez row. Title
  of Source blanked (not in the text): Richard Kogura "owner of his family's
  Japanese gift shop", Phil Cosentino "family farm owner". Erika McEntarfer
  (all rows): Title "research scholar at SIEPR" -> "former head of the U.S.
  Bureau of Labor Statistics; research scholar at SIEPR", and the "former
  head of..." part removed from her SJ. "“Between 2012 and the pandemic,
  prices in the Bay Area increased faster...” researchers at SIEPR said.":
  Named Organization / Stanford Institute for Economic Policy Research ->
  Unnamed Group of People / Title "researchers at SIEPR" (item 41, v62 item
  22). "Average rental rates... the Silicon Valley index reported.": Named
  Organization / Joint Venture Silicon Valley / SD "nonprofit, think tank"
  -> Document / Silicon Valley Index / SD blank (Minor 22). Stasia Hansen's
  "Black workers have historically..." row trimmed to its first sentence;
  "They are also more likely, according to the index, to be paid less..."
  added as Document / Silicon Valley Index. Rows added: Richard Walker,
  "Walker, the Berkeley professor emeritus, said the inequality gripping the
  Bay Area is difficult to escape..."; Erika McEntarfer, "McEntarfer, who
  moved from Washington, D.C., to the Bay Area... said she loves the
  region." SD added: A.J. Fernandez "grand-nephew"; Kassandra Gutierrez
  "single parent". The "experts say" row keeps "experts" in Title (user).
- **Phase 4 Step 2 batch 8a, continued, 2026-11-23.** `65` Kelly Snider:
  Title "developer and professor in the Department of Urban and Regional
  Planning at San José State University" -> "developer; professor in the
  Department of Urban and Regional Planning at San José State University"
  (two titles, v62 item 24). Kassandra Gutierrez's Title "case worker at a
  mental health care agency in Oakland" kept (user).
- **Phase 4 Step 2 batch 8b (105 rows 1-24), original GT-II batch,
  2026-11-23.** Approved item by item with full before-rows. `105` (now 50
  rows). "A Texas Department of Criminal Justice spokesperson points
  out...": Named Organization / Texas Department of Criminal Justice ->
  Named Person / Amanda Hernandez / Title "Texas Department of Criminal
  Justice spokesperson" (v62 item 23: the one spokesperson the story names).
  Amanda Hernandez's other rows: Title "spokesperson, Texas Department of
  Criminal Justice" -> "Texas Department of Criminal Justice spokesperson"
  (as written). "The spokesperson declined to answer whether the TDCJ
  agrees... But generally, the agency frames fieldwork as a benefit to
  prisoners.": trimmed to the second sentence (the first is a non-response,
  item 16) and Amanda Hernandez -> Named Organization / Texas Department of
  Criminal Justice. "Per the disciplinary rules, minor penalties...": Named
  Organization / Texas Department of Criminal Justice -> Document / Name
  blank / SD "disciplinary rules" (Minor 22). State audit rows: SD "state
  audit" -> "audit report"; "Regardless of any perceived benefit to inmates,
  the benefit to taxpayers is not clear." dropped from the "The 2021 audit
  found..." row (reporter's assessment). "“When you think about cotton
  picking in Texas...” said Bianca Tylek... in an interview with the
  Marshall Project in 2021.": Named Person / Bianca Tylek / Title "founder
  of Worth Rises" / SJ "Founder of Worth Rises, a nonprofit opposing forced
  labor," -> Named Organization / The Marshall Project / Title and SJ blank
  (item 73). "Current inmates and their families describe...": SD "current
  inmates, families" -> "current inmates", plus a second Unnamed Group of
  People row with SD "families". Craig Anderson (all rows): SD "recently
  released" -> "released". Held for the item 44 sweep: Robert Hockley's
  narrated-experience rows ("His first sight of the cotton field...", "He
  was in shock...", "That’s how he learned...", "When Hockley refused to
  pick cotton...", "Back at Clemens...").
- **Phase 4 Step 2 batch 8c (105 rows 25-50), original GT-II batch,
  2026-11-23.** Approved item by item with full before-rows. `105` (now 53
  rows). Step 1 NOTFOUND row fixed: the Robert Hockley row that joined two
  paragraphs ("...dehumanize prisoners.He points out...", no final period)
  split into "Hockley suspects that the lack of air conditioning was never
  an oversight. Instead, he sees it as..." and a new row "He points out how
  some improvements that might seem compassionate... report misbehavior to
  officers." Rows added for "The lawsuit claims there have been more than
  270 heat deaths since 2001, and advocates identified a dozen last summer
  alone.": Document / SD "lawsuit" and Unnamed Group of People / SD
  "advocates". "State data shows..." SD "state data" -> "data". Michele
  Deitch row trimmed to her quote (the "is familiar with the conditions..."
  sentence is her SJ). Geologist row SJ "geologist from mid-1800s" (not in
  the text) -> blank. Telegraph and Texas Register row trimmed to its quote
  sentence (reporter background dropped). The TDCJ "A spokesperson
  responded, “I don’t have that information.”" row kept (user: a substantive
  reply to factual material, not a non-response).
- **Phase 4 Step 2 batch 9a (113 rows 1-33, p1-p69), original GT-II batch,
  2026-11-23.** Approved item by item with full before-rows. `113` (now 102
  rows). Title of Source as written: Mike Kubo (all rows) blank ->
  "astrobiology researcher at NASA"; Trish Hopkinson (all rows) "head of 360
  Value" / "head of 360Value" -> "Verisk’s head of 360Value"; Dan Veroff
  "Insurance Law Attorney" -> "attorney specializing in insurance law"; Jay
  Feinman "professor and author" -> "professor emeritus at Rutgers Law
  School; author of “Delay, Deny, Defend: Why Insurance Companies Don't Pay
  Claims and What You Can Do About It”" (v62 item 24); Peter Wells (all
  rows) "inventor of component-cost algorithms" -> "inventor of the
  technology". Veroff's and Feinman's SJs kept (user: not zero-sum with
  Title). Richard Salazar's own rows: SD blank -> "Vietnam War veteran".
  "Researchers in Colorado analyzed insurance claims data..." row: SJ
  "Regulators and consumer advocates have identified widespread
  underinsurance..." (another source's statement) -> blank. Rows added:
  "Regulators and consumer advocates have identified widespread
  underinsurance following wildfires..." twice, as Unnamed Group of People /
  Title "regulators" and Unnamed Group of People / SD "consumer advocates";
  Named Organization / CSAA for "And when a CSAA policyholder lost her home
  in a wildfire just four months after her agent set her policy limit with
  360Value, the insurer’s own claims process found she was underinsured by
  76%."
- **Phase 4 Step 2, 103 follow-up, 2026-11-23.** `103` (now 41 rows): row
  added, Named Organization / INDY, for "As the INDY previously reported,
  the tenants, some of whom have inhabited the homes for a decade or more,
  were shocked when notices to vacate were posted on their doors in
  September." (v62 item 10: the outlet credited by name for its own earlier
  reporting; the story is INDY Week's own). Now row 2; old rows 2-40 ->
  3-41.
- **Phase 4 Step 2 batch 9b (113 rows 37-69), original GT-II batch,
  2026-11-24.** Approved item by item with full before-rows. `113` (now 106
  rows, re-sorted into text order: the "In one case examined by
  regulators..." / "In another case, the department said..." rows and the
  two 2019-memo rows had been out of order). "But in practice, the Chronicle
  found..." Name "The Chronicle" -> "San Francisco Chronicle". Rows added,
  Named Organization / Verisk (SD "insurance analytics firm"): "Agents are
  supposed to verify the prefilled details, Verisk training materials
  instruct." (Minor 22: producer named) and "It taps a pricing database the
  company says relies on millions of claims that builders and adjusters
  submit." Memo row "The company acknowledged in the memo that agents rarely
  altered 360Value’s prefilled data." extended to its whole paragraph (from
  "Verisk had examined the accuracy of one prefill characteristic..."). Mike
  Kubo rows: "An astrobiology researcher at NASA, Kubo was hardly an expert
  in building homes." dropped from the "But the number felt like “a
  joke,”..." row (reporter's characterization); "Kubo found that this newer
  report was riddled with errors..." extended with "Those matched the house
  that burned down." Two-group rows split into one row per group (user):
  "Even when 360Value’s prefill information is accurate..., building
  contractors and public adjusters across the state told the Chronicle." ->
  Title "public adjusters" and SD "building contractors"; "Insurance agents
  and industry representatives interviewed by the Chronicle said..." ->
  Title "insurance agents" and Title "industry representatives" (was SD),
  same SJ on both. SD: Matt Everson "contractor" -> "contractor in Sonoma
  County"; licensing agreement row "licensing agreement" -> "user licensing
  agreement".
- **Phase 4 Step 2 batch 9b.1, 2026-11-24.** `113` (now 108 rows): the two
  Colorado-study rows ("Three-quarters of policyholders in the sample were
  underinsured." and "Nine in 10 homeowners in their dataset...") each split
  into two Unnamed Group of People rows, Title "researchers" -> "researchers
  at the University of Colorado at Boulder" and "researchers at the
  University of Wisconsin" (user: researchers at two universities are two
  groups; SJ replicated). "Researchers in Colorado analyzed insurance claims
  data..." keeps Title "researchers" (user: "in Colorado" may stand for the
  university, but not inferred).
- **Phase 4 Step 2 batch 9c (113 rows 76-108), original GT-II batch,
  2026-11-24.** Approved item by item with full before-rows. `113` (now 110
  rows, in text order). Ruth Stroup Title "Retired Farmers Agent" ->
  "retired Farmers agent" (as written). "In fact, California insurance
  department reports show...": Document / SD "insurance department reports"
  -> Named Organization / California Department of Insurance / SD blank
  (Minor 22: producer named). Vardaman's report: "Vardaman’s team had
  noticed..." Unnamed Group of People / SD "risk management team" ->
  Document / SD "report", extended with "The tools drew from the same
  data... But that wasn’t happening."; "A big part of the problem,
  Vardaman’s report found..." extended with "Nine characteristics had the
  greatest impact..."; "The problems went further. Vardaman’s team found
  instances..., " (cut off at a comma; Unnamed Group of People / risk
  management team) and "Vardaman’s team found instances..., which his
  colleague pointed out..." (Unnamed Person / SD "colleague") merged into
  one Document / SD "report" row with the full sentence (internal records:
  the people named in them are not sources). Email rows (Cleveland/Vardaman,
  five rows): SD blank -> "email"; "Inaccuracies continued to skew
  360Value’s calculations, Cleveland wrote." extended with "Farmers and
  Verisk employees had identified multiple features that were “frequently
  incorrect”...". "Soon after adopting 360Value for “main street” homes..."
  extended with "The company also “locked” certain values...". "The Salazars
  don’t remember their agent ever mentioning an algorithm..." Name "Linda
  and Richard Salazar" -> "Linda Salazar", plus a replicated Richard Salazar
  row (SD "Vietnam War veteran"). Rows added: Unnamed Person / Title
  "judge", "The judge ruled that because underinsurance is so specific to
  each person..."; Amy Bach, "She’s heard stories from consumers who say
  they asked insurance companies for more coverage, only to be told no."
- **Phase 4 Step 3 (merges) batch 3-1, original GT-II batch, 2026-11-24.**
  Same source, same paragraph -> one row (v61 Note 12a); other fields from
  the first row; no differing SJs. `94` (now 7 rows): Venus Williams "Venus
  Williams remembers, at 14 years old, her mother sewing her first tennis
  skirt." + "“I fell in love,” Williams said..."; Andrew Bolton "It’s for
  all those reasons that Bolton saw “Costume Art”..." + "“At once deeply
  personal and profoundly social...”"; Andrew Bolton "As Williams pointed
  out, Bolton stressed..." + "Or, as Bolton noted: “Even in its absence,
  dress persists...”" joined with a single space, leaving out the reporter's
  sentence between them ("In that way, the clothed human form is a
  reflection both of its times..."). `105` (now 51 rows): Amanda Hernandez
  "A Texas Department of Criminal Justice spokesperson points out..." +
  "Also, prisoners stopped picking cotton by hand..."; Texas Department of
  Criminal Justice "Most of the state’s roughly 100 prisons today are only
  partially air-conditioned..., per data from TDCJ." + "(We presented the
  TDCJ with a list of 13 prisons... “I don’t have that information.”)"
  joined with a single space, leaving out the outlet's analysis between them
  ("But the rates are much worse among the prisons which we know were built
  on former slave plantations...").
- **Phase 4 Step 3 (merges) batch 3-2, original GT-II batch, 2026-11-24.**
  `113` (now 109 rows): Verisk "Publicly, Verisk promotes its prefill data
  as “reliable”..." + "The company told the Chronicle such studies exist,
  but refused to provide them..." joined with a single space, leaving out
  the reporter's sentence between them; the merged row takes the second
  row's SJ "Yet Verisk has never published an in-depth study on the
  reliability of its prefill or pricing data." (the first row's SJ was
  blank). `124` (now 19 rows): duplicate World Health Organization row "The
  World Health Organization (WHO) reports there are now more than 600
  suspected Ebola cases..." removed (the copy at the top of the file, out of
  text order; the copy in place kept); Lina Moses "“Of course, it’s
  problematic because vaccines are some of our best tools...”" + "But other
  public health tools — public education, contact tracing, quick testing —
  still work, she said." merged. `152` (now 16 rows): Alaska Survey Research
  "An Alaska Survey Research poll on Aug. 11 reported..." + "A New York
  Times/Siena poll in June had Sullivan 2 points ahead, and an Alaska Survey
  Research poll in April..." merged (the New York Times/Siena row on the
  second sentence stays). `160` (now 28 rows): China "China has issued
  scathing remarks in response annual training recently conducted by
  Taiwan..." + "China went on to conduct what it called a joint maritime
  drill..." merged (the Jakarta row on the second sentence stays).
- **Phase 4 Step 3 batch 3-3, original GT-II batch, 2026-11-24.** `155`:
  "His attorneys argued that the state's double jeopardy law protects
  him..." Title "legal team" -> "attorneys" (as written; Minor 11).
- **Phase 4 Step 1 on SZ-9 (163, 164, 177, 178, 179, 180, 182, 183, 184),
  2026-11-24.** Approved as one table with full rows. Whitespace: 45
  trailing line breaks/spaces removed (Sourced Statements and Source
  Justification) in 163, 177, 178, 179, 180, 183, 184; two 163 rows ("Welch
  said in a statement shared with Newsweek over email that..." and "“I was
  charged with assaulting a police officer. Can you believe it?” she said.")
  applied after the user saw them in full (trailing line break only). Text fidelity: `164` Lonnie Bunch III
  row, straight apostrophes -> the text's curly ones; `164` Greg Werkheiser
  row, comma restored ("...changes to Washington, said in a statement...");
  `180` Itamar Ben-Gvir row, curly apostrophes. Full sentences: `178` André
  Ventura row extended to its whole sentence ("Santiago Abascal, of the
  Spanish Vox party, called it a “spectacular result...”, while André
  Ventura... praised a “historic victory”."); `180` Randy Fine row starts
  "U.S. Rep. Randy Fine..."; `180` Andy Burnham office row, final period
  restored; `183` Bipartisan Policy Center row ends "...in legal
  challenges." Titles: `177` "State authorities" -> "state authorities"
  (capital only from sentence start); `177` and `179` Gerhard Feige "Bishop
  of Magdeburg" -> "bishop of Magdeburg" (the text: "The bishop of
  Magdeburg, Gerhard Feige"). Elected/cabinet title capitals checked against
  the text and kept (President, Foreign Secretary, Israeli Foreign Minister,
  Prime Minister, Missouri Attorney General).
- **Phase 4 Step 2 Z1 on SZ-9 (164, 182, 184, 163), 2026-11-24.** Approved
  item by item with full before-rows. `164` (now 10 rows): row added, Lonnie
  Bunch III, "In a July interview with CNN’s Christiane Amanpour, Bunch
  said, “It’s our job to tell an accurate, complex and truthful history.”"
  (CNN is the story's own outlet); "Two of nine public regents’ terms have
  expired..., according to a term chart on the Smithsonian website.":
  Document / SD "chart" -> Named Organization / Smithsonian Institution / SD
  blank (Minor 22: producer named). `182` (now 8 rows): Brett Kavanaugh row
  "The ruling ends the confusion that had unfolded when the judge issued his
  order soon after Supreme Court Justice Brett Kavanaugh on Sept. 8 declined
  to block..." removed (a court action reported as an event, as 113's "a
  judge denied..."); Denny Hoskins Title "Missouri’s secretary of state" ->
  "Missouri’s secretary of state, Republican", SD "Republican" -> blank;
  Catherine Hanaway Title "Missouri Attorney General" -> "Missouri
  Republican Attorney General", SD "Republican" -> blank (elected officials:
  party in the Title, as written). `184` (now 14 rows): rows added --
  Document / SD "report", "Threat actors included criminals, suspected
  state-sponsored groups, spyware vendors and state propaganda institutions,
  the report said."; Named Organization / Anthropic / SD "company", "In
  China, three accounts aligned with the PRC municipal security service used
  Claude for surveillance and transnational repression..." (carries on
  Anthropic's account). `163` (now 15 rows): "According to a report from the
  outlet Spectrum News from March of 2025...": Document / SD "report" ->
  Named Organization / Spectrum News / SD "outlet" (v62 item 10, Minor 22);
  Jill Stein SD "former presidential candidate" -> "former presidential
  candidate, environmental activist, politician"; row added, Jill Stein,
  "She is a prominent figure in third-party politics and has regularly
  argued that the Democratic and Republican parties do not adequately
  address issues including climate change."
- **Phase 4 Step 2 Z2 on SZ-9 (183, 177, 178), 2026-11-24.** Approved item
  by item with full before-rows. `183`: Bipartisan Policy Center row trimmed
  to "As of 8 September, the US has collected about $210bn in tariff and
  excise tax revenue in 2026." (first sentence is its SJ; the third is the
  reporter's own calculation); Bernie Moreno Title "US senator" -> "US
  senator, Ohio Republican", SD "Ohio Republican" -> blank (elected
  official: party in the Title). `177` (now 17 rows): removed -- "The CDU,
  on about 23%, under its state prime minister, Sven Schulze, hopes to be
  able to construct a multiparty coalition..." (a hope, no speech act; item
  21) and "Siegmund has sought to capitalise on a mood of ostalgie..."
  (reporter's analysis). `178` (now 26 rows): Matteo Salvini and Geert
  Wilders SD blank -> "far-right" (Note 21a); Ulrich Siegmund SD "lead
  candidate" -> "lead candidate in Saxony-Anhalt"; removed -- "AfD won 43.8%
  of the vote..." and "The centre-left Social Democrats scored 9.3%..."
  (election results with no attribution, typed as Document / polls).

### AV

- **Story 171 (Bracing_Next_George_Floyd) row 3 — resolved a row you'd
  flagged "Unresolved."** Your draft had this row (Sourced Statement:
  "Corporations and government officials promised diversity and
  equity.") with Type of Source explicitly marked "Unresolved" and every
  other field "null" — a fair call, since it's a genuinely tricky case:
  one sentence, one verb, but two different *kinds* of unnamed
  attribution jointly named as its subject (an unnamed group of people —
  "government officials" — and an unnamed group of *organizations* —
  "corporations" — which our schema doesn't have a type for at all).
  Resolved by checking this row against its immediate neighbors: row 2
  ("protesters filled the streets demanding change") and row 4 ("many
  Black Minnesotans say...") are both already typed Unnamed Group of
  People in your own draft, and row 3 fits that exact same pattern. Fixed
  to Type of Source "Unnamed Group of People," Title of Source
  "government officials" (credentialing, per the schema's own UGOP
  definition, which lists "officials" directly as an example). The
  "corporations" half isn't captured as its own structured field — there's
  currently no type for an unnamed group of *organizations* — but it's
  still right there in the Sourced Statement text itself, nothing is
  lost from the row overall. Logged as an open schema gap in
  `development-of-v59.md` (item 10) in case this pattern comes up again
  often enough to justify adding a real type for it.
- **Story 72 (Quantum_Campus) row 17 — split what looked like two
  sources back into one.** Sourced Statement: "A. Anne Holcomb, co-chair
  of ETHOS and a 15-year South Shore resident, confirmed two documented
  events." Your draft had Type of Source "Named Person and Unnamed
  Person" and Name of Source "A. Anne Holcomb and Resident" — reading
  this as two people, when it's actually one person (Holcomb) with two
  facets: her formal title (co-chair of ETHOS) and a non-credentialing
  descriptor (resident). Fixed: Type of Source → "Named Person", Name of
  Source → "A. Anne Holcomb" only, Title of Source unchanged, Source
  Descriptors → "South Shore resident" (dropping "15-year" as a duration
  modifier, keeping the location — matches how this exact phrase is
  already used elsewhere in this corpus, e.g. story 55's Andrew Torrence
  and Jayna McGruder, also South Shore residents).
- **Story 114 (Kansas_Gender_Transitioning_Ban) row 9 — flagged and now
  fixed, same schema gap as story 171 above.** Sourced Statement: "Last
  week, a New York City hospital said it was one of several to have
  received a grand jury subpoena..." — an unnamed hospital, which is an
  unnamed *organization*, not a group of people. Your draft had Type of
  Source "Unnamed Group," which (if just meant as shorthand for "Unnamed
  Group of People," as it does elsewhere in your files) would misrepresent
  this as a group of people rather than an institution. Set Type of
  Source to the literal placeholder "Not Defined-Unnamed Org(s)" rather
  than force it into either Named Organization (no name is given) or
  Unnamed Group of People (it's not people) — see `development-of-v59.md`
  item 10 for the fuller reasoning and the convention this established.
  2026-09-18 update: also backfilled Source Descriptors with "New York
  City hospital" (previously blank) — the org descriptor the reporter
  did give, even without a formal name, shouldn't be lost just because
  the row can't get a real Type; Source Justification already had
  substantive content and was left as-is. Worth discussing at the next
  meeting since this is the second time this
  exact gap has come up in one day.
- **Thin Source Justification fixed — story 74, "One speaker, a student
  at Gale Ranch Middle School" row.** Your original annotation had Source
  Justification as just the single word "Speaker" — accurate but too thin
  to explain the source's actual standing to be quoted (the reporter is
  signaling their connection to the story: they were one of the speakers,
  and a Gale Ranch Middle School student, at the meeting). Expanded to
  "speaker, a student at Gale Ranch Middle School." Flagging as a pattern
  worth watching for in future annotation, not just this one row.
- **Thin Source Justification fixed — story 74, Bruce Hixon row.** Your
  copy had Source Justification as just the single word "Speaker" for
  Bruce Hixon (introduced in the article as "one of the first speakers of
  the night"). SZ's copy of the same story already had the fuller "one of
  the first speakers of the night" for this same source — applied that
  same fuller text to your copy for consistency. Second thin-SJ instance
  caught in this same file (see the Gale Ranch item above).
- **Story 108 (Youth vs. Apocalypse) — Source Descriptors trimmed, Source
  Justification added.** Applies to both your copy and SZ's. The
  article's own description, "YVA is a youth-led, Bay Area-based
  collective of young climate justice activists," had been condensed
  entirely into Source Descriptors ("youth-led climate justice
  collective") with nothing captured in Source Justification at all,
  across all 6 rows where YVA is the source. Per the schema, only the
  bare category word belongs in Source Descriptors ("collective") and the
  fuller descriptive sentence belongs in Source Justification. Trimmed SD
  to "climate justice collective" and populated SJ with the full sentence
  on all 6 rows (combined via ";" on the one row that already had
  different context). Likely why SD had grown overloaded: that
  descriptive sentence sits between two quotes in the reporter's own
  voice, not directly attached to any single "YVA said/did X" row, so
  it's an easy thing to read past when annotating row-by-row.
- **Corpus text mismatch found and fixed for 8 story numbers (166, 167,
  168, 170, 171, 173, 174, 176) while bringing in your new "solidarity
  reporting initiative" text batch (2026-09-16).** Our local
  `extracted_articles_boilerplate/` already had *something* saved under
  these 8 numbers, left over from an earlier fetch/numbering pass — but
  checking each one's actual headline against the current Expansion List
  showed none of them matched. Two (166, 167) both held a duplicate copy
  of the same wrong article (the same underlying bug already flagged and
  marked resolved for story 167's URL — the spreadsheet URL got fixed at
  the time, but the already-fetched text file never got refreshed to
  match). Four more (168, 170, 173, 174) turned out to be real articles,
  just sitting under the wrong number — each one's true story number
  (190, 189, 187, 170 respectively) already has its own correctly-numbered
  text in that day's delivery, so these were simply redundant duplicates.
  The remaining two (171, 176) don't match any headline in the 166–190
  batch at all — moved aside to `stale-story-text-leftovers/` (not
  deleted) rather than guessed at, since forcing them onto a number
  without confirmation is exactly the kind of mistake this check exists
  to catch. **To be clear: nothing about the GT Expansion List
  spreadsheet itself was wrong here** — every one of that day's freshly
  delivered texts matches its assigned number's headline exactly. This
  was purely a stale local-copy issue on our side, now fixed: all 17
  numbers in that batch (166–176, 185–190) have verified,
  correctly-matched text in `extracted_articles_boilerplate/`.
- **Story 128 (Congress_Pay_Battle) row 1 — same schema gap as stories
  171/114 above.** Sourced Statement: "A federal court has finally
  weighed in on the sensitive topic of congressional member pay..." — an
  unnamed court, an unnamed organization rather than a group of people.
  Your draft had Type of Source "Unnamed Group of People," Title "federal
  court." Set Type to "Not Defined-Unnamed Org(s)," moved "federal court"
  from Title (which should stay blank for org-shaped sources) into Source
  Descriptors instead — the descriptor shouldn't be lost just because the
  row can't get a real Type. See `development-of-v59.md` item 10 for the
  fuller convention.
- **Story 50 (Peru_Election) row 8 — Source Descriptors backfilled from
  Source Justification.** Sourced Statement ends "...as critics say
  Congress has weakened the oversight mechanisms meant to combat crime."
  Your draft had "Critics" sitting only in Source Justification, with
  Source Descriptors empty — "critics" is non-credentialing (a stance
  word, not a role), so it belongs in SD. Backfilled SD "critics"; left
  SJ as "Critics" rather than blanking it — SD and SJ aren't mutually
  exclusive, and there's no richer context anywhere in the article to
  expand SJ with (the surrounding text is reporter scene-setting, no
  elaboration on who the critics are), so it stays thin but not
  fabricated.
- **Story 128 (Congress_Pay_Battle) rows 3, 8, 9 — moved non-credentialing
  words from Title of Source to Source Descriptors.** Your draft had
  "Plaintiffs" (rows 3, 8) and "COLA proponents" (row 9) in Title of
  Source — both are non-credentialing (a legal-party role and a
  stance-holder word, not professional/institutional titles), so they
  belong in SD per the credentialing test. `170-Fire_Prevention_Homeless.csv`
  row 14 already has "Plaintiffs" correctly in SD, so this also brings
  128 in line with that existing precedent. "COLA proponents" kept whole
  (the generic topic "COLA" stays attached to the non-credentialing word,
  per item 35's rule) rather than trimmed to bare "proponents."
- **Story 189 (Cameroon_Unpaid_Wages) — "publishers" and "journalists"
  moved from Source Descriptors to Title of Source, for consistency.**
  Row 22 had "Publishers" in SD while row 29 had "publishers" in Title —
  same word, same file, same UGOP type, no contextual difference between
  them. Professional/industry-role nouns are credentialing per this
  corpus's own established precedent elsewhere ("analysts," "Doctors,"
  "experts," "researchers," "forecasters" are all Title, not SD), so
  moved row 22 to match row 29 rather than the other way around. Same
  logic extended to "journalists" (rows 1, 5), also previously sitting in
  SD — reporters/journalists are a credentialed-role class too, so both
  moved to Title.
- **Mechanical cleanup: 1,505 literal `"null"` string cells cleared to
  true blank, across all 17 of AV's solidarity-batch files (166-176,
  185-190).** A commit earlier in the Sep17 batch's Phase 0 pass claimed
  this was already "cleared throughout," but that only covered the 3
  specific rows called out in that commit (168 row 28, 172 row 16, 171
  row 3) — the rest of these 17 files still had the literal string
  "null" (not a true empty cell) scattered through Name/Title/SD/SJ
  wherever a field was blank. Purely mechanical, no judgment involved;
  caught while fixing story 189's Title/SD rows above and swept across
  the whole batch once found.
- **Story 186 (Homeless_Camp_Sweeps) rows 45, 46 — "ARR worker"/"HSO
  employee" moved from Title of Source to Source Descriptors.** Both
  Austin Resource Recovery and Austin's Homeless Strategies and
  Operations Department are named in the article, but the sourced
  individuals here are just described as line staff/laborers ("the
  employee said," "two HSO employees emerged"), not officials,
  spokespeople, or anyone in a position of authority within either
  department. "Worker"/"employee" denote employment status, not a
  licensed profession, expertise, or leadership role — non-credentialing,
  same as "protestor" or "resident," unlike "Doctors" or "publisher"
  (effectively CEO-level) elsewhere in this batch. Moved the whole
  phrase to Source Descriptors rather than stripping the org name: with
  no Name of Source to make the org tag redundant (unlike the
  org-affiliation-stripping cases, e.g. Sean Crist), the org affiliation
  is the only identifying signal these two rows have at all.
- **Story 176 (Roadblocks_to_Relief) rows 3, 5, 8, 11 — Source
  Descriptors backfilled from the atomic word already sitting in the
  Sourced Statement text.** All four rows had Name/Title/SD/SJ
  completely empty despite an explicit descriptor word right there in
  the SS: "one interviewee" (rows 3, 5, 11) and "one community member"
  (row 8, matching this same file's existing "Community members" SD
  precedent at row 4). Backfilled SD "interviewee" and "community
  member" respectively; nothing else in these rows changed.
- **Correction to the above, found during the Phase 3 Source
  Descriptors compliance audit: "interviewee" doesn't actually qualify
  as a Source Descriptor — blanked on rows 3, 5, 11.** Not an
  annotator error (this was our own earlier migration fix, not your
  original draft) — a genuinely new gap identified this session.
  "Interviewee" names the source's role in the journalism process
  itself (the fact of being interviewed), not an identity, action, or
  relationship within the story, the way "protestor" or "resident"
  does — the same underlying problem as a bare "person"/"individual"
  count-noun carrying no discriminating information. Row 13's
  "community members, interviewees" trimmed to "community members"
  only, same reasoning. Row 8's "community member" (row 4's
  "Community members" precedent) is unaffected — that's a real
  identity/relationship word, not a sourcing-process word. Drafted as
  a new prompt addition, `development-of-v59.md` item 31 in the
  Prompt Updates checklist.
- **Story 176 — two new rows added, found during Phase 4a (missing
  Sourced Statement sweep), inserted after the old row 5.** Note this
  shifts every row number from the old row 6 onward by +2 — the two
  entries just above (row numbers 8, 11, 13) refer to this file's
  *pre-Phase-4a* numbering; those same rows are now 10, 13, and 15
  respectively. Two genuinely uncaptured statements, both in the same
  passage: (1) "Many shared that the documentation required — for
  example, three prior pay stubs, a drivers license, or a Social
  Security Number — made applying to social services nearly
  impossible." — Unnamed Group of People, "shared" is already an
  established attribution verb elsewhere in this same file (old rows
  6, 10). (2) "'I needed a cosigner, so not everything is under my
  name,' one shared when expressing their difficulty in filling out a
  rental assistance application." — Unnamed Person, a direct quote
  with clear attribution ("one shared"), Source Justification carrying
  the trailing context clause. New rows inserted as 6 and 7; everything
  from the old row 6 onward renumbered by +2 to preserve article order
  (final sequence: 1-15).
- **Story 189 (Cameroon_Unpaid_Wages) rows 2, 3, 4 (Mohamed Auwal) —
  "private media journalist" moved from Source Descriptors to Title of
  Source, matching this same file's own "journalists"/"publishers"
  fix.** Same inconsistency as rows 1/5/22/29 fixed earlier in this
  batch: "journalist" is credentialing, belongs in Title. Checked
  whether "private" was decorative before moving the whole phrase
  together — it isn't: the article is specifically about the difference
  between private and state-aligned media in Cameroon's press-freedom
  context, so "private" is category-defining here, not elaboration.
  Source Justification ("in Ngoundere, northern Cameroon") unchanged.
- **Story 50 (Peru_Election) row 4 — Document reclassified to Named
  Organization.** Sourced Statement cites "recent surveys by Datum
  International and the Institute of Peruvian Studies" — two named
  polling organizations releasing survey data under their own names,
  which per an existing GT-2026 rule (a named org formally releasing/
  standing behind a document in its own name is Named Organization, not
  Document) should never have been typed Document to begin with. Fixed:
  Type → Named Organization, Name of Source → "Datum International and
  the Institute of Peruvian Studies" — captured with "and" exactly as
  the article phrases it, not split into two rows (see the Note 12
  polling-consortium clarification, `development-of-v59.md` item 23).
- **Story 171 (Bracing_Next_George_Floyd) rows 11, 12 — same fix.**
  Name of Source was already correctly "NPR/PBS NewsHour/Marist poll"
  (three named orgs), but Type was still Document — the original
  capture was half-done. Fixed: Type → Named Organization, Name trimmed
  to "NPR/PBS NewsHour/Marist" (dropped "poll," genre language rather
  than part of the org's actual name), Source Descriptors cleared (was
  "poll" — same reasoning). Not split into 3 rows, matching the same
  polling-consortium convention as story 50 above.
- **Story 71 (Global_Deforestation_Slows) rows 1, 2 — same
  reclassification, single-org case.** "A report published... by World
  Resources Institute" — a single named org releasing a report under
  its own name, same rule as stories 50/171 above (no joint-credit
  question here, only one org). Fixed: Type → Named Organization, Name
  of Source → "World Resources Institute". Title/SD stay empty; Source
  Justification ("Report by World Resources Institute") unchanged.
- **Story 50 (Peru_Election) rows 1, 5, 6, 7 — Source Descriptors
  backfilled "polls" from Source Justification.** All four cite polls
  generically ("according to polls," "in the polls") with no
  organization named — correctly stays Document. SD was empty while SJ
  already had the bare genre word "polls." Backfilled SD "polls" on all
  four rows; SJ left unchanged (independent fields, same as row 8's
  earlier fix in this same file).
- **Story 69 (Trump_Assassination_Suspect) rows 4, 5, 8, 9 — Source
  Descriptors backfilled "affidavit"; fixed a spelling error in Source
  Justification.** Considered and confirmed these stay Document, not
  Named Organization, even though "F.B.I." is named — the affiant (the
  individual F.B.I. agent who swore the affidavit) is never named, so
  there's no named individual to attribute it to either; the affidavit
  is evidentiary/procedural material from a legal proceeding, not the
  F.B.I. releasing an institutional publication the way an organization
  releases a report. Backfilled Source Descriptors "affidavit" (bare
  genre word) on all four rows; also fixed "affadavit" → "affidavit" in
  Source Justification on all four (a misspelling that didn't match the
  article's own spelling, "federal affidavit unsealed on Monday").
- **Story 172 (Delhi_Flyover_Homeless) row 16 — Source Descriptors
  backfilled "data."** "Data suggests that nearly 99% of people living
  on the streets suffer inadequate and interrupted sleep during extreme
  heat" — no organization named, correctly stays Document (this is the
  same row that was resolved from "Unresolved" earlier in the migration;
  a documentation note elsewhere had mislabeled it "row 17," now
  corrected to row 16 to match the actual CSV). Backfilled Source
  Descriptors "data," matching the "state data"/"census data"
  generic-citation precedent this row was originally typed under. Source
  Justification stays empty — no additional context in this row's own
  text to capture.
- **Story 67 (California_Ships_Whales) row 12 — reclassified Named
  Organization to Unnamed Group of People; NOAA doesn't formally stand
  behind individual scientists' beliefs.** "Federal scientists at the
  National Oceanic and Atmospheric Administration believe changes in
  gray whale feeding grounds..." — your draft had this typed Named
  Organization (Name "National Oceanic and Atmospheric Administration")
  with "Federal Scientists" sitting in Title of Source, which Named
  Organization can never carry (a schema violation on its own). But the
  deeper issue: NOAA employs hundreds of scientists, and there's no
  reason to assume the agency has put its institutional stamp of
  approval behind these particular scientists' individual research
  beliefs, unlike a spokesperson whose statement genuinely represents
  an org's official position. The reporter is signaling *where these
  scientists work*, not attributing an official NOAA position. Fixed:
  Type → Unnamed Group of People, Title kept "Federal Scientists"
  (already correct there), Name of Source cleared — the NOAA affiliation
  isn't lost, it's still right there in the Sourced Statement text, just
  not promoted to a structured Named Organization row. See
  `development-of-v59.md` item 41 for the fuller reasoning — this
  establishes a new general distinction (formal spokesperson/official
  capacity vs. individual professional judgment with an org given only
  as employment context) worth watching for elsewhere in the corpus.
- **Story 168 (Texas_Power_Outages) row 5 — reclassified Named
  Organization "Austin" to Unnamed Group of People, same pattern as
  story 67 above.** "Officials in Austin, for example, said Feb. 19
  that restoring water services would likely be a multiday process for
  the whole city." Your draft had Name of Source "Austin" (a city, not
  a clearly-designated representative capacity) — this reads as
  "officials" (an unnamed group) located in Austin, not the City of
  Austin formally speaking through a designated representative. Fixed:
  Type → Unnamed Group of People, Title → "Officials", Name of Source
  cleared.
- **Story 170 (Fire_Prevention_Homeless) row 8 — same reclassification,
  found while checking a related question on story 181.** "Berkeley
  city staff also cited problems like crime and domestic disputes." —
  your draft had Name of Source "Berkeley," but "city staff" is a
  generic/informal collective reference (same shape as "officials in
  Austin" above), not Berkeley's government formally speaking. Fixed:
  Type → Unnamed Group of People, Title → "city staff", Name of Source
  cleared.
- **Story 174 (PPP_Loans_Low_Income) — Source Descriptors populated for
  8 Named Organization rows.** Direct same-row captures: row 7 and row
  23 (U.S. Small Business Administration, both say "the agency" in
  their own Sourced Statement text) → SD "agency"; row 13 (Color Of
  Change, "a racial justice organization") → SD "racial justice
  organization"; row 29 (Wells Fargo, "the fourth-largest PPP lender")
  → SD "lender". Carry-forward captures (the category word is stated in
  a different sentence about the same organization elsewhere in the
  article, not this specific row's own text): rows 16 and 27 (also SBA)
  → SD "agency"; row 17 (Bank of America, "the top PPP lender" appears
  in a different sentence) → SD "lender"; row 43 (Square, "both top PPP
  lenders" appears in a different sentence) → SD "lender". The
  carry-forward extension to Named Organization category words is a new
  principle confirmed this session — see `development-of-v59.md` item
  27 for the reasoning (an org's category is a stable, story-wide fact,
  not moment-specific the way a human source's descriptor can be) — no
  fix needed on your part, this reflects new schema guidance being
  applied, not an error in your annotation.
- **Story 186 (Homeless_Camp_Sweeps) rows 36, 37, 39, 40, 42, 67, 68 —
  full name recovered for Reyes.** The article introduces him once, at
  first mention, as "Alfredo Reyes, a worker with the advocacy group
  VOCAL-TX" (line 9) — every later reference uses only "Reyes," and
  your draft had Name of Source as just "Reyes" across all 7 rows.
  Fixed to "Alfredo Reyes," matching the one full-name instance in the
  article. Checked the other first-name/surname-only sources in this
  same file (Puma, Betty, August, Niedzielski, Mike-Mike, Brianna,
  Joey, Angel, Jazz) against the full article text — none of them are
  ever given a fuller name anywhere, so those stay as-is; Reyes was the
  one genuine case.
- **Story 169 (Black_Mothers_Gun_Violence) rows 5, 6, 7, 8 (Shea
  Kuykendoll) — "student advocate" moved from Title of Source to
  Source Descriptors.** "She is a student advocate at the University of
  Memphis" — "advocate" is non-credentialing, same pattern as the
  established "homeless advocate" precedent (Paul Boden, GT-2026).
  "Student" is the cause/topic being advocated for, so it stays attached
  to "advocate" (per the same rule that keeps "homeless advocate" whole
  rather than trimming to bare "advocate"); "at the University of
  Memphis" is the actual organizational affiliation, which strips out —
  it's already captured in Source Justification, so nothing is lost.
  Fixed: Title of Source cleared, Source Descriptors "mother" →
  "mother, student advocate" (accumulating alongside the existing
  descriptor), Source Justification left untouched.
- **Story 188 (Trans_Women_Incarcerated) rows 19, 20, 21, 22, 23, 27,
  28 (Bamby Salcedo) — Title of Source backfilled from later in the
  same story.** Rows 45-46 already correctly had Title "CEO and
  President of Los Angeles-based TransLatin@ Coalition" (the article
  reveals this at "Salcedo is now the CEO and President..."), but the
  earlier rows — describing her incarceration decades ago, before that
  title is stated — had Title empty. Per the existing carry-forward
  rule (a title is a global identifier for the source across the whole
  article, applied even when the reporter introduces someone by name
  first and states their title later), backfilled the same Title onto
  all 7 earlier rows to match. Source Descriptors and Source
  Justification unchanged.
- **Story 190 (West_Texas_Drag_Queens) rows 9, 10, 36 (Miss Calvina) —
  Title of Source and Source Descriptors both populated from a
  credentialing fact stated in a different source's sentence.** The
  article never introduces Miss Calvina with a title directly — the
  fact comes embedded in a sentence about Grace Rogers (a different
  named source in this file): "She attended the drag queen story hour
  to support Miss Calvina, who works as the choir director and organist
  at her Episcopal church in Lubbock." Still unambiguously about Miss
  Calvina by name, satisfying the no-inference rule. Split the two
  roles per the credentialing test: "choir director" is a formal
  leadership position (same bucket as other "director" titles already
  established as credentialing) — kept whole with its org/location
  context, matching the "director, the Brennan Center" convention.
  "Organist" is a skilled occupation without institutional authority —
  same bucket as the trades explicitly excluded from Title (plumber,
  carpenter) — moved to Source Descriptors instead. Fixed: Title →
  "choir director, Episcopal church, Lubbock", Source Descriptors
  "drag queen" → "drag queen, organist", Source Justification
  unchanged.
- **Story 185 (The_Uncounted) rows 27, 41, 43, 53 (Katie O'Bryant) —
  reconciled two different title phrasings for the same role into one
  global Title, with the non-credentialing half moved to Source
  Descriptors.** The article uses two different phrasings at two
  points: "Punks With Lunch outreach worker Katie O'Bryant" (line 130,
  row 41's existing Title) and "Outreach coordinator Katie O'Bryant"
  (line 137, rows 43/53's existing Title); row 27 had no title at all.
  "Outreach worker" is a generic employment role without institutional
  authority — same bucket as the "ARR worker"/"HSO employee" cases
  fixed in Phase 2.5, non-credentialing. "Coordinator" is a real
  institutional job title, credentialing. Fixed: Title → "Outreach
  coordinator" on all 4 rows (backfilled to row 27, replacing row 41's
  org-attached phrasing); Source Descriptors → "outreach worker" on all
  4 rows (org name "Punks With Lunch" stripped, per the org-affiliation
  rule). Source Justification unchanged throughout.
- **Story 174 (PPP_Loans_Low_Income) — pulled from active migration,
  needs a direct session with you (2026-09-21).** While reviewing a
  Title-of-Source question for KB Brown/Katie Brown ("owner of
  Wolfpack Promotionals" — a strong precedent for this exists
  elsewhere in the corpus), the user checked the original XLSx directly
  and decided this file needs full attention together with you rather
  than continuing through the normal process. The Phase 2.4/2.5 work
  already done and committed on this file stays as-is; only the
  remaining Phase 2.6 work is paused. Will come back to this in a
  future batch pass — see `development-of-v59.md` item 42.
- **Pattern flagged across the whole solidarity journalism initiative
  set (166-176, 185-190) — please re-review, don't wait for us to fix
  it during migration (2026-09-22).** A number of Sourced Statement
  rows across these stories are the reporter narrating what a source
  knew, believed, saw, felt, or lived through — in the reporter's own
  third-person voice, with no attribution verb ("said," "recalled,"
  "told us," etc.) and no quotation marks anywhere in the row. These
  read as background/context about the source, which is what Source
  Justification is for — not a statement the article is attributing
  *from* the source, which is what Sourced Statement requires. Two
  clear examples, both following the same shape — unattributed
  narration rows immediately followed by a properly-attributed row
  from the same source:
  - **Story 174, rows 1-4 (KB Brown / Katie Brown):** "As the pandemic
    hit, orders cratered at KB and Katie Brown's print shop in
    Minneapolis. They knew they would need help..." and "But despite
    trying three lenders, they never got one of the forgivable
    COVID-19 loans... They had to furlough their four employees and
    still don't fully understand what went wrong..." — no attribution
    anywhere in either row. Row 5, right after, is properly attributed:
    "'I just don't think the program was slated towards really small
    businesses,' said KB Brown..."
  - **Story 175, rows 24-27 (Jamie Barboan):** "Barboan, 44, knew the
    equipment was helping her mom stay alive. But she worried she
    would hear the sound for the rest of her life." and "Her brother,
    Andreas Tolth, was 39. She thought he would be OK. He never drank
    or smoked." — no attribution anywhere in either row. Row 28, right
    after, is properly attributed: "she had heard others warn, 'Once
    they're in the hospital, you won't see them (again).'"
  - **Also worth a look, not yet confirmed as the same pattern:** story
    175 row 85 (Kris Pikaart) and story 190 rows 31-32 (Donabela
    Kanela, Vanessa P. Nevaeh).
  - In general check whether such content should move into Source
    Justification for downstream attributed rows from the same source.
    Because the reporter may be narrating the lived experience/
    journey/"world through the eyes" of the source in order to set up
    for quotes that come later, which makes such lines candidates for
    SJ.
  Once you send back revised files, we'll diff them against what's
  already migrated to reconcile rather than starting over. Full
  writeup and reasoning in `development-of-v59.md` item 44.
  **New example found 2026-09-24, story 185 row 50** (San Francisco
  Department of Public Health): "The San Francisco Department of
  Public Health intends to maintain a registry of people at 'high
  risk' of overdose, and deploy firefighter paramedics and public
  health workers to meet with survivors within 72 hours of an
  overdose to connect them with services and treatment. Hospitals,
  911 responders, and community groups are supposed to identify
  people who have just survived overdose, and report them to the
  program. It's part of a $13.2 million effort to prevent overdoses in
  San Francisco. No such government-led efforts are currently underway
  in Alameda County." No attribution verb anywhere in it, and the last
  sentence is explicitly the reporter's own comparative observation
  (SF vs. Alameda County) — same pattern as the KB/Katie Brown example
  above, just for a Named Organization row instead of Named Person.
  Left as-is for now, flagged for your review rather than fixed on our
  side, matching the exclusion for this whole story range.
- **Story 173 (Mountain_View_RV_Dwellers) rows 1, 2, 3 (Misty Masvalo)
  — Title of Source backfilled from later in the same story.** Rows
  4-7 and 23-24 already correctly had Title "part-time yoga instructor
  and preschool teacher," but rows 1-3 — earlier in the article, before
  that occupation is stated — had it empty. Confirmed "teacher" (K-12/
  school-context) is credentialing per real precedent already in
  GT-2026 (`6-OUSD-basic-job.csv`, "Sixth-grade science teacher, Frick
  United Academy of Language"), and "instructor" is the same kind of
  credentialed position in the U.S. Backfilled the same Title onto all
  3 earlier rows to match.
- **Story 185 (The_Uncounted) rows 45, 46, 47, 48, 49, 51, 52 (Ana) —
  Source Descriptors backfilled "unhoused, mother" from Source
  Justification.** SJ already reads "an unhoused mother from Mexico who
  has lived near the park for years" on every row, but SD was empty
  throughout. Checked both the atomic-word rule and corpus precedent
  before deciding what to keep: bare "mother" is the overwhelming norm
  across the corpus (20+ instances with no qualifier in story 169
  alone), and while location qualifiers are consistently kept on
  "resident" ("South Shore resident," "resident of San Ramon," etc. —
  location is constitutive of what "resident" means), no precedent
  exists anywhere for a nationality/origin qualifier attached to
  "mother" or "unhoused." "From Mexico" reads as elaborating detail
  (the prompt's own "witnesses with criminal records" example — trim
  it), not a tightly-bound label like "former felons." Backfilled SD
  "unhoused, mother" (bare); Title and Source Justification unchanged.
- **Story 185 (The_Uncounted) rows 21, 22, 23, 24, 34, 35, 36, 40
  (Thad) — Source Descriptors backfilled "resident of North Oakland."**
  Row 21's own text directly identifies him: "we spoke with Thad, a
  36-year-old resident of North Oakland." A second possible descriptor,
  "unhoused" (row 40: "Thad, like many unhoused people who use drugs,
  obtains Narcan..."), was considered and set aside — that's a
  simile/comparison, not a direct statement that Thad himself is
  unhoused, so it wasn't captured. Backfilled SD "resident of North
  Oakland" (matching the "resident of San Ramon"/"South Shore resident"
  location-kept precedent) on all 8 rows.
- **Story 186 (Homeless_Camp_Sweeps) row 16 (Jennifer Miller) — Source
  Descriptors populated "unhoused."** "Jennifer Miller said that she
  had been unhoused for three years..." — checked prevalence across
  all of GT-II first: 26 existing "unhoused"/"homeless" SD instances,
  every one bare, zero with a duration qualifier attached anywhere in
  the corpus. Trimmed "for three years" per that consistent precedent
  (same pattern as "longtime resident" -> "resident"). SD "unhoused";
  Title and Source Justification unchanged.
- **Story 188 (Trans_Women_Incarcerated) rows 51, 52, 55, 56, 62
  (Jennifer Orthwein) — Source Descriptors populated "friend," trimmed
  from "friend of Jones."** SJ already reads "now a friend of Jones."
  You raised a sharp question here: doesn't this contradict keeping
  "resident of San Ramon" whole elsewhere? Resolved — no contradiction,
  both already follow the same existing rule (item 21), it just wasn't
  phrased broadly enough. "Jones" is a specific named person (already a
  distinct source in this same story) — structurally the same role a
  specific named organization plays in the org-affiliation-stripping
  rule, so it strips the same way. "San Ramon" is a generic place, not
  a specific named entity, so it stays. Checked precedent directly: all
  7 existing "friend" SD values in the corpus are bare, never "friend
  of [Name]." Drafted the broadened rule as item 30 in the Prompt
  Updates checklist. Applied bare "friend" on all 5 rows; Title and
  Source Justification unchanged.
- **Story 190 (West_Texas_Drag_Queens) rows 1, 2, 3, 4, 7, 11, 12, 13,
  18, 21, 37, 38, 39, 40 (Destiny Adams) — Title backfilled, new Source
  Descriptors added, both applied globally.** Rows 11-40 already
  correctly had Title "a former field director for the Democratic
  Party" (stated at row 11), but rows 1-4 and 7 — earlier in the
  article, before that's revealed — had it empty. Backfilled to match.
  Separately, row 7's own text ("However, parents, drag queens, and
  event organizers like Adams say drag queen story times are safe...")
  identifies her as an "event organizer" — "like Adams" attaches to
  the last item in that list, not "drag queens." Added Source
  Descriptors "event organizer" across all 14 rows, per the same
  carry-forward principle used for Title.
- **Story 190 rows 17, 19, 20 (Steve Miller) — Source Descriptors
  populated "critic," trimmed from "critic of Tumbleweed + Sage."** SJ
  already reads "Miller had been a vocal critic of Tumbleweed + Sage in
  the weeks leading up to the event." Same rule just applied to
  Orthwein above (item 30): "Tumbleweed + Sage" is a specific named
  entity (the coffee shop), strips the same way a named organization or
  person would. SD "critic"; Title ("pastor of the Temple Baptist
  Church in Lubbock") and Source Justification unchanged.
- **Story 50 (Peru_Election) rows 9.0, 10.0 (Veronica Mendoza) —
  "Vendor from Juliaca" moved from Title of Source to Source
  Descriptors.** Your draft had this correctly captured, but in the
  wrong field — "vendor" is a non-credentialing, informal occupation
  (no institutional authority), same bucket as the trades explicitly
  excluded from Title. "Juliaca" is a generic place, so it stays
  attached to the descriptor rather than getting stripped, matching the
  "resident of San Ramon" location-kept pattern. Fixed: Title cleared,
  Source Descriptors "vendor from Juliaca" (moved as one whole phrase,
  not split).
- **Story 128 (Congress_Pay_Battle) row 4 (Steny Hoyer) — corrected a
  wrong Title of Source, found during the Phase 3 single-word-SJ
  check.** Your draft had Title "Republican Representative, Oklahoma"
  — but Hoyer's own row states "Rep. Steny Hoyer (D-Md.)," a Democrat
  from Maryland. That title actually belongs to Tom Cole (row 13, a
  genuine Republican Representative from Oklahoma) — looks like a
  copy-paste error between the two rows. Fixed to "Rep. (D-Md.)",
  captured exactly as the article phrases it and matching an identical
  precedent already in GT-2026 (`31-SV-GT-vermont_bill.csv`, Rep.
  Saudia LaMont's Title is "Rep. D-Morristown," pulled the same way
  from "Rep. Saudia LaMont, D-Morristown"). Also added Source
  Descriptors "plaintiff" — this file already treats "plaintiff(s)" as
  a valid SD word for other rows (3, 8), and Hoyer's own text directly
  names him as "one of the plaintiffs." Source Justification ("plaintiff")
  left unchanged.
- **Story 189 (Cameroon_Unpaid_Wages) rows 6, 7, 8, 9, 10, 11, 13, 14,
  15, 41, 42 — "journalist"/"senior journalist" moved from Source
  Descriptors to Title of Source, applying the same rule already used
  for Mohamed Auwal in this same file consistently across everyone
  else.** Found during the Phase 3 Source Descriptors compliance audit
  — these 5 people (Yerima Kini Nsom, Moma Sandrine, Maureen Ndi,
  Dingana Raymond, Wawa Jackson Nfor) still had it in SD, an
  inconsistency with Auwal's earlier fix in this same file. Title →
  "journalist"/"senior journalist" as appropriate; existing Source
  Justification content (Nsom's "with over 25 years experience,"
  Sandrine's "who left the private sector and got into public
  service," Raymond's "in Bamenda, North-West Cameroon") left
  unchanged.
- **Story 167 (Youre_Swamped) row 5 — "journalists" moved from Source
  Descriptors to Title of Source, same rule.** "both randoms on the
  Internet and whole ass journalists—ask..." — an Unnamed Group of
  People row; matches the same credentialing treatment already
  established for occupational-group words (e.g., "Federal
  Scientists") elsewhere in this batch.
- **Story 167 (Youre_Swamped) — new row added, found during Phase 4a
  (missing Sourced Statement sweep).** "Never before had The National
  Weather Service declared a flash flood emergency in New York City."
  had no row at all. Borderline call, discussed directly: this line
  sits inside an essay-voice paragraph the author is narrating
  themselves ("Climate change is coming for everybody... Ida made that
  clear...") with no "according to NWS" framing — but "declared" is a
  speech-act verb, the same family as "announced," and a flash flood
  emergency declaration can only be attributed to NWS itself. For
  consistency in applying the Sourced Statement guidance to
  declaration-type verbs, added as its own row (only this one
  sentence, not the surrounding essay commentary): Type Named
  Organization, Name of Source "The National Weather Service" (matches
  existing corpus precedent verbatim, `168-Texas_Power_Outages.csv`).
  Added as row 6 (end of file, matching its position in the article
  text, after the existing rows 1-5).
- **Story 186 (Homeless_Camp_Sweeps) rows 36, 37, 39, 40, 42, 67, 68
  (Alfredo Reyes) and row 44 (JJ Ramirez) — org name stripped from
  Title and Source Descriptors, found during the Phase 3 Source
  Descriptors compliance audit.** Reyes's Title was "worker with the
  advocacy group VOCAL-TX" — a double violation: "worker" is a
  non-credentialing word that shouldn't be in Title at all (same rule
  as "vendor" for story 50 above), and "VOCAL-TX" is a specific named
  organization that gets stripped from an attached descriptor per the
  org-affiliation-stripping rule (used earlier this batch on Jennifer
  Orthwein's "friend of Jones" and Steve Miller's "critic of Tumbleweed
  + Sage," both above). Fixed: Title cleared, Source Descriptors →
  "worker." Ramirez's Source Descriptors was "member of VOCAL-TX" —
  same fix, → "member." **Source Justification deliberately left
  untouched on all 8 rows** — whether VOCAL-TX belongs in SJ is a
  separate, content-dependent editorial call (does the org's work tie
  directly to the specific attributed statement?), not an automatic
  consequence of stripping the org name from SD/Title. Logged as a new
  standing item (`development-of-v59.md` item 46) for a dedicated SJ
  review across all 17 "solidarity journalism initiative" stories
  (166-176, 185-190) — not run yet, and not part of this fix.
- **Story 175 (Indigenous_Health_COVID) rows 73, 74, 75 (Michael
  Weahkee) — dropped a stray leading article from Source Descriptors,
  found during the Phase 3 Source Descriptors compliance audit.** Your
  value was "an enrolled member of the Zuni Tribe" — a full corpus grep
  (GT-2026 + all of GT-II) found this was the *only* Source Descriptors
  value anywhere carrying a leading "a"/"an"; every other multi-word
  value drops it. Fixed to "enrolled member of the Zuni Tribe." Checked
  first whether "Zuni Tribe" itself should be stripped under the
  org-affiliation rule (same test as story 186's VOCAL-TX above) —
  decided no: tribal-nation membership is an identity/heritage category,
  the same bucket as "Diné" (kept bare elsewhere in this file) or a
  place name, not an organizational affiliation whose *work* is being
  cited. "Enrolled" also kept, not trimmed as decoration — formal tribal
  enrollment is a distinct legal/citizenship status from an informal
  heritage claim, a category-defining qualifier rather than a decorative
  one. Row 60 (Mary Owen, "member of Alaska's Tlingit band") reviewed
  alongside this and left unchanged — already correctly formatted, same
  reasoning applies. Source Justification unchanged on all 4 rows.
- **Story 128 (Congress_Pay_Battle) row 3 — one-word casing fix, found
  during a Phase 3 Source Descriptors casing review.** SD was
  "Plaintiffs" (capital), but the prompt's own rule is to capture SD
  exactly as the reporter wrote it in that row's own text — and this
  row's Sourced Statement has lowercase "plaintiffs" mid-sentence
  ("Many of the plaintiffs have publicly argued..."), not a
  sentence-initial capital. Likely carried forward from row 8, which
  *does* legitimately have "Plaintiffs" capitalized (its own sentence
  starts with the word). Fixed row 3 only, to "plaintiffs"; row 8
  unchanged. Checked the rest of the batch's other capitalized SD
  values (Speakers/speakers, Activists/activists, Supporters,
  Opponents, Citizens, Community members, Workers unions, "Advocates,
  homeless communities," "Formerly incarcerated trans women") against
  their own rows' text — all of those matched their own sentence's
  actual casing, so no other changes; this was the one genuine
  mismatch.
- **Story 50 (Peru_Election) — new row added, found during Phase 4a
  (missing Sourced Statement sweep).** "Such issues have gained
  traction in Peru, but polling reflects a fragmented electorate and
  an anti-establishment, anti-Congress sentiment, not a clear
  ideological shift, analysts say." had no row at all, despite being a
  distinct, separately-attributed "analysts say" claim from the one
  already captured in row 2 ("Analysts say the vote will not be driven
  by ideology"). The preceding sentence in the same paragraph
  ("Regionally, there has been a shift toward conservative, law-and-
  order politics...") is the reporter's own synthesis with no
  attribution signal, so it was left out — only the attributed
  sentence was added. New row inserted as row 9.0 (Type Unnamed Group
  of People, Title "analysts," matching row 2's exact wording), with
  the two rows after it (old 9.0/10.0, Veronica Mendoza) renumbered to
  10.0/11.0 to keep article order.
- **Story 50 (Peru_Election) — two more new rows added, same Phase 4a
  sweep, a Note 12-style split.** "She is a deeply polarizing figure:
  Supporters credit her father's government with stabilizing the
  economy and defeating violent insurgencies in the 1990s, while
  critics associate it with authoritarianism and corruption." — a
  two-sided characterization with no row at all, sitting right after
  the Keiko Fujimori polling row. Split into two rows, both carrying
  the *same full* Sourced Statement text (not divided into half-
  sentences) since the two characterizations are two sides of one
  continuous statement: Type Unnamed Group of People on both, Source
  Descriptors "supporters" / "critics" respectively (not Title —
  matches this same file's existing "critics" SD precedent at row 10,
  a stance word, non-credentialing). No Source Justification on either
  — nothing beyond what's already in the Sourced Statement text.
  Inserted as rows 5.0/6.0; everything from the old row 5.0 onward
  renumbered up by two to keep article order (final sequence: 1-13).
- **Story 68 (Georgia_Wildfires_Spreading) — new row added, found
  during Phase 4a (missing Sourced Statement sweep).** "Forecasters
  warned of 'super fog.'" had no row, sitting right after the National
  Weather Service Atlanta office's air-quality warning (row 6).
  Discussed directly whether this is really a separate attribution or
  just a continuing paraphrase of what NWS already said in the
  preceding line — resolved as its own attribution, not a merge, per
  the standing rule already established earlier in this migration
  (`development-of-v59.md` item 13): classify each attribution by what
  is actually stated at that specific point in the text, don't infer
  or merge it with an adjacent, differently-worded attribution just
  because the connection is plausible. "Forecasters" doesn't say NWS
  at that point, so it gets its own row rather than being folded into
  row 6 — matching the exact Title wording ("Forecasters") already
  used for this same word in row 1. Added as row 7 (Unnamed Group of
  People, Title "Forecasters"); everything from the old row 7 onward
  renumbered up by one to keep article order (final sequence: 1-12).
- **Story 69 (Trump_Assassination_Suspect) row 8 split into two rows,
  found during Phase 4a (missing Sourced Statement sweep) — a
  genuinely mis-combined row, not a missing one.** The old row 8's
  Sourced Statement combined two separately-attributed clauses into
  one row: "...the affidavit said" (one source) and "...according to a
  copy of the roughly 1,000-word document shared by two law
  enforcement officials who were not authorized to disclose the
  information" (a different source). Discussed directly before
  settling on the fix — first considered duplicating the full sentence
  across two rows (the Note 12 joint-attribution pattern used
  elsewhere in this batch), but the two clauses are actually two
  distinct, separable statements, not one statement jointly
  attributable to two sources, so a true split (each row gets only its
  own clause) fit better here. Split into: **row 8** — "Shortly before
  the attack, Mr. Allen sent an email to his family and a former
  employer in which he expressed deep anger at the administration and
  the president, the affidavit said." (Document, Source Descriptors
  "affidavit," Source Justification "federal affidavit" — carried
  forward from rows 4/5, the same affidavit already established
  earlier in this file). **New row 9** — "Administration officials
  were the suspect's 'targets, prioritized from highest-ranking to
  lowest,' according to a copy of the roughly 1,000-word document
  shared by two law enforcement officials who were not authorized to
  disclose the information." (Anonymous Source — this migration's
  narrow disclosure test is met here, "not authorized to disclose the
  information" is the anonymity-granted signal; Title of Source "law
  enforcement officials," matching this file's existing credentialing
  treatment of "officials"; Source Justification captured verbatim
  from the article text, "roughly 1,000-word document shared by two
  law enforcement officials who were not authorized to disclose the
  information."). Everything from the old row 9 onward renumbered up
  by one (final sequence: 1-18).
- **Story 128 (Congress_Pay_Battle) row 13 (Tom Cole) — merged a setup
  sentence into the existing quote row, found during Phase 4a.** "House
  Appropriations Chair Tom Cole (R-Okla.) cast doubt, however, on any
  immediate impact." had no row, but discussed directly whether it
  needed its own row or was the same source's material — checked and
  confirmed it's in the exact same unbroken paragraph as the existing
  quote row, immediately adjacent, same person, no other source in
  between. Merged into row 13 rather than added as a separate row:
  Sourced Statements now reads "House Appropriations Chair Tom Cole
  (R-Okla.) cast doubt, however, on any immediate impact. 'We don't
  know anything really about the opinion yet,' Cole cautioned." Also
  moved "House Appropriations Chair" from Source Justification into
  Title of Source (it's literally how the article introduces him at
  that point, not justification text), and added the party/
  constituency code the article also gives right there — "House
  Appropriations Chair (R-Okla.)" — matching this same file's own
  precedent for Hoyer (Title "Rep. (D-Md.)," row 4).
- **Story 128 row 12 (Steny Hoyer) — fixed a second instance of the
  same copy-paste error already caught once on row 4 (Phase 3c, item
  45).** Found while double-checking row 13's new Title against every
  other Hoyer/Cole row in the file. Row 12 still had Title "Republican
  Representative, Oklahoma" — that's Cole's title, not Hoyer's (Hoyer
  is a Democrat from Maryland); row 4 had the identical error and was
  already fixed to "Rep. (D-Md.)," but row 12 was missed at the time.
  Fixed to match: "Rep. (D-Md.)." Source Descriptors/Justification
  unchanged.
- **Story 174 (PPP_Loans_Low_Income) — unsuspended, resumed with Phase
  4a/4b first, out of the normal phase order (2026-09-23).** Since
  `174` was the very file that originally surfaced the Phase 4b
  pattern (item 44), and since the whole 166-190 range is being sent
  back to the annotator anyway, the user chose to run Phase 4a/4b on
  this file before returning to the paused Phase 2.6 work — trimming
  the row set to what should actually be there before polishing
  Title/SD/SJ on rows that might get dropped. **Phase 4a: no missing
  rows found.** **Phase 4b: 12 rows removed** — every one either had no
  attribution verb at all and narrated a source's own inner/subjective
  experience as apparent fact (the item-44 pattern), or contained one
  such clause inside an otherwise-factual row:
  - Rows 1-2 (KB Brown / Katie Brown): "As the pandemic hit, orders
    cratered... **They knew** they would need help..."
  - Rows 3-4: "...they never got one of the forgivable COVID-19
    loans... **still don't fully understand** what went wrong in a
    process that **felt** Byzantine and inequitable to them."
  - Row 26 (Rosa Daniels): "...**Not sure what she would do** if she
    didn't get the loan, Daniels tried another institution..."
  - Row 31 (Rosa Daniels): "**What made the uncertain wait for a PPP
    loan so stressful for Daniels and her daughter** was that it came
    as the bakery struggled from a one-two punch..."
  - Rows 40-41 (KB Brown / Katie Brown): "The Browns, **unsure what was
    happening, thought** the funding would eventually arrive..."
  - Rows 46-47: "**From the Browns' perspective**, the PPP was opaque
    and confounding..."
  - Rows 56-57: "...**Because the Browns thought they'd been
    approved**, they brought back two furloughed employees..."
  Considered and left alone (factual/procedural narration about the
  same sources, not inner-state — same bucket as an already-rejected
  Phase 4a conduct-narration candidate on a different file): "Wolfpack
  Promotionals saw business dwindle... tried their bank... applied to
  PayPal and Square" (old rows 36-37, now the surviving row 30-31).
  Row numbers renumbered sequentially after removal (61 rows → 49).
  **Heads-up from the user**: they separately asked you to consider
  lived-experience content like this as possible Source Justification
  material when you re-review these stories (due back Friday) — so
  some of what was just removed here may resurface as new SJ text on
  the revised file. When that comes back, this migration will diff the
  revision against the already-migrated CSVs rather than re-migrating
  from scratch (see `development-of-v59.md` item 44's workflow note).
- **Story 174 row 46 (renumbered, KB Brown) — trimmed one unattributed
  sentence, a borderline case reviewed separately after the main
  removal batch above.** "Then both got COVID-19. They're still
  recovering, trying to keep the shop going with orders they can
  scrounge together. Business is down 85% to 90%, maybe more, KB Brown
  estimated. **He's frustrated big companies that didn't seem to need
  help got it anyway this year while some very small firms like his
  were left out.**" — the last sentence has no attribution verb of its
  own and could be read either as an implied continuation of "KB Brown
  estimated" (same sentence-group, immediately adjacent) or as another
  instance of the same unattributed inner-state pattern just removed
  elsewhere in this file ("frustrated" is exactly this kind of
  language). Trimmed per the user's call, favoring the same
  precision-over-recall default already established for Phase 4a's
  implied-continuation checks.
- **Story 174 — the original KB Brown/Katie Brown Title question,
  finally resolved: "owner of Wolfpack Promotionals," backfilled
  across all rows for both (2026-09-23).** This was the question that
  paused the file back on 2026-09-21 (see the entry above). Discussed
  the sourcing directly before applying: the article never states
  "owner of Wolfpack Promotionals" as one phrase — it says "KB and
  Katie Brown, the owners of the Minneapolis print shop" in one
  sentence, then names the shop "Wolfpack Promotionals" as the subject
  of the very next sentence, same paragraph. Confirmed this is the
  same kind of adjacent-sentence resolution already accepted elsewhere
  in this migration (the Ilya Sutskever "another member" -> "board
  member" case), not the kind of distant cross-article inference this
  migration deliberately excludes (item 13). Applied to every KB
  Brown/Katie Brown row (current numbering): KB Brown — 1, 30, 36, 46,
  47; Katie Brown — 31, 32, 33, 37, 48, 49. Source Descriptors/
  Justification unchanged on all 11 rows.
- **Story 174 rows 20-21 (Rosa Daniels) — Source Descriptors "Black"
  backfilled backward from row 23, found during Phase 2.6.** Row 23
  (later in the article) has SD "Black" (from "said Daniels, who is
  Black"); rows 20-21 (earlier) had it blank. Applied the same
  both-directions backfill logic already established for Title of
  Source (item 28) — a stable identity fact isn't moment-specific, so
  it should backfill backward the same way a title does, not just
  forward. Backfilled SD "Black" onto rows 20-21. **A parallel
  question — whether row 7's SJ ("whose own firm received a PPP
  loan," Anirban Basu) should backfill backward to row 6 the same
  way — is paused, not applied.** User wants to check how backward SJ
  backfill interacts with the existing ";"-concatenation convention
  for combining multiple Source Justification facts about the same
  source, before deciding; needs a precedent search first. Both rows
  (6, 7) left as-is for now.
- **Story 172 (Delhi_Flyover_Homeless) — new row added, found during
  Phase 4a (missing Sourced Statement sweep).** "'Sometimes, when
  there is no money, we don't cook at all,' Shahida says. 'Those days
  we either stay hungry or rely on alms from passersby.'" had no row
  at all — a direct quote sitting between the two Nikita Popat rows
  (17, 18 at the time), interleaved in the article's own paragraph
  order. Added as row 18 (Named Person, Shahida), pushing the second
  Popat row and everything after down by one. Source Justification
  carried forward from row 5 (Shahida's first quote), matching how
  this same file already treats her later rows (e.g. row 19/20) under
  the currently-active forward-carry-forward convention (item 48's
  proposed row-level policy hasn't been adopted yet).
- **Story 100 (OpenAI_Court) — unblocked and cleaned up, first pass
  (2026-09-24).** This story was blocked on text since the original
  delivered PDF (an MSN embed) was image-only; the user supplied a new
  Reuters PDF with real extractable text, which was converted and
  added to `extracted_articles_boilerplate/`. Checking the existing
  15-row draft (annotated by reading the image PDF directly, since no
  extractable text existed at the time) against the real Reuters text
  found two things: (1) **row 15**, "Jobs focused on AI are growing,
  according to LinkedIn." — does not appear anywhere in the actual
  8-page Reuters article; looks like a stray MSN sidebar/related-
  content snippet that got mixed into the manual annotation. Dropped.
  (2) OCR-style typos carried over from the original manual read:
  "OpenAl" → "OpenAI" (rows 1, 5, 8, 10 at the time — capital I
  misread as lowercase l) and "lowa" → "Iowa" (row 13's attorneys-
  general list). Fixed throughout. 15 rows → 14, renumbered
  sequentially. Full Phase 0-4 migration for this file to follow.
- **Story 100, Phase 0/2 continued — three schema-violation fixes.**
  (1) Rows 1-2 had a generic genre word, "Court Document," sitting in
  Document's Name of Source field — checked the whole corpus (GT-2026
  + GT-II) and confirmed no other row anywhere does this; Name of
  Source is reserved for an actual stated document title, a generic
  genre reference belongs in Source Descriptors instead. Moved: Name
  cleared, SD "court document" on both rows. (2) The old row 6
  actually combined two separate document references into one row —
  "The document also revealed Altman had sold off a stake in Reddit
  by the end of 2025" (continuing the same court document from rows
  1-2) and "...according to SEC filings at the time" (a distinct,
  generic reference for a different fact). Split into two rows: the
  court-document continuation (SD "court document") and a new row for
  the SEC filings reference (SD "SEC filings," Name blank, same fix as
  item 1). (3) Row 13 (Unnamed Group of People) had "attorneys
  general" sitting in Name of Source — a schema violation, since UGOP
  rows can never carry Name of Source. Moved to Title of Source
  (credentialing — a formal legal/government position). Also added
  Source Descriptors "Republicans" from the same row's own text ("all
  Republicans"), trimming the quantifier "all" per the established
  atomic-word convention. 14 rows → 15 (the split added one row).
- **Story 100 — PAUSED mid-Phase-4a, resume next session.** Six more
  candidates found re-checking the article paragraph-by-paragraph
  (more than any other file in this batch, expected since this was
  manually annotated from an image-only PDF originally). Full detail,
  exact proposed text, and confidence levels are in
  `development-of-v59.md`'s item 42-adjacent log (the "PAUSED
  2026-09-24, PICK UP HERE" entry right after the three schema-
  violation fixes above) — not repeated here to avoid drift between
  the two copies.
- **Story 100, resumed 2026-09-25 — candidate A applied.** "Ten U.S.
  attorneys general... asked the U.S. Securities and Exchange
  Commission to scrutinize documents from OpenAI ahead of an expected
  initial public offering, and the U.S. House of Representatives'
  Committee on Oversight and Government Reform last week asked Altman
  for information on OpenAI policies to prevent conflicts of
  interest." Split into two rows (same "genuinely separable
  statements, not a joint one" test as story 69's row 8 split):
  Unnamed Group of People "attorneys general" (Title "attorneys
  general," SD "Republicans," carried forward from the same group's
  established row) and Named Organization "U.S. House of
  Representatives' Committee on Oversight and Government Reform."
  Kept the article's own lowercase "the" at the start of the second
  row's text rather than capitalizing it for readability, per the
  user's explicit call to match the article exactly. Inserted after
  row 3; everything renumbered (15 rows → 17).
- **Candidate B, resolved as an SJ addition rather than a new row.**
  "In court on Tuesday, Musk's lead trial lawyer Steven Molo exhibited
  a document that revealed Altman's holdings in nine companies that
  had done business with OpenAI and their fair market value as of
  December 31, 2025." User caught that "exhibited a document" is
  conduct narration (same exclusion as story 166's rideshare-drivers
  case, story 185's row 50), not a statement — but the content usefully
  explains why the "court document" is in the story, which is exactly
  what Source Justification is for. Added verbatim as SJ to all three
  rows that reference this same document (rows 1, 2, 8) — not row 9,
  which is the separate "SEC filings" reference.
- **Candidate C, applied as a new Document row, with the same SJ carried
  onto it.** "The list of companies with OpenAI deals included a $1.7
  billion stake in fusion power company Helion Energy, a $633 million
  stake in financial software company Stripe, and $258 million in
  anti-aging pharmaceutical company Retro Biosciences, all of which have
  OpenAI deals." This is the same court document's list continuing
  (row 8 itself opens "The other companies on the list included...").
  Type: Document, Name/Title: null, SD: "court document". Per the
  user's call ("the SJ seems like it will apply here — it's the same
  document"), the same Molo-exhibited-document SJ text used on rows
  1/2/8 was added here too. Inserted after the Forbes row (old row 7),
  before "The document also revealed..." row (old row 8); everything
  renumbered (17 rows → 18).
- **Candidate D, applied as a new Document row, SJ left null.** "The
  other companies on the list included chip maker Cerebras, people
  management software maker Degree, which is known as Lattice, AI
  device maker Humane, AI software maker Software Applications and AI
  pharmaceutical company Trialspark, now known as Formation Bio."
  Same court-document company list continuing (matches rows 8/9/10,
  all also SJ-null). Type: Document, Name/Title: null, SD: "court
  document", SJ: null — the SJ call here is deferred pending the
  still-undecided row-level SJ policy revision (item 48; no decision
  yet on whether/how the Molo SJ text should extend to this row).
  Inserted after the SEC-filings row (old row 10), before Altman's
  Helion-friendship testimony (old row 11); renumbered (18 rows → 19).
- **Candidates E and F, dropped — no CSV change.** E ("The company,
  which is aiming to build the world's first fusion power plant, does
  not generate any revenue but has been valued in the private market
  at $5.4 billion.") and F ("Helion first signed an agreement to
  secure future energy for OpenAI in 2024. Altman stepped down from
  Helion's board in March 2026, as the companies explored a larger
  deal.") were both judged weak SS candidates by the reviewer and
  dropped without being added. This closes Phase 4a for story 100 —
  all six candidates (A-F) resolved.
- **Phase 3 (compliance audit) and Phase 4b (unattributed-narration
  check), both run 2026-09-25 — clean, no changes on either.** Phase 3:
  SJ substantive not medium-of-contact; "Musk's lead trial lawyer"
  judged credentialing not bare-relational; no single-word SJ; Source
  Descriptors ("court document," "SEC filings," "Republicans") atomic/
  exact/consistent with corpus convention; no Title qualifier-stacking.
  Phase 4b: every one of the 19 rows traces to an explicit manifest
  attribution in the article text — none read as unattributed reporter
  narration. **Story 100 is now fully done, Phases 0-4, 19 final rows.**
- **`77-SF_Drug_Detention.csv` rows 7/8 — copied link label removed,
  2026-10-07.** Row 7's Sourced Statement and Source Justification, and
  row 8's Source Justification, read "signed the legislation(opens in
  new tab) inside City Hall" — "(opens in new tab)" is the website's
  accessibility label on a hyperlink, not article text, carried over
  when the sentence was copied. Changed to "signed the legislation
  inside City Hall" in all three cells, matching the article. Found
  during the corpus text sweep (`corpus-text-sweep-2026-10-07.md`); a
  scan of every GT CSV for similar copied web clutter found no other
  cases.
- **`69-Trump_Assassination_Suspect.csv` row 1 removed, 2026-10-08 (18 ->
  17 rows, renumbered).** New standing rule, adopted 2026-10-08: annotate the article body only -- photo captions and scrolling-graphic text panels count as body; subtitles (the summary line under the headline) and summary boxes don't, since they recap the body. Row 1 ("The authorities say the
  California man stormed a black-tie gala on Saturday seeking to kill
  the president.") annotated the article's subtitle. Nothing lost: row
  2 (now row 1) covers the same claim from the body ("A California man
  who the authorities say ran through a security perimeter and fired a
  gun outside a packed black-tie gala...").
- **Graphic-derived rows moved to the graphics side file, 2026-10-08.**
  New standing rule, adopted 2026-10-08: we only annotate text the models actually receive -- article body, photo captions, and scrolling-graphic text panels. Text that belongs to a chart, map, table, or an image of a document isn't part of that, so rows annotating it were moved (not deleted) to `GT-II/graphics-only-rows.csv`, kept in case we ever evaluate models on page images. `113-California_Homes_Uninsured.csv` rows 84 and 85 (the two
  Farmers Insurance emails, which appear in the article only as
  screenshots: "Hi Joel. We finally got the list..." and "Jon, CDI
  customer complaints...") -- 99 -> 97 rows, renumbered.
- **Copy errors fixed, 2026-10-08 (corpus text sweep, Tier 4).**
  `58-Louisiana_Shooting.csv` row 5: the first sentence ("The police
  identified the gunman as Shamar Elkins, 31, and said he was the father
  of seven of the eight dead children.") had been pasted twice -- the
  duplicate removed. `121-Meta_Job_Cuts.csv` row 5: "internal Al tools"
  (lowercase L) -> "internal AI tools," matching the article.
- **`72-Quantum_Campus.csv` row 14: duplicated sentence removed,
  2026-10-08.** "After residents rebelled at the Bowen meeting, Moreno
  said, the city stopped holding public meetings." had been pasted twice,
  run together; one copy kept. The row now matches the article exactly.
- **Rows split at paragraph breaks, 2026-10-08.** New rule (adopted
  today): all statements from the same source within one paragraph go
  in one row, a new paragraph starts a new row, and a reporter's own
  sentence that isn't from the source is left out. Sentences are always
  copied whole and exact. `63-Maine_Data_Center.csv` row 1 spanned two
  paragraphs: it now ends at "...to go ahead.", and the second Mills
  quote ("This project — ... Mills wrote in the press release announcing
  her decision.") is a new row 2 with the same Type/Name/Title/SJ (9 ->
  10 rows, renumbered). `72-Quantum_Campus.csv` row 26 spanned two
  paragraphs: it now ends at '..."lakefront," he said.', and "He said
  Friends of the Parks is part of the coalition demanding a CBA; it has
  taken a neutral position on the quantum park." is a new row 27 (38 ->
  39 rows, renumbered). The three sentences about Friends of the Parks'
  demands in between are the reporter's framing and stay out.
- **Invisible/odd characters cleaned up, 2026-10-08 (no visible text changed; makes rows match the article text exactly): non-breaking spaces -> normal spaces, non-breaking hyphens -> normal hyphens, the "ʼ" modifier apostrophe -> "’", zero-width characters removed, and the "ﬁ" ligature -> "fi". These usually come from copying text out of web pages or PDFs.** Files: `55-Pentagon_Lab.csv`, `56-Philz_Pride_Flags.csv`.
- **`57-Trump_Jesus_Memes.csv` row 4 reclassified, 2026-10-10 (Phase 1).**
  Anonymous Source -> Unnamed Group of People. The source is "some of
  Mr. Trump's advisers", and the article never says they were granted
  anonymity (no "spoke on condition of anonymity", "not authorized to
  speak", "asked not to be named", or similar). We use Anonymous Source
  only when the article states that anonymity was granted. Name and Title,
  which held the literal word "null", are now blank (the convention for
  empty fields).
- **Type of Source spellings fixed and a joint row split, 2026-10-10.**
  `58-Louisiana_Shooting.csv` rows 1, 3, 5, 6, 11: "Unnamed group of
  people" -> "Unnamed Group of People" (capitalization only).
  `62-D4vd_Murder_Charges.csv` row 4: was one row typed "Named Group"
  (not one of the six types) crediting three lawyers together ("Blair
  Berk; Marilyn Bednarski; Regina Peter"). Split into three Named Person
  rows (now rows 4-6), one per lawyer, same Sourced Statement, Title
  "lawyer" (the article says "the lawyers, Blair Berk, Marilyn Bednarski
  and Regina Peter"), Source Justification "Lawyers for Mr. Burke" (20 ->
  22 rows, renumbered). When one statement is credited to several named
  people, each gets their own row (system prompt Note 12).
- **Unnamed Group of People rows tidied, 2026-10-10 (Phase 2.1).**
  `57-Trump_Jesus_Memes.csv` row 4: Title "advisers" added ("some of
  Mr. Trump's advisers" -- "Mr. Trump's" left out since it names a
  specific person). `58-Louisiana_Shooting.csv` row 11: "Elkin's family
  members" moved out of Title of Source into Source Descriptors as
  "family members" -- a family relationship describes the source but
  isn't a credentialed title ("Elkin's" also wasn't in the text).
  `62-D4vd_Murder_Charges.csv` row 9: the literal word "null" in Name of
  Source and Source Justification replaced with blanks.
- **Document rows tidied, 2026-10-10 (Phase 2.4 Pass 1).** A Document
  never has a Title of Source, and Name of Source is only the document's
  own title when the reporter states one; a general word for the kind
  of document goes in Source Descriptors, as written in the sentence.
  `58-Louisiana_Shooting.csv` rows 9 and 14: Title "Police records" /
  "Records" -> blank, Source Descriptors "Records"; row 10: Title
  "Police records" -> blank, Source Descriptors "police description".
  `62-D4vd_Murder_Charges.csv` rows 11 and 13: Name "Felony complaint"
  -> blank, Source Descriptors "felony complaint" (row 11) and
  "complaint" (row 13), each as worded in its sentence; the literal word
  "null" in Title and Source Justification replaced with blanks.
- **Named Organization rows tidied, 2026-10-15 (Phase 2.5 Pass 1).**
  A Named Organization never has a Title of Source, and its name follows
  the article's wording. `57-Trump_Jesus_Memes.csv` row 7: Title "CNN
  website" removed. `58-Louisiana_Shooting.csv` row 8: "The Army" ->
  "Army". `62-D4vd_Murder_Charges.csv` rows 3, 8, 10: the literal word
  "null" in Title and Source Justification replaced with blanks (names
  kept; "Los Angeles County district attorney's office" follows the
  article's own identification of Mr. Hochman as the Los Angeles County
  district attorney). `98-AV_Trump_China.csv` row 5: "Commerce
  Departments Bureau..." -> "Commerce Department's Bureau of Industry
  and Security", as written.
- **Named Person rows, first check, 2026-10-15 (Phase 2.6 Pass 0).**
  `57-Trump_Jesus_Memes.csv` (8 rows) and `62-D4vd_Murder_Charges.csv`
  (12 cells across its Named Person rows): the literal word "null"
  replaced with blanks. `54-Coachella_Influencer.csv` rows 8-9: Title
  typo "Beauty Conent Creator" -> "Beauty Content Creator".
  `58-Louisiana_Shooting.csv`: row 2 Title typo "Shreveport Major" ->
  "Shreveport Mayor"; old row 12 credited two people in one row
  ("Mahelia Elkins and Mr. Jackson") -- split into one row each (now
  rows 12-13), each person's relationship in Source Descriptors
  ("mother", "stepfather") rather than in Title, and the opening
  sentence ("On Easter Sunday, Mr. Elkins called his mother...") left out
  of the Sourced Statement, since it's the reporter's own context, not
  something attributed to them; old row 15 (now 16): "Mr. Jackson" ->
  "Marcus Jackson" (his full name is in the article), Title "Elkin's
  stepfather" -> blank, Source Descriptors "stepfather".
- **Name of Source "Donald Trump" -> "Trump", 2026-10-15.** Name of Source is the person's full name *as stated in the article*. Where the article only ever says "Mr. Trump", "President Trump" or "Trump" -- never "Donald Trump" -- the Name is "Trump".
  `57-Trump_Jesus_Memes.csv` rows 1, 6, 14.
- **`58-Louisiana_Shooting.csv` row 14: source is Shamar Elkins,
  2026-10-15.** "Mr. Elkins told them through tears that he wanted to
  take his own life. He told Mr. Jackson that..." -- the person speaking
  here is Shamar Elkins (the article names him: "The police identified
  the gunman as Shamar Elkins"); Mr. Jackson is only who he told. It had
  been credited jointly to "Mahelia Elkins and Mr. Jackson".
- **Titles, 2026-10-16 (Phase 2.6 Pass 1).** A title must be stated
  for the person in the article, and it names a position of authority or
  expertise; occupations and descriptions go in Source Descriptors.
  `57-Trump_Jesus_Memes.csv`: rows 1, 6, 14 "President of the United
  States" -> "President" (the article says "President Trump"); rows 8-9
  "a longtime Republican pollster" -> "Republican pollster" ("longtime"
  isn't part of a title); row 10 "a former White House press secretary
  for Mr. Trump" -> "former White House press secretary".
  `98-AV_Trump_China.csv` row 4: "President, United States" ->
  "President". `58-Louisiana_Shooting.csv` row 7: Title "UPS employee"
  removed (the article says he "used to work at UPS"; that's not a
  title). `54-Coachella_Influencer.csv`: "content creator" is an
  occupation, not a credentialed title, so for rows 1-2, 6-10, 13-16 it
  moved from Title to Source Descriptors ("content creator", "beauty
  content creator", and for Cathie O'Brien "human resources
  professional, content creator").
- **Missing titles added, 2026-10-16 (Phase 2.6 Pass 2).**
  `62-D4vd_Murder_Charges.csv` rows 7 (Kent Schaffer, "a lawyer for Mr.
  Burke's family") and 21 (Patrick Steinfeld, "a lawyer for the
  family"): Title "lawyer". The title is the role itself; who the lawyer
  represents is a client relationship, not part of the title.
- **Source Descriptors added, 2026-10-16 (Phase 2.6 Pass 3).** Words the
  reporter uses to describe the person, taken from the row's own text.
  `62-D4vd_Murder_Charges.csv` row 22: "father" (Jesus Rivas, "her
  father"). `57-Trump_Jesus_Memes.csv` row 15: "onetime candidate" (Sarah
  Palin, "onetime candidate for vice president").
  `54-Coachella_Influencer.csv` rows 1, 2, 10: "content creator, fan"
  (Cedoni Francis, "a die-hard fan of Mr. Bieber").
- **Descriptors from anywhere in the article, 2026-10-16.** A word the
  reporter uses to describe a person counts for all of that person's rows,
  even when it appears in a sentence that isn't one of their sourced
  statements. `58-Louisiana_Shooting.csv` row 14: Source Descriptors
  "gunman, father" (Shamar Elkins: "The police identified the gunman as
  Shamar Elkins, 31, and said he was the father of seven of the eight dead
  children").
  `122-SJ_Climate.csv` (from the candidate-list sentence, "...serial
  entrepreneur Mike Hennessy, longtime city employee Gordon Chester,
  licensed marriage and family therapist Genny Altwer and tech worker Rick
  Ator"): rows 15-16 Source Descriptors "District 9 candidate" (Hennessy,
  matching the other candidates); row 9 Title "licensed marriage and
  family therapist" (Altwer); rows 13-14 Source Descriptors "District 9
  candidate, city employee" (Chester; "longtime" left out, it only says
  how long); rows 7-8 Source Descriptors "District 9 candidate, tech
  worker" (Ator; an occupation, not a title).
- **Source Justification, 2026-10-16 (Phase 3).** Source Justification
  says why the source matters to the story -- not how the reporter got
  the statement (a news conference, an interview), and not an unrelated
  sentence. `58-Louisiana_Shooting.csv` row 2: blanked ("described the
  killings in a news conference"); row 7: "who used to work at UPS with
  Mr. Elkins, said in an interview with The Times" -> "who used to work
  at UPS with Mr. Elkins"; row 11: blanked ("according to family members
  who spoke with The Times." -- the attribution itself; "family members"
  is already in Source Descriptors). `57-Trump_Jesus_Memes.csv` row 7
  (CNN): blanked ("A conga line of reporters also call and ask him..." is
  about reporters calling Mr. Trump, not about CNN).
  `62-D4vd_Murder_Charges.csv` row 18: Name and Title were the literal
  word "null" -- now blank.
- **Title as stated, 2026-10-16 (Phase 3).** `58-Louisiana_Shooting.csv`
  row 2: "Shreveport Mayor" -> "Mayor of Shreveport". The article says
  "Mayor Tom Arceneaux of Shreveport"; the title keeps the city (the place
  he holds the office), in the article's own word order.
- **Missed statements and paragraph splits, 2026-10-16 (Phase 4a).**
  `54-Coachella_Influencer.csv` (now 22 rows; row numbers below are the
  new ones):
  - New row 1, Cedoni Francis: "Now a full-time content creator with
    over 250,000 followers on TikTok, Ms. Francis said she had previously
    received invitations to attend the music festival, held each spring
    in the Southern California desert, from brands a few months before
    the event." -- it says "Ms. Francis said", but no row had it.
  - New row 2, Cedoni Francis: "When she didn't have any offers by March,
    she took matters into her own hands, posting a six-slide presentation
    on TikTok pitching herself and her talents, using marketing skills
    honed at a previous job at Google." -- a separate row from row 1
    because it reports her TikTok post (which the reporter could see),
    not what she told the reporter; same as Katie Begley's row 18.
  - Source Descriptors on all five of her rows (1, 2, 3, 4, 14) is now
    "content creator, fan, graduate student": the article calls her "a
    graduate student at Columbia Business School", and a describing word
    counts for all of a person's rows even when it isn't in the row's own
    text.
  - New row 13, Unnamed Group of People (Source Descriptors
    "influencers"): "In the weeks leading up to the festival, several
    influencers described on TikTok having been invited and then
    subsequently uninvited on brand trips to Coachella."
  - New row 18, Katie Begley (Source Descriptors "content creator"): the
    paragraph "Katie Begley, a content creator better known as Popstar
    Katie, posted a TikTok in which she laid out her plans for such a
    trip. She bought her flights, ... she was still looking for a place to
    stay." -- it reports what her TikTok said, so it's her statement, not
    only background (the same text also stays as row 19's Source
    Justification).
  - One row per paragraph: Shannae Ingleton Smith's old row 4 is now rows
    6 and 7 ("It's a 'rite of passage,' she said..." is its own
    paragraph); old row 11 is now rows 15 and 16 ("...chalked it up to
    'Coachella fatigue.'" ends Ms. Francis's paragraph; the "Brands have
    picked up..." quote starts a new one).
  - Not annotated: the top photo caption, which only repeats part of Ms.
    Ingleton Smith's row 16 quote (credited there to "one influencer
    manager"). A caption that just repeats a statement from the body
    isn't annotated.
- **Missed statement and exact copying, 2026-10-16 (Phase 4a).**
  `57-Trump_Jesus_Memes.csv` (now 16 rows; numbers below are the new
  ones): new row 1, Trump: "In a 12-hour span this week, President Trump
  promised that the war with Iran was ending soon. He threatened to fire
  the chair of the Federal Reserve." -- "promised" and "threatened"
  report things he said. Left out of the row: "He picked a fight with the
  pope on social media." (the reporter describing his behavior, not
  something he said) and "He posted an illustration of himself..." (the
  same post row 2 already covers). Row 6: "reshoring" -> "re-shoring" (as
  in the article); row 9: quotation marks restored around "He has always
  been like this,"; row 14: quotation marks restored around "When you're
  wrong, ... just get past it,".
  Row 5 (Mr. Trump's advisers): Source Justification "some of Mr.
  Trump's advisers" blanked -- it only repeats who the source is (the
  attribution and the Title), not why they matter.
- **Name of Source cleared for unnamed sources, 2026-10-23.** Unnamed Person and Unnamed Group of People sources have no Name of
  Source; the words that had been put there now go in Title of Source if
  they name a role with authority or expertise (officials, regulators,
  economists, officers, leaders), and in Source Descriptors otherwise
  (residents, advocates, organizers, investors).
  `72-Quantum_Campus.csv`: row 3 -> Source Descriptors "neighborhood
  watchdog groups"; row 7 -> Title "community leaders", Source
  Descriptors "Southeast Side residents, environmental justice
  advocates"; row 12 -> Source Descriptors "Southeast Side organizers";
  row 22 -> Source Descriptors "legal assistant"; row 33 -> Title "state
  regulators"; row 34 -> Source Descriptors "resident"; row 38 -> Source
  Descriptors "advocates, residents". `80-Quantum_Medicine.csv` row 3:
  Name "investors" -> Source Descriptors "investors" (this row will be
  removed later: its Sourced Statement isn't a valid one -- see the
  Sourced Statement validity sweep).
- **Missing Type of Source, 2026-10-23.** `116-Trump_Christian_Independance_Day.csv`
  row 6 ("And Interior and other federal agencies have awarded no-bid
  contracts ... citing an 'urgency' exemption...") had no Type and no
  Name. Now Named Organization, Name "Interior Department" (the article
  names it in the sentence before). The reason the agencies cite comes
  from Interior; "other federal agencies" aren't named, so they aren't a
  separate row.
- **Missed statements, 2026-10-30 (Phase 4a).** `58-Louisiana_Shooting.csv`
  (now 21 rows; row numbers below are the new ones). This is a
  live-updated New York Times explainer ("Published April 20, 2026,
  Updated April 21, 2026"), and our text is the April 21 version. The
  original annotation (workbook sheet "58 - Sydney - Manual") stops at
  Marcus Jackson's quote and words one sentence differently ("The Times"
  for "The New York Times"), so it was most likely done on an earlier
  version of the article. These rows are added so the annotation matches
  the text we use; they don't reflect a careless annotation.
  - New row 3, Unnamed Group of People (Title "Police"): "The killings
    occurred during a shooting rampage that spanned at least three
    locations ..., the police said. Officers responded ... Later, they
    said that there was another crime scene at a second residence."
  - New row 8, Unnamed Group of People (Source Descriptors "relatives"):
    "Mr. Elkins's relatives said he worked for UPS."
  - New row 19, Betty Walker: "Betty Walker — the woman who had raised
    Mr. Elkins as her son, though she was not his biological mother —
    said on Monday that three years ago, she witnessed a disagreement
    between the gunman and Ms. Pugh, his wife-to-be." (Source
    Justification: "the woman who had raised Mr. Elkins as her son,
    though she was not his biological mother").
  - New row 20, Shaneiqua Pugh (Source Descriptors "wife"): "While
    their daughters played outside, Ms. Pugh said that she was
    considering leaving Mr. Elkins and taking her children with her."
    The statement is Ms. Pugh's, though Ms. Walker witnessed it, the same
    way row 16 credits Mr. Elkins for words his parents relayed.
  - New row 21, Betty Walker: "Ms. Walker said he became furious and
    declared that if they tried to leave, 'I'll kill you, my kids and
    myself.'"
  - Row 9 (Willie Vasher): "in an interview with The Times" -> "in an
    interview with The New York Times", as in the article.
- **Paragraph boundaries and exact copying, 2026-11-02 (Phase 4a).**
  `62-D4vd_Murder_Charges.csv` (now 25 rows; numbers below are the new
  ones). A Sourced Statement holds one paragraph; a new paragraph starts
  a new row.
  - Rows 4-9: the lawyers' two paragraphs had been one row per lawyer.
    Now each lawyer (Blair Berk, Marilyn Bednarski, Regina Peter) has a
    row for "Lawyers for Mr. Burke have said they will 'vigorously defend
    David's innocence.'" (rows 4-6) and a row for "'The actual evidence
    in this case...' the lawyers... said in a statement on Monday." (rows
    7-9).
  - Row 12 (officials): added the sentence before it in the same
    paragraph, "The case has captivated the public since the authorities
    announced in September that they had discovered human remains in a
    Tesla registered to Mr. Burke." -- the authorities and the officials
    are the same unnamed group.
  - Row 2: the opening sentence ("Because of the special circumstances,
    the charge carries a maximum sentence...") left out -- it's the
    reporter's own explanation before Mr. Hochman's statement; the row is
    now "Mr. Hochman said his office had not decided whether to pursue
    the death penalty."
  - Row 14: added the paragraph's last sentence, "At the time, she was 13
    years old." (it continues the complaint's account).
  - Row 17: quotation marks restored around "A parent's nightmare,".
- **One row per paragraph, 2026-11-02 (after the text re-flow).** Once
  the PDF-converted story texts were re-flowed to one paragraph per line
  (see `extracted_articles_boilerplate/REFLOW_2026-11.md`), two rows turned
  out to span two paragraphs; each is now two rows.
  `57-Trump_Jesus_Memes.csv` (now 17 rows): Stephanie Grisham's row 11 ->
  row 11 ("He'll double down... said Stephanie Grisham, a former White
  House press secretary for Mr. Trump.") and row 12 ("But she added:
  'He's being erratic, even for him.'"). `128-Congress_Pay_Battle.csv`
  (now 18 rows): Tom Cole's row 13 -> row 13 ("House Appropriations Chair
  Tom Cole (R-Okla.) cast doubt, however, on any immediate impact.") and
  row 14 ("'We don't know anything really about the opinion yet,' Cole
  cautioned.").
- **Row removed, 2026-11-02 (Phase 4b).** `62-D4vd_Murder_Charges.csv`
  old row 21 removed (now 24 rows): "In November, a judge granted a
  request from the Los Angeles Police Department to place a security hold
  on a report from the medical examiner." No one is quoted or cited; it's
  the reporter stating what a court did. The medical examiner's report is
  what the hold applies to, not where the sentence comes from. The rest
  of that paragraph (Mr. Hochman saying the report would be released
  soon) is already row 21 now.
- **Source Descriptors filled on every row for the same source, 2026-11-05.**
  A source's Source Descriptors now apply to all of that source's rows in
  a story, not only the row whose sentence states the descriptor.
  `168-Texas_Power_Outages.csv` rows 11-14 and 29-33 (Brianna Blake):
  blank -> "mother" (as on rows 10 and 28). `166-DC_Essential_Workers.csv`
  rows 1-4, 15, 16 (Tamara Keitt-Barnes): -> "nanny, Prince George's County
  resident" (rows 1 and 15 had "nanny, resident" and "nanny"; the others
  were blank; "Prince George's County resident" is how the article
  describes her). `122-SJ_Climate.csv` rows 10-12 (Scott Hughes): blank ->
  "District 9 candidate", matching SZ's copy. `128-Congress_Pay_Battle.csv`
  row 12 (Steny Hoyer): blank -> "plaintiff" (as on row 4).
  `175-Indigenous_Health_COVID.csv` row 78 (Phillip Smith): blank -> "Diné"
  (as on rows 79 and 80).
- **Organization category words filled from the article, 2026-11-05.**
  When the article calls an organization by a category word, that word
  goes in Source Descriptors on all of the organization's rows.
  `76-AV_Ghiradelli_Deal.csv` row 5 (1823 Partners): blank -> "asset
  management firm, company", matching SZ's copy.
  `113-California_Homes_Uninsured.csv` row 95 (Farmers): blank -> "home
  insurer" ("Farmers, the second-largest home insurer in California").
  `116-Trump_Christian_Independance_Day.csv` row 7 (Event Strategies):
  blank -> "company" ("Event Strategies, a company that organized the
  rally...").
- **Source Descriptors must be the reporter's own word, 2026-11-05.** A
  Source Descriptor has to be a word the article actually uses for that
  source. `61-AV_Atlantic_Current.csv` row 9 (Science Advances): "study"
  -> "research", matching SZ's copy. `113-California_Homes_Uninsured.csv`
  rows 80-83 (the emails): "internal email" -> blank (the article only
  says people "emailed" or "wrote"); rows 17, 93, 94 (Richard Salazar;
  Linda and Richard Salazar): "Grizzly Flats homeowner" -> blank (the
  article says "one of the few replaced homes belongs to Linda and Richard
  Salazar", never homeowner). `55-Pentagon_Lab.csv` rows 2, 11, 22 (Jerry
  Whirley): "South Shore homeowner" -> blank, matching SZ's copy. *(Still
  open: the same rows' Source Justification is "South Shore Home Owner",
  which isn't the article's wording either.)*
- **Organization Source Descriptors trimmed to the category word, 2026-11-05.**
  `108-Youth_POC_Climate.csv` rows 2, 5, 9, 12 (Youth vs. Apocalypse):
  "climate justice collective" -> "collective", matching SZ's copy (the
  article says "a youth-led, Bay Area-based collective of young climate
  justice activists", which is already the Source Justification).
- **Source Descriptors: kind-words kept, describing words left out, 2026-11-05.**
  A Source Descriptor keeps the words directly in front of it that say
  what kind of person or group it is (a topic word like "affordable
  housing," a word like "presidential" or "full-time," an identity or
  political word like "unhoused" or "far-right," and for people, a place
  like "Florida"). It leaves out words that rate or describe this one
  source ("longtime," "top," "disgruntled") and anything that comes after
  it ("of four," "of the settlements"); those stay in Source Justification.
  `173-Mountain_View_RV_Dwellers.csv` rows 21, 22 (Lenny Siegel): "advocate"
  -> "affordable housing advocate". `113-California_Homes_Uninsured.csv`
  row 26 (United Policyholders): "consumer advocacy group" -> "insurance
  consumer advocacy group". `174-PPP_Loans_Low_Income.csv` rows 13 (Bank
  of America) and 24 (Wells Fargo): "lender" -> "PPP lender".
  `54-Coachella_Influencer.csv` rows 1-4, 14 (Cedoni Francis): "content
  creator, fan, graduate student" -> "full-time content creator, fan,
  graduate student". `185-The_Uncounted.csv` rows 45-49, 51, 52 (Ana):
  "unhoused, mother" -> "unhoused mother". `187-Trans_Youth_Dont_Say_Gay.csv`
  rows 19-21 (Maryn Russ): "non-binary, student" -> "non-binary sixth
  grade student". `168-Texas_Power_Outages.csv` rows 3, 34 (group):
  "residents" -> "Texas residents". `175-Indigenous_Health_COVID.csv`: row
  60 Mary Owen "member of Alaska's Tlingit band" -> "member", Source
  Justification blank -> "a member of Alaska's Tlingit band"; rows 73-75
  Michael Weahkee "enrolled member of the Zuni Tribe" -> "enrolled member",
  row 73 Source Justification blank -> "an enrolled member of the Zuni
  Tribe" (belonging to the community the story is about is why they're in
  it). `167-Youre_Swamped.csv`: row 2 (group) "social media" -> blank (it
  doesn't describe people; the row itself may not be a sourced statement
  -- flagged for review); row 4 (group) "randoms on the Internet" ->
  "randoms".
- **Source Descriptors lowercased, 2026-11-06.**
  A Source Descriptor is copied in the article's own case. A capital
  letter that is there only because the word starts a sentence ("Critics
  have also warned...") isn't part of it, so the descriptor is lowercase
  unless it's a proper noun.
  All rows below are groups of people. `128-Congress_Pay_Battle.csv` row 8
  "Plaintiffs" -> "plaintiffs"; `166-DC_Essential_Workers.csv` row 13
  "Workers unions" -> "workers unions"; `170-Fire_Prevention_Homeless.csv`
  row 14 "Plaintiffs" -> "plaintiffs", row 21 "Advocates, homeless
  communities" -> "advocates, homeless communities";
  `176-Roadblocks_to_Relief.csv` row 4 "Community members" -> "community
  members"; `186-Homeless_Camp_Sweeps.csv` row 18 "Speakers" -> "speakers";
  `187-Trans_Youth_Dont_Say_Gay.csv` row 3 "Supporters" -> "supporters",
  row 4 "Opponents" -> "opponents"; `188-Trans_Women_Incarcerated.csv` row
  8 "Activists" -> "activists", row 29 "Formerly incarcerated trans women"
  -> "formerly incarcerated trans women", row 31 "Opponents" ->
  "opponents"; `189-Cameroon_Unpaid_Wages.csv` row 33 "Citizens" ->
  "citizens".
- **Unnamed spokesperson credited to the spokesperson the story names, 2026-11-06.**
  When a story names exactly one person as speaking for an organization,
  an earlier "a spokesperson said" for that organization is that person.
  `109-Waymo_West_Valley.csv` row 1 ("A spokesperson said the latest
  expansion covers approximately 60 square miles..."): Named Organization
  / Waymo -> Named Person / Sandy Karp / Title "Spokesperson" (she is named
  in the next paragraph), matching SZ's copy. The row's Source Descriptors
  ("autonomous ride-sharing company") and Source Justification (the
  article's opening line about Waymo) were Waymo's, so both are now blank,
  as in SZ's copy.
- **Source Descriptors lowercased, 2026-11-06 (missed in the first pass).**
  `58-Louisiana_Shooting.csv` rows 11 and 17 (Document): "Records" ->
  "records" -- the capital is there only because "Records show..." starts
  the sentence.
- **Phase 2.1 (groups of people), late-text stories, 2026-11-09.**
  `82-Met_Gala_Bezos.csv`: row 2 Source Descriptors blank -> "social media
  users"; row 11 Title of Source blank -> "lawmakers" ("the gala drew
  rebukes from lawmakers"); rows 22 and 24 Source Descriptors blank ->
  "critics" ("his critics," "Critics of Bezos' involvement"); row 5
  ("Reports of skittish stars and upset fashionistas have peppered
  tabloid pages...") Unnamed Group of People -> Document, Source
  Descriptors "reports" -- no group of people is speaking; the reporter
  cites unnamed tabloid reports. `114-Kansas_Gender_Transitioning_Ban.csv`:
  row 11 Title of Source blank -> "lawyers" ("Lawyers for the teenagers";
  the clients aren't part of the title); row 13 Source Descriptors blank
  -> "Republican supporters" ("Republican supporters of the law").
- **Statements that reached the article through someone else, 2026-11-09.**
  When the reporter got a statement from a person or outlet who heard it
  from the original speaker, the person or outlet the reporter heard it
  from is the source. `72-Quantum_Campus.csv` row 22 ("A legal assistant at
  the company told Holcomb at a coalition meeting: ...") Unnamed Person /
  "legal assistant" -> Named Person / A. Anne Holcomb, with her Title
  (co-chair of ETHOS) and Source Descriptors ("South Shore resident"); row
  34 ("'Jobs come and go,' one resident told ABC7, ...") Unnamed Person /
  "resident" -> Named Organization / ABC7 (same reasoning as 210 row 14,
  "told the Wall Street Journal").
- **Phase 2.4 (documents), late-text stories, 2026-11-09.**
  For a document, Name of Source is only the document's own title, and the
  kind of document ("report," "polls") goes in Source Descriptors. When the
  article names the organization the document comes from, that
  organization is the source (Named Organization).
  `72-Quantum_Campus.csv` row 18: Name "remediation report" -> blank,
  Source Descriptors "remediation report". `73-TSA_ICE.csv`: row 4 Document
  "Defense Deparment bulletin" -> Named Organization / Defense Department
  ("according to a Defense Department bulletin"); row 6 Document "American
  Airlines website" -> Named Organization / American Airlines ("a note on
  American Airlines' website"); row 5 Name "U.S. D.H.S. privacy impact
  statement" -> blank, Source Descriptors "privacy impact statement" (the
  article doesn't name DHS there). `82-Met_Gala_Bezos.csv` row 25: Source
  Descriptors blank -> "polls" ("polls show").
  `114-Kansas_Gender_Transitioning_Ban.csv`: row 14 Document -> Named
  Organization / Trump administration ("a report commissioned by the Trump
  administration"); row 10 Source Descriptors blank -> "lawsuit" ("The
  Kansas lawsuit, filed... by the American Civil Liberties Union of
  Kansas..., claimed" -- the article credits the lawsuit itself, so it
  stays a document).
- **A named person speaking in a deposition is the source, 2026-11-09.**
  `113-California_Homes_Uninsured.csv` row 85 ("...said Ari Berman, the
  plaintiff's insurance agent, in a deposition"): Document / "deposition"
  -> Named Person / Ari Berman, Title of Source "insurance agent"; Source
  Justification ("The documents from the Woolsey Fire survivors' case
  contained other revelations...") blanked -- it described the case
  documents, not him. Matches row 24 (Jeff Dailey, "indicated in a 2018
  deposition," already Named Person).
- **Phase 2.5 (organizations), late-text stories, 2026-11-10.**
  `114-Kansas_Gender_Transitioning_Ban.csv` row 1 ("...a state judge ruled
  on Friday"): Named Organization / "Kansas State Judge" -> Unnamed Person,
  Title of Source "state judge" -- no judge or court is named in that
  sentence. `72-Quantum_Campus.csv` row 37 ("...according to an NBC5 and
  DePaul University investigation published this week"): one row for NBC5
  became two -- NBC5, and a new row 38 for DePaul University with the same
  Sourced Statement, since the sentence credits both. 72 now has 40 rows;
  old rows 38 and 39 are now 39 and 40.
- **Phase 2.5 (organizations' Source Descriptors), late-text stories, 2026-11-10.**
  The article's word for what kind of organization it is goes in Source
  Descriptors on all of that organization's rows.
  `72-Quantum_Campus.csv` row 5 (Related Midwest): blank -> "master
  developer" ("the 440-acre site's master developer").
  `82-Met_Gala_Bezos.csv` rows 3 and 4 (Everyone Hates Elon): blank ->
  "guerrilla activist group, anti-billionaire group" (two separate
  phrases in the article). `114-Kansas_Gender_Transitioning_Ban.csv` row
  15 (Williams Institute): blank -> "demographics research center".
- **Phase 2.6 Pass 0 (named people), late-text stories, 2026-11-11.**
  Titles written the way the article writes them, and the same for every
  row of a person. `82-Met_Gala_Bezos.csv`: old row 29 ("Micheal Gross,"
  the quote "Every eye in the world is going to be on that...") removed --
  it repeated the end of row 28 (Michael Gross), so 82 now has 28 rows;
  row 26 (Cynthia Nixon) Title of Source "null" -> blank; row 1 (Zohran
  Mamdani) Title "New York Mayor" -> "mayor" (the article says "the city's
  newly elected mayor" and "Mayor Mamdani"; "New York" isn't attached to
  his title). `80-Quantum_Medicine.csv` row 10 (Jerry Chow): Title
  "chief technology officer of quantum-centric computing, IBM" -> "IBM's
  chief technology officer of quantum-centric computing," as on his other
  rows. `114-Kansas_Gender_Transitioning_Ban.csv`: row 3 (Carl Folsom III)
  Title -> "judge, State District Court in Douglas County," as on row 6;
  row 2 (Harper Seldin) "American Civil Liberties Union Lawyer" -> "lawyer
  for the American Civil Liberties Union"; rows 4, 5, 12 (Kris W. Kobach)
  "Attorney General, Kansas" -> "Attorney General" and row 7 (Laura Kelly)
  "Governor, Kansas" -> "Gov." -- the article doesn't attach "Kansas" to
  either title.
- **Phase 2.6 Pass 1 (titles of named people), late-text stories, 2026-11-11.**
  A Title of Source is a position of authority or expertise; other words
  for who a person is go in Source Descriptors. An elected official's Title
  includes the party the article gives with it.
  `72-Quantum_Campus.csv` rows 6, 8, 9, 11, 25, 30 (Sanya Bhartiya): Title
  "environmental justice organizer with Southside Together" -> blank,
  Source Descriptors "environmental justice organizer" ("organizer" isn't
  a position title; the group she works with is left out).
  `73-TSA_ICE.csv` row 9 (John Garamendi): Title "California
  Representative" -> "Rep. D-Calif." ("Rep. John Garamendi, D-Calif.").
  `82-Met_Gala_Bezos.csv` row 13 (Jon Reinish): Title "veteran Democratic
  political strategist" -> "political strategist," Source Descriptors
  "Democratic" (he doesn't hold office; "veteran" describes him and is left
  out). `114-Kansas_Gender_Transitioning_Ban.csv`: rows 4, 5, 12 (Kris W.
  Kobach) Title "Attorney General" -> "Attorney General, a Republican"; row
  7 (Laura Kelly) "Gov." -> "Gov., a Democrat" -- the party the article
  gives for each official.
- **Phase 2.6 Pass 3 (named people's Source Descriptors), late-text stories, 2026-11-12.**
  The article's own label for a person goes in Source Descriptors on all of
  that person's rows. `72-Quantum_Campus.csv` rows 17, 20, 21, 29, 40 (A.
  Anne Holcomb): blank -> "South Shore resident," as on rows 16 and 22 ("a
  15-year South Shore resident"). `82-Met_Gala_Bezos.csv`: row 1 (Zohran
  Mamdani) -> "democratic socialist"; row 6 (Mayan Rajendran) -> "passer-by"
  ("one passer-by"); row 26 (Cynthia Nixon) -> "actress, progressive
  activist" ("the 'Gilded Age' actress and progressive activist"); row 27
  (Bill de Blasio) -> "Democrat" ("Former Mayor Bill de Blasio, a two-term
  Democrat"; he no longer holds office, so his party isn't part of his
  title).
- **Phase 3 checks, late-text stories, 2026-11-12.**
  Source Justification should say why the source is in the story. A word
  "null," the attribution alone ("polls show"), or a repeat of the Title
  isn't that, so those were cleared.
  `82-Met_Gala_Bezos.csv`: row 18 (Fern Mallis) "null" -> blank; row 25
  (polls) "polls show" -> blank.
- **Phase 4a (missed sourced statements), `72-Quantum_Campus.csv`, 2026-11-12.**
  All of a source's statements in one paragraph go in one row.
  Row 13 (Moises Moreno): added the rest of his paragraph, "The governor
  came in. They didn't invite community. They just came in and made the big
  announcement," he said. Row 19 (ETHOS): added "According to their
  findings, every sample came back hazardous to human health. Some were
  classified extremely hazardous." ahead of "ETHOS believes there is a
  sixth Hot Zone not in the report" (same paragraph; the report sentences
  between them belong to row 18). Row 23 (Isabela Jaimes): old rows 23 and
  24 were one paragraph, now one row, with the paragraph's last sentence
  added ("And that's usually because they enter into non-disclosure
  agreements..."). New rows: 24, Document "federal websites" ("According to
  federal websites, its portfolio includes stealth weapons..."); 31, Named
  Organization GreenLatinos ("GreenLatinos also raised what the statement
  left unanswered..."); 33, Named Person Sharon Waller, Title "Dr." ("That
  question connects to what Dr. Sharon Waller documented in last week's
  report..."). 72 now has 42 rows; old rows 31-40 are now 32 and 34-42.
  Row 18 (remediation report): added "Three are in the PsiQuantum parcel.
  Hot Zone One is kitty-corner to New Sullivan Elementary School." -- both
  give details of the five Hot Zones the report identified, so they come
  from the same report.
- **Phase 4b, `72-Quantum_Campus.csv`, 2026-11-12.**
  One row per source per paragraph; the reporter's own sentences are left
  out of a Sourced Statement. New row 1 (Unnamed Group of People,
  Source Descriptors "residents"): "On the evening of March 18, 2025,
  several dozen residents packed the auditorium at Bowen High School...,
  demanding answers about the $9 billion Quantum Campus..." -- "demanding
  answers" is the residents' own expression, so it's a sourced statement.
  Row 2 (the woman in the audience, old row 1): Sourced Statement trimmed to
  "'How are you going to deal with the contaminated land?' she asked." (the
  first paragraph is now the residents' row, and "A woman in the audience
  cut to the heart of it." is the reporter's lead-in); Source Justification
  "A woman in the audience" -> "A woman in the audience cut to the heart of
  it." (her presence at the meeting and why her question matters). Sanya
  Bhartiya's and Moises Moreno's rows that ran across two paragraphs were
  each split in two (old rows 6 and 14; new rows 8 and 17 hold the second
  paragraphs). 72 now has 45 rows; every row from old row 1 on moved down
  by one, from old row 7 by two, and from old row 15 by three.
- **Phase 4 (rows checked against the article), `73-TSA_ICE.csv`, 2026-11-13.**
  No missed statements. Row 2 (Reuters): the reporter's background sentence
  ("The traveler records were gathered by TSA's Secure Flight Program,
  created in 2007...") left out; the row is now "Tips from the TSA led to
  more than 800 arrests by ICE, Reuters found." When a statement reached
  this Forbes story through another outlet, that outlet is the source: row
  8 ("...three immigration attorneys told Reuters") Unnamed Group of
  People / "immigration attorneys" -> Named Organization / Reuters; row 9
  ("...Rep. John Garamendi, D-Calif., told the New York Times at the
  time") Named Person / John Garamendi -> Named Organization / New York
  Times.
- **Phase 4 (rows checked against the article), `80-Quantum_Medicine.csv`, 2026-11-13.**
  Old row 3 (group "investors": "A lot of quantum research has centered on
  this type of work... However, investors have been eager for results...")
  removed -- the reporter describes investors' mood; nothing is credited to
  them. Old row 7 (Kenneth Merz) repeated row 6's sentence ("The latest
  experiment shows how existing hardware can be paired with quantum
  machines..., he said") and ran into the next paragraph; it is now only
  his next-paragraph quote ("You take a very big problem and break it into
  small pieces..."). Old row 8 (Jerry Chow): added the rest of his
  paragraph, "The scientific community isn't there yet, and to date, there
  is no consensus whether or not true advantage has been achieved." New
  row (IBM): "IBM last month announced it would create 750 jobs in Chicago
  over the next five years to staff a new 'innovation and delivery hub.'"
  80 now has 10 rows: old rows 4-10 are now 3-9, and the IBM row is 10.
- **Phase 4 (rows checked against the article), `82-Met_Gala_Bezos.csv`, 2026-11-13.**
  No missed statements. One row per source per paragraph: Micah Uetricht's,
  Bill de Blasio's and Michael Gross's rows each ran into the next
  paragraph and were split in two; Amy Odell's two rows from one paragraph
  were merged. Old row 7 (Blakely Thornton, "He wasn't the only one asking
  that question. Other critics have included past boosters like Blakely
  Thornton...") removed -- it introduces him but credits him with nothing;
  his Source Justification moved to his remaining row ("an influencer who
  interviewed celebrities on the red carpet last year"). Old row 21 ("The
  split screen of the gala's opulence juxtaposed with the economic
  struggle..., where an estimated 1 in 4 residents live in poverty, has
  long been evident") removed -- the reporter's own framing, no source
  named. The polls row: the reporter's following sentence ("The mayor's
  early and successful embrace of affordability...") left out. The polls
  and critics rows put back in article order. 82 still has 28 rows; old ->
  new: 8-12 -> 7-11, new 12 (Uetricht's second paragraph), 13-14 same,
  16-20 -> 15-19, 22-23 -> 20-21, 25 -> 22, 24 -> 23, 26-27 -> 24-25, new
  26 (de Blasio's second paragraph), 28 -> 27, new 28 (Gross's quote).
- **Phase 4a (missed sourced statements), `114-Kansas_Gender_Transitioning_Ban.csv`, 2026-11-13.**
  New row 10 (Supreme Court): "Last year, the Supreme Court upheld a
  Tennessee ban on transition treatments for minors in the case United
  States v. Skrmetti. That ruling, which found that states had the ability
  under federal law to restrict those procedures, was a crushing blow to
  the transgender rights movement." New row 16 (unnamed organizations,
  Source Descriptors "medical societies"): the same sentence as row 15 --
  "The major American medical societies endorse them, but a report
  commissioned by the Trump administration concluded..." -- credits both
  the medical societies and the administration's report. Row 17 (Williams
  Institute): added "That amounts to about 1 percent of the U.S. population
  in that age group." 114 now has 17 rows; old rows 10-14 are now 11-15,
  old row 15 is 17.
- **Phase 4b, `114-Kansas_Gender_Transitioning_Ban.csv`, 2026-11-13.**
  Two Source Justifications described a different source and were cleared:
  row 14 (Republican supporters of the law) "a report commissioned by the
  Trump administration" -> blank; row 15 (Trump administration) "a
  demographics research center at the University of California, Los
  Angeles" (that describes the Williams Institute) -> blank. That
  description moved to row 17 (Williams Institute), whose SJ was blank.

- **Anonymity disclosure in Source Justification, 2026-11-13.** The
  reporter's disclosure that a source asked for or was granted anonymity
  is Source Justification on every row for that source (now v61 Note 22a);
  existing valid SJ text kept, with the disclosure after "; ".
  `77-SF_Drug_Detention.csv` rows 11, 12: + "who asked not to be named".
  `176-Roadblocks_to_Relief.csv` rows 1, 2, 9: + "El Tímpano is only using
  the first names of interviewees to protect their privacy." (the story's
  footnote). `189-Cameroon_Unpaid_Wages.csv` rows 2, 3, 4: + "indicates
  changed names for protection" (the footnote for the starred name).
  `98-AV_Trump_China.csv` row 1: blank -> "who was not authorized to
  comment publicly and spoke on the condition of anonymity" (as in
  `98-Trump_China.csv`).

- **Solidarity reconciliation with AV's revised files, 2026-11-16.** AV's
  revisions brought into the migrated CSVs (each difference checked against
  the Sep17 original to tell her edits from migration edits; migration
  edits kept). `167-Youre_Swamped.csv`: both rows for "Every hurricane
  season... people—both randoms on the Internet and whole ass
  journalists—ask: "Why don't people on the Gulf Coast just move?""
  removed (the "randoms" and "journalists" rows; AV removed one of them;
  a question, not a claim, so neither stays); now 4 rows.
  `171-Bracing_Next_George_Floyd.csv`: rows 7 (Artika Tyner, "remembers
  hearing protests outside of her window...") and 14 (Jerome Treadwell,
  "Treadwell and other organizers have been calling out racist
  incidents...") removed and their sentences made the Source
  Justification of Tyner's row 8 and Treadwell's rows 15 and 16 (AV's
  revision); row 2 removed ("Corporations and government officials
  promised diversity and equity." -- AV's removal accepted); row 1
  (protesters "demanding change") kept despite AV's removal (a group's
  demand is a Sourced Statement); now 13 rows.
  `190-West_Texas_Drag_Queens.csv` rows 9, 10 (Miss Calvina): Source
  Justification extended to the full sentence, "...she saw this event as an
  opportunity to teach literacy in a different way." (AV's revision).
  `189-Cameroon_Unpaid_Wages.csv`: AV's file has no changes of her own.

- **Phase 4b (deferred solidarity 4b), `190-West_Texas_Drag_Queens.csv`,
  2026-11-16.** Rows narrating what a source did or felt, with nothing
  crediting the source, removed: Destiny Adams rows 1 (arriving at the
  coffeehouse; hosting the event), 2 (her doubts), 3 (her surprise), 11
  (opening the shop), 12 (her vision for it), 21 (hiring security), 37
  (exhausted and grateful), 39 (fielding inquiries); row 14 (Victoria
  Hughes's background -- already her Source Justification); rows 31 and 32
  (Donabela Kanela and Vanessa P. Nevaeh "saw their presence... as a way to
  fight back against misconceptions about their community" -- moved to
  their rows 33 and 34 as Source Justification). Destiny Adams's remaining
  rows: Source Justification blank -> "In a few hours, Adams would be
  hosting a Pride event featuring the first-ever drag queen story time in
  her small, conservative town of Wolfforth, a suburb of Lubbock with a
  population under 6,000." Trailing or lead-in reporter text taken out of
  Sourced Statements: row 9 (Miss Calvina, "Having worked seven years..."
  -- already her Source Justification), row 17 (Steve Miller, "Miller had
  been a vocal critic..." -- already his Source Justification), row 35
  (Grace Rogers, "She attended the drag queen story hour..."), row 25
  (Bryan Slaton, the lead-in "Since the next legislative session doesn't
  convene until Jan. 10, 2023, Slaton has yet to submit any
  legislation."), row 26 (Slaton, "A similar law already exists for
  employees of sexually oriented businesses."). Now 29 rows.

- **Phase 4b (deferred solidarity 4b), `189-Cameroon_Unpaid_Wages.csv`,
  2026-11-16.** Row 42 removed (Wawa Jackson Nfor: "Before going to jail,
  Jackson worked for a media house that used to pay him less than FCFA
  25,000... He had no contract with his employer." -- narration, nothing
  credits him); row 41 (Jackson): the lead-in "Another journalist, Wawa
  Jackson Nfor, was in prison in Nkambe for 33 months without trial." taken
  out of the Sourced Statement, and Source Justification blank -> that
  sentence plus the removed row-42 text (his own experience). Rows 25, 26,
  27 (Nkafu Institute, Denis and Lenora Foretia Foundation, Atlas
  Network): the lead-in "Cameroon’s public debt has risen very quickly
  within the last five years." taken out. Row 46 (Mbomda Djomeni Casimir):
  the trailing "This provides hope for journalists like Mohamed Auwal*, who
  fears retribution for coming forward." taken out. Now 45 rows.

- **Solidarity reconciliation + deferred 4b, `176-Roadblocks_to_Relief.csv`
  and `166-DC_Essential_Workers.csv`, 2026-11-17.** Roberto (176), Yolanda
  (176) and Sally (166) stay Anonymous Source (first names only, with the
  story's privacy disclosure) -- AV's change to Named Person not taken.
  176: row 1 (Roberto narration, "For Roberto, a 55-year-old father...
  he never heard back.") removed and made the Source Justification of his
  quote row, followed by "; El Tímpano is only using the first names of
  interviewees to protect their privacy." (AV's revision); the "One
  interviewee learned that she missed a deadline..." row: those first two
  sentences moved from Sourced Statement to Source Justification, the row
  now starting "She tried going in person..." (AV's revision); "several
  shared that once their children turned 18..." -> the full sentence ("For
  instance, while undocumented immigrants can receive CalFresh...");
  new rows, Unnamed Group of People: "Despite the urgency of their need for
  assistance, many have expressed a wariness..." and "That said, the
  overwhelming sentiment community members have expressed throughout the
  pandemic is that of frustration." (Source Descriptors "community
  members"); the cosigner quote row's quotation marks made curly, as in the
  text. AV's Source Descriptors "interviewee" not taken (a word for the
  source's role in the reporting). Now 16 rows. 166: row 1 (Tamara
  Keitt-Barnes) -> the full sentence, "A nanny for Capitol Hill families,
  Prince George’s County resident Keitt-Barnes said..." (AV); row 7 (Sally
  driving home, her dog) removed (AV); the "For Keitt-Barnes, the nanny from
  Prince George’s County, it was a long day..." row removed and made the
  Source Justification of her last row (AV); row 12 (Washington Post
  reporter's firsthand account): Named Organization / Washington Post ->
  Unnamed Person, Title "Washington Post reporter" (AV; the reporter speaks
  for themself, not the Post); new row, Unnamed Group of People: "After the
  chaos settled, a video of masked Capitol custodial staff... prompting
  outrage as people attempted to coordinate donation efforts for the
  employees, and called for hazard pay."; "Workers unions called for
  downtown hospitality workers to be evacuated..." kept (AV had removed
  it); Jaime Contreras's Source Justification kept (AV's version restated
  his statement). Now 15 rows.

- **Solidarity reconciliation + deferred 4b, `173-Mountain_View_RV_Dwellers.csv`
  and `174-PPP_Loans_Low_Income.csv`, 2026-11-17.** 173: Misty Masvalo's
  narration rows removed -- 1 (living in a retrofitted van), 4 (her
  wages), 6 (buying the van), 7 (watching the ballot measure) (AV's
  revision), and 3 ("Masvalo wants to find permanent housing...") and 23
  ("Masvalo’s van is eight feet tall... She feels her job is
  important...") (our 4b); her remaining rows' Source Justification ->
  the row-1 paragraph + "; Masvalo’s van is eight feet tall, which means
  she’d be affected by Measure C if it passes."; row 18 removed
  (supporters "hold a widely held belief..." -- a belief, not a
  statement; AV); homeowners "who complain that oversized vehicles block
  bike lanes..." kept (AV had removed it); AV's Source Justifications for
  Prody Hazarika and Margaret Abe-Koga not taken (they restated their own
  statements). Now 17 rows. 174: row 2 (Congress: "Congress wanted the
  U.S. Small Business Administration to ensure that PPP loans prioritized
  small businesses in "underserved" markets...") removed and made the
  Source Justification of the SBA row (AV); rows 30, 31 (the Browns'
  narration, "That’s exactly what it was for KB and Katie Brown...")
  removed (AV); Katie Brown's rows: Source Justification blank -> "As the
  pandemic hit, orders cratered at KB and Katie Brown’s print shop in
  Minneapolis. They knew they would need help — exactly what the Paycheck
  Protection Program was meant to provide." (AV), and KB Brown's row 1 adds
  the same after "; " to his existing one; the Rosa Daniels row: the
  lead-in "That’s what happened to Rosa Daniels and her daughter, Dana
  Myers." taken out (the row stays -- "Daniels said...") and AV's
  Source Justifications that restated it not taken; the "Bank of America
  issued more than 340,000, according to the SBA’s records" row kept (AV
  had removed it); new rows for KB Brown and Katie Brown: "From the
  Browns’ perspective, the PPP was opaque and confounding." (from AV's
  revision, first sentence only); AV's other new row ("The Browns, unsure
  what was happening, thought the funding would eventually arrive...") not
  taken (narration). Now 48 rows.

- **Solidarity reconciliation + deferred 4b, `168-Texas_Power_Outages.csv`,
  2026-11-17.** Removed: rows 1 and 2 (from the series blurb at the top,
  "top state lawmakers called for investigations..." and "Texans demanded
  accountability..." -- not body text); row 6 (Texas officials "have
  signed an order..." -- an action); Lisa Blevins rows 16 (the household
  in the car -- now the Source Justification of her next row) and 19 (her
  son's family); Ashlynn Hoffner rows 20-23 and KD Hoffner row 25 (their
  night, narrated). Narration moved from Sourced Statement to Source
  Justification: Brianna Blake row 12 (the row now "There was “no
  alternative” to keep the heat going, she said."), Lisa Blevins row 15
  (now "Blevins, 58, said she couldn’t take her daughter anywhere..."),
  Sarra Rushing row 36 (now "The local government could have done a
  better job preparing residents for the storm, she said."). Hoffner
  Source Justifications: KD -> "Ashlynn Hoffner, in Richardson, called
  emergency responders around 4:30 a.m. as her partner’s oxygen machine
  ran low on battery.; The couple will drive to Plano tomorrow so KD
  Hoffner can get dialysis needed for chronic kidney and heart issues.";
  Ashlynn -> the first of those sentences. Row 5: Title "Officials" ->
  "officials in Austin." National Weather Service row: the outlet's own
  guidance ("Some cities are providing warming centers... find nonprofits
  who are helping people here.") taken out. Kept despite AV's revision:
  state and city officials "urging patience" (a group's statement), "Many
  faulted the government..." (a group's statement), the ERCOT official as
  Named Organization / Electric Reliability Council of Texas, Sergio
  Tristan's paragraph as one row, Brianna Blake's Source Justification on
  all her rows. Most of AV's changes taken (her narration-to-SJ moves).
  Now 31 rows.

- **Solidarity reconciliation + deferred 4b, `170-Encampment_Fires.csv`,
  2026-11-17.** Removed (narration of actions/events; AV's revision): Cheryl
  Davila row 3 ("pushed to create an official program..." -- now the
  Source Justification of her other two rows); Caltrans row 7 ("the
  Seabreeze Camp was scheduled to be dismantled by Caltrans..."); Mama
  West rows 16-18 (tent fires, the puppy, Animal Control); Derrick Soo
  rows 23, 26, 29, 31, 33 (fire safety system, solar grid, air fryer,
  fire retardant paint, Conestoga huts); Kyle Mitchell rows 24, 25 (tools
  purchased, extinguisher refills). Also removed (ours): Soo row 34 ("Since
  2016, Soo has been advocating for the sanctioning of his camp..." --
  narration; AV had kept it trimmed). Soo's remaining rows: Source
  Justification -> "In deep East Oakland, leader Derick Soo set up a fire
  safety system for the 77th Avenue Rangers and advocated for temporary
  access to a nearby fire hydrant.; Since 2016, Soo has been advocating
  for the sanctioning of his camp, and he also helps camp residents get
  connected to housing services." Mama West's row: second sentence ("She
  threw it in the sand...") taken out; Source Justification adds "; In
  2021, Mama West’s tent has caught fire three times as winds spread the
  flames from fires set by neighbors for cooking and warmth." Mitchell's
  row: trimmed to "Other options when propane is not available and money
  is tight are bottles filled with hot water or heated stones, according
  to Mitchell."; Source Justification blank -> "Mitchell has purchased fire
  suppression tools for the Rangers like extinguishers, alarms, Thermoses
  and power banks." Berkeleyside row: trimmed to its last sentence ("Two
  people accepted a four-person tent... according to a Berkeleyside
  report..."; AV), type kept Named Organization. Oakland Fire Department
  fire-figures row: kept (AV had removed it), lead sentence "These tools
  can make a difference, especially when it’s cold." taken out. Row 8
  Title "city staff" -> "Berkeley city staff". Not taken from AV: Paul-
  Kealoha Blake's Title moved to Source Justification; Source
  Justifications that weren't in the text or restated the row. Now 22
  rows.

- **Solidarity reconciliation + deferred 4b, `172-Delhi_Flyover_Homeless.csv`,
  2026-11-17.** Removed (narration; AV's revision): Shahida rows 7 ("When
  her partner left shortly after Jannat’s birth...") and 23 ("In such
  situations, the family visits a local doctor..."). Shahida's rows: the
  reporter's scene narration taken off the front of rows 5 ("Shahida has
  lived through these realities..."), 6 ("But this year Shahida is
  enduring..."), 8 (the shanty history), 9 (the flyover), 10 (the honk),
  12 (the water tap), 13 (asleep again, the rashes), 14 ("After trying to
  nap again, Shahida gives up."), 15 ("Nights on the street are spent in
  fear of harassment."), 20 (after cooking), 28 ("A shower at the public
  toilet costs 10 rupees.") -- rows 5 and 9 from AV's revision, the rest
  ours; Abida row 21: "The afternoon passes waiting for the sun to
  weaken." taken out; Centre for Holistic Development row: "The
  consequences are deadly." taken out. Shahida's Source Justification, on
  all her rows -> her existing one + "; But this year Shahida is enduring
  Delhi’s extreme heat with a baby, her nine-month-old daughter, Jannat.;
  When her partner left shortly after Jannat’s birth, Shahida moved back
  with her family under the flyover. Today, the family of 10 – including
  her mother, father, two sisters, a brother-in-law and four children –
  all live on the pavement." (AV's revision had a Source Justification
  that grew row by row -- not taken). Kept despite AV's revision: the two
  India Meteorological Department temperature rows (credited by the
  story's end note), "Data suggests that nearly 99%..." (Document). Now
  32 rows.

- **Solidarity reconciliation + deferred 4b, `185-The_Uncounted.csv`,
  2026-11-17.** Removed (narration; AV's revision): Jeremiah rows 1, 3, 4,
  5 (the BART ride and the overdose reversal); David Modersbach row 13
  ("Modersbach works to understand this picture..."); Thad rows 21 ("In
  the first story in this series, we spoke with Thad...") and 40 ("Thad,
  like many unhoused people who use drugs, obtains Narcan..."); Ana rows
  46, 47, 51 (Biddulph's Facebook photo, going to work, Berkeley NEED).
  Also removed (ours): Ana row 45 ("The morning of the overdose, Ana spent
  time with Eric Cumby...") and Modersbach row 15 ("...Modersbach hopes to
  assemble a community task force..." -- a hope). Source Justifications:
  Thad's rows (blank) -> "In the first story in this series, we spoke
  with Thad, a 36-year-old resident of North Oakland. Thad has survived
  three overdoses since the pandemic began, all after smoking fentanyl."
  (AV); Jeremiah's (blank) -> "Wherever he goes, Jeremiah carries
  Narcan—that’s a brand name for the opioid overdose revival drug
  naloxone."; Modersbach's adds "; Modersbach works to understand this
  picture by cobbling together whatever mortality information he can
  find."; Ana's adds "; The morning of the overdose, Ana spent time with
  Eric Cumby." Trimmed of the reporter's explanation or scene-setting: the
  BART "ridership plummeted" row, Jeremiah's "This wasn’t the first time..."
  -- since restored (see below), the Alameda County Public Health Department row's last
  sentence, Thad's "Thad knows fentanyl is potent... He carries Narcan..."
  and "EMS workers have a name..." sentences, the EMS row's "Street Spirit
  reached out..." lead-in, Ale Del Pinal's "Punks With Lunch is a spunky
  volunteer organization..." lead-in, Ana's "Toxicology reports take
  months..." sentence, the San Francisco Department of Public Health
  row's "No such government-led efforts..." sentence, Katie O’Bryant's
  "Standing in the Punks With Lunch office, " lead-in -- since restored (see below). Alameda County row
  ("Alameda County submits data...") trimmed to "Multiple coroner’s bureau
  staff confirmed to Street Spirit that their records do not consistently
  note whether someone was unhoused when they died." and retyped Named
  Organization / Alameda County -> Unnamed Group of People, Title
  "coroner’s bureau staff". Then (user rule: one sentence is never cut to
  keep only its attributed part): "many people who use drugs say police
  represent the opposite." -> "While harm reduction advocates employ
  empathy and acceptance in their work, many people who use drugs say
  police represent the opposite."; the Jeremiah and Katie O’Bryant
  lead-ins above put back, since each was part of the same sentence. Now
  43 rows.

- **Solidarity reconciliation + deferred 4b, `186-Homeless_Camp_Sweeps.csv`,
  2026-11-17.** Puma's rows 6 and 7 trimmed to start at the quote ("“You
  see the trash?” he asked..." and "“I got stability here...”") -- the
  scene narration before them taken out (AV's revision). Removed: Alfredo
  Reyes row 67 ("Reyes revisited the St. Elmo camp on the morning of
  Monday, June 8...") (narration; AV); row 3 (Homeless Strategies and
  Operations Department: "Under the program, sweeps are happening more
  frequently..." -- no attribution); row 46 ("As we did, a car drove up
  and two HSO employees emerged. We asked to speak with them. “No
  thanks,” one replied." -- a refusal to comment). Trimmed at sentence
  boundaries: Austin Homeless Strategies and Operations Department row
  ("The public got its first briefing..." taken out), David Gray ("He
  also presented a map..." taken out), Reyes ("He is HIV positive..."
  taken out -- it is his Source Justification; and "Reyes spoke low and
  fast." taken out), Angel ("Angel’s voice was also quiet." taken out).
  Paul Tardie Title "a staffer at the Texas Harm Reduction Alliance" ->
  "staffer at the Texas Harm Reduction Alliance". Snehal Patel Title "an
  internal medicine physician who has treated homeless patients" ->
  "internal medicine physician"; Source Justification blank -> "who has
  treated homeless patients". Added: Reyes, "Reyes told them the camp was
  about to be swept, that the city had arrived."; Unnamed Group of People
  (Source Descriptors "members"), "Members of VOCAL-TX, Austin Mutual Aid,
  the Texas Harm Reduction Alliance, and other grassroots organizations
  denounced the sweeps to members of the media." Kept despite AV's
  revision: Betty and August rows ("...which they said happened in
  March."); Reyes's Source Justification. Now 67 rows.

- **Solidarity reconciliation, `175-Indigenous_Health_COVID.csv`, batch 1
  (AV's revision), 2026-11-17.** Removed (narration; AV): Melvina Musket
  rows 1, 3-9, 11, 116, 118, 119; Jamie Barboan rows 23-27, 29, 31; Kris
  Pikaart rows 42, 81, 82, 84, 85; Gabriella Lee rows 46-49; Rajiv Patel
  rows 91, 93, 94; Michelle Lujan Grisham row 35 ("...issued an order for
  bar closures" -- an action); Kevin Foley row 38 ("Some staffers got sick
  or quit, forcing the detox center to shut down..." -- unattributed);
  Sanjay Choudhrie row 66 ("The gym never housed more than 10 patients..."
  -- unattributed). Source Justifications (blank before): Musket ->
  "Melvina Musket stared at her dying father through the cellphone
  screen.; On June 11, their mother died, five days after her husband.";
  Barboan -> "Her brother, Andreas Tolth, was 39.; After a monthlong
  battle, he died from the coronavirus in July."; Pikaart -> "As the
  crisis grew, Pikaart set up final phone and video calls with families,
  becoming a conduit in the human tragedy.; For six months, Pikaart has
  been surrounded by critical illness and death."; Lee -> "Lee’s uncle got
  sicker and sicker until he could barely breathe."; Phillip Smith -> "Dr.
  Phillip Smith, 70, spent most of his career as a national Indian Health
  Service administrator before retiring to head Utah’s Monument Valley
  Health Clinic, which also got hit hard by COVID-19." (taken off the
  front of his row 78, which now starts "He said Gallup Indian Medical
  Center has long lacked..."); Gary Morsch -> his row 112's first two
  sentences ("Gary Morsch, an emergency doctor from Olathe, Kansas,
  recognized the pandemic threat early..."), the row now starting "Morsch
  said his teams filled in...". Foley row 37: "They were given
  metal-framed beds in large common rooms. Although screening for fevers
  had begun already, sickness crept inside." taken out. Kept despite AV's
  revision: the five "warned for decades" rows; Patel's "As she helped
  patients and families, others on the Rehoboth McKinley hospital staff
  were overwhelmed by the patient surge, according to Dr. Rajiv Patel..."
  and "Patel, whose wife is Diné, resigned citing safety concerns."; "But
  she had heard others warn, “Once they’re in the hospital, you won’t see
  them (again).”" kept as Named Person / Jamie Barboan (AV had retyped it
  Unnamed Group of People "others"; the others' warning is nested in what
  Barboan told the reporter) and given her Source Justification. Now 85
  rows.

- **`175-Indigenous_Health_COVID.csv`, batch 2 (deferred 4b), 2026-11-17.**
  Title of Source: David Conejo "the medical center’s former chief
  executive" -> "medical center’s former chief executive"; Ina Burmeister
  "a Rehoboth McKinley spokesperson" -> "Rehoboth McKinley spokesperson";
  Jonathan Iralu: leading "the" dropped; Gary Morsch "an emergency doctor
  from Olathe, Kansas; founder of COVID Care Force; former Army
  physician" -> "emergency doctor; founder of COVID Care Force; former Army
  physician"; Kevin Foley "heads the town’s detox center" (one row) ->
  "executive director at Na’Nizhoozhi Center" (as on his other row); Kris
  Pikaart "Hospital chaplain" -> "chaplain at Rehoboth McKinley Christian
  Hospital"; Tom Udall "U.S. Sen., vice chair of the Committee on Indian
  Affairs" -> "U.S. Sen., D-N.M.; vice chair of the Committee on Indian
  Affairs"; Rajiv Patel "supervised the hospital’s eight-bed intensive
  care unit" -> "Dr.", Source Justification blank -> "who supervised the
  hospital’s eight-bed intensive care unit"; Randy Chatto blank ->
  "community leader". Sourced Statements: medical experts row -- "The
  federal government, which oversees health care for Native Americans
  under treaty obligations, used a modified influenza plan to address the
  pandemic." taken out; Arizona Republic row trimmed to "In the first two
  weeks of May, Navajo police issued nearly 500 citations to curfew
  violators, according to The Arizona Republic."; Pikaart's first row
  trimmed to "Hospital chaplain Kris Pikaart said she saw 50 patients
  die..." (the drip-line sentences before and after taken out); activists
  row "and even today, activists on the Navajo Nation stress..." (started
  mid-sentence) -> the full sentence "Tribes have until Dec. 30 to spend it
  – and even today, activists on the Navajo Nation stress that the money
  has still not trickled down to the communities that need it most." 85
  rows.

- **Deferred 4b, `169-Black_Mothers_Gun_Violence.csv`, 2026-11-18.** Shea
  Kuykendoll: Title blank -> "student advocate, University of Memphis";
  Source Descriptors "mother, student advocate" -> "mother". Sharmaine
  Brown's Source Justification: the repeated last sentence ("Her son died
  of a gunshot wound in 2015." -- the text says it twice) dropped. 53 rows.

- **Deferred 4b, `187-Trans_Youth_Dont_Say_Gay.csv`, 2026-11-18.** Rows 1-2
  (critics / Ron DeSantis) and 7-8 (DeSantis / critics) each split one
  sentence between two sources -> both rows of each pair now the full
  sentence ("What critics call the "Don't Say Gay" bill, Gov. Ron DeSantis
  called in an email to supporters “a Florida bill that sensibly
  prohibits...”" and "The governor has repeatedly said the bill does not
  specify gay — or any other queer identity — but critics say that's not
  true."). Removed: Rain Weinstein row 6 ("He's one of several youth
  documenting his experiences... TikTok account..." -- now his Source
  Justification, blank before); Salem Smith row 16 ("...is worried that
  kids who aren’t exposed... will suffer...") and Maryn Russ row 19 ("...is
  worried about the provision...") -- reported concerns. Weinstein's row
  26: "This includes Texas... investigations have already started." taken
  out. Weinstein Title "senior at Palm Harbor University High School" ->
  blank (Source Descriptors already "transgender, senior"; matches Sage
  Ackerman and the rest of the GT). Russ Source Descriptors "non-binary
  sixth grade student" -> "non-binary, sixth grade student". Now 25 rows.

- **Deferred 4b, `188-Trans_Women_Incarcerated.csv`, 2026-11-18.** Title of
  Source, leading article dropped: Rodrigo Heng-Lehtinen ("the incoming
  executive director..."), Richard Saenz ("a senior attorney with Lambda
  Legal"), Jennifer Orthwein ("a public interest attorney in
  California"). Jasmine Rose Jones row 1: "Jasmine Rose Jones is a woman."
  taken out; Title blank -> her Title from her other rows. Jones row 3:
  "At the time, Jones did not feel like she could report the assaults..."
  taken out. Trimmed at sentence boundaries: academic research row ("And
  that is still the case nearly 30 years after... Farmer v Brennan..."
  out), activists row ("Now, they are working to change policies..."
  out), Dee Farmer row ("Farmer v Brennan centered on the alleged prison
  rape of a trans woman, Dee Farmer." out), NCTE survey row ("Once they are
  incarcerated, trans people are at significantly higher risk of
  violence." out), Bamby Salcedo row ("Salcedo is now the CEO and
  President of Los Angeles-based TransLatin@ Coalition..." out). Removed
  (actions): Farmer row 15 ("Activists like Dee Farmer are still fighting
  to institute..." -- now her Source Justification, blank before); George
  W. Bush row 34 (signed PREA into law); Gavin Newsom row 47 (signed SB
  132 into law); Orthwein row 51 ("...filed a lawsuit on June 11..." --
  now her Source Justification, replacing "now a friend of Jones", which
  it contains). Added: Named Organization / Department of Justice, "The
  Department of Justice tells CNN that the Federal Bureau of Prisons
  “follows and enforces PREA standards...”". Now 59 rows.

- **Phase 4 Step 1 (mechanical fixes), original GT-II batch, 2026-11-18.**
  Settled rules only; changes approved as one batch table (Part A).
  Sourced Statements made to match the article text exactly -- curly vs
  straight quote marks and apostrophes as the text has them: `63` row 10;
  `87` rows 5, 8, 15; `89` rows 2, 3, 4, 5, 7, 9, 11, 12, 13, 15;
  `94-Met_Gala.csv` rows 1, 3, 6, 7, 9; `109-Waymo_West_Valley.csv` rows
  1, 2, 4, 7, 8, 12; `116` rows 2, 4. Quotation marks the text doesn't
  have around these sentences, removed: `122-SJ_Climate.csv` rows 2, 4, 5,
  6, 9, 10, 11, 15, 18. Title of Source: words the article doesn't
  capitalize lowercased (capitals the reporter uses, including
  organization names, kept), and a leading article dropped: `56` rows 7, 8
  "Executive Director of San Francisco Pride" -> "executive director of
  San Francisco Pride"; `61-AV_Atlantic_Current.csv` rows 1, 2, 3, 8
  "Scientists" -> "scientists"; `61-AV_Atlantic_Current.csv` row 4
  "Researcher, Inria Centre de recherche Bordeaux Sud-Ouest" ->
  "researcher, Inria Centre de recherche Bordeaux Sud-Ouest";
  `61-AV_Atlantic_Current.csv` rows 5, 6, 7, 10, 11 "Professor, Potsdam
  Institute for Climate Impact Research" -> "professor, Potsdam Institute
  for Climate Impact Research"; `74-AV_San_Ramon_Pride.csv` row 10 "Former
  mayor" -> "former mayor"; `74-AV_San_Ramon_Pride.csv` row 16 "Teacher in
  San Ramon" -> "teacher in San Ramon"; `75` row 5 "Law Professor and
  Associate Dean at Case Western Reserve University" -> "law professor and
  associate dean at Case Western Reserve University"; `77` rows 3, 4, 5
  "Chief Legal Counsel at the San Francisco Sheriff's Office" -> "chief
  legal counsel at the San Francisco Sheriff's Office"; `77` rows 7, 8
  "Drug Rehab Director" -> "drug rehab director"; `77` rows 11, 12 "Nearby
  Business Manager" -> "nearby business manager"; `87` row 5 "Shipping
  executives" -> "shipping executives"; `94-Met_Gala.csv` rows 1, 2, 3, 4,
  7 "Co-chair for the 2026 Met Gala" -> "co-chair for the 2026 Met Gala";
  `94-Met_Gala.csv` row 5 "Museum's director and CEO" -> "museum's
  director and CEO"; `94-Met_Gala.csv` rows 6, 8, 9 "Curator in charge of
  the Costume Institute" -> "curator in charge of the Costume Institute";
  `108-Youth_POC_Climate.csv` rows 6, 17, 18, 20 "YVA Organizer" -> "YVA
  organizer"; `108-Youth_POC_Climate.csv` rows 7, 8, 10, 11, 16, 19 "YVA
  Education and Organizing Coordinator" -> "YVA education and organizing
  coordinator"; `108-Youth_POC_Climate.csv` row 15 "YVA Program Director"
  -> "YVA program director"; `109-Waymo_West_Valley.csv` rows 1, 2
  "Spokesperson" -> "spokesperson"; `112-Hantavirus_Ship.csv` row 10
  "Federal officials" -> "federal officials"; `112-Hantavirus_Ship.csv`
  rows 11, 13, 15 "CDC Incident Manager" -> "CDC incident manager"; `113`
  row 4 "President of the Personal Insurance Federation of California" ->
  "president of the Personal Insurance Federation of California"; `113`
  row 5 "Director of strategic communication for the Insurance Information
  Institute" -> "director of strategic communication for the Insurance
  Information Institute"; `113` row 7 "Head of 360 Value" -> "head of 360
  Value"; `113` rows 8, 9, 27, 69, 96 "Executive Director of United
  Policyholders" -> "executive director of United Policyholders"; `113`
  row 16 "Co-founder of the West Slope Foundation" -> "co-founder of the
  West Slope Foundation"; `113` rows 18, 19, 20, 21, 22 "Inventor of
  Component-Cost Algorithms" -> "inventor of component-cost algorithms";
  `113` row 23 "Professor and Author" -> "professor and author"; `113`
  rows 35, 63, 64 "Head of 360Value" -> "head of 360Value";
  `122-SJ_Climate.csv` rows 10, 11, 12 "Chief of Staff for terming out
  incumbent Vice Mayor Pam Foley" -> "chief of staff for terming out
  incumbent Vice Mayor Pam Foley"; `122-SJ_Climate.csv` rows 17, 18, 19
  "Event Moderator" -> "event moderator". Row numbers:
  `89-Sydney_SCC_Trump_Cuts.csv` ("1.0, 2.0..." -> "1, 2...");
  `109-Waymo_West_Valley.csv` ("1.0, 2.0..." -> "1, 2...").
- **Phase 4 Step 1 Part B (text fidelity), original GT-II batch,
  2026-11-19.** Approved item by item with full before-rows. Sourced
  Statements only; row numbers are before the splits unless shown with an
  arrow. Fragment -> the full sentence(s) from the article, lead-in included
  (v61 Note 12a): `56` rows 4, 5; `74-AV_San_Ramon_Pride.csv` rows 1, 7, 17;
  `113` rows 62, 81; `122-SJ_Climate.csv` rows 16, 17. A row crossing a
  paragraph break -> one row per paragraph, other fields copied (Note 12a;
  breaks checked against the live pages): `77` row 13 -> rows 13, 14, row 14
  -> rows 15, 16; `109-Waymo_West_Valley.csv` row 9 -> rows 9, 10, row 14 ->
  rows 15, 16; `113` row 80 -> rows 80, 81, row 97 -> rows 98, 99. Wording
  made to match the article text (typos, dropped/changed words, dashes,
  ellipses, quote marks): `64` row 7; `87` rows 2, 7, 10, 13, 18, 21; `89`
  rows 10, 17; `94-Met_Gala.csv` rows 2, 4, 5, 8; `113` rows 39, 66, 77;
  `122-SJ_Climate.csv` rows 1, 3, 7, 8, 12, 14. Rows renumbered after the
  splits: `77` (19 -> 21 rows); `109-Waymo_West_Valley.csv` (14 -> 16 rows);
  `113` (97 -> 99 rows). Same-source merges deferred until after 4b (Step
  3).
- **Matching GT fix, 2026-11-19 (Phase 4 Step 1 Part B).** `113` row 35
  Sourced Statement "...on top of demand surge." -> "...on top of demand
  surge ." -- the article text and its PDF both have the space before the
  period. (The same day the `113` text had "premiums" restored and a
  paragraph break moved, from the PDF; no GT row quoted that sentence.)
- **Phase 4 Step 1 (text fidelity) on `55-Pentagon_Lab.csv`, 2026-11-19.**
  Approved with full before-rows. Row 1 fragment -> the full sentence ("He
  didn’t learn about “Quantum City” from the governor or the mayor, but from
  a teach-in at his local library, where community activists explained...",
  v61 Note 12a). Rows 5 and 10 Title of Source "Assistant deputy director
  for the Illinois Department of Commerce and Economic Opportunity, State
  Official" -> "assistant deputy director for the Illinois Department of
  Commerce and Economic Opportunity, state official" (the text writes both
  in lowercase). Row 10 Sourced Statement: a stray leading ". " removed.
  Paragraph crossings -> one row per paragraph (other fields copied): row 5
  (Glezer) -> 2 rows; row 15 (Torrence) -> 3 rows. Rows renumbered (23 -> 26
  rows): old 5 -> 5, 6; old 6-14 -> 7-15; old 15 -> 16, 17, 18; old 16-23 ->
  19-26. Row numbers above are before the splits.
- **Phase 4 Step 2 batch 1 (4a recall, 4b precision, SJ, Title), original
  GT-II batch, 2026-11-19.** Approved item by item with full before-rows.
  `75` Title of Source for elected officials, with party and constituency as
  written (v61): row 2 Yvette Clarke "New York Representative" ->
  "Democratic Rep. of New York; chair of the caucus"; rows 3, 4 Terri Sewell
  "Alabama Representative" -> "Rep., Democrat from Alabama".
  `76-AV_Ghiradelli_Deal.csv` rows 3, 4 Anant Bhalla Title "JAB Insurance
  U.S. Holdings CEO" -> "CEO of 1823 Partners; global head of insurance at
  JAB Holdings" (as the text has it); rows 1, 2 ("sources say") Unnamed
  Person -> Unnamed Group of People (plural). `121`: row added for "Meta had
  78,865 employees at the end of 2025, up 6% from a year earlier, according
  to the company." (Named Organization / Meta; now row 5, old rows 5-8 ->
  6-9). `56`: 3 rows added -- staff at a Noe Valley Philz "told the
  Chronicle..." (Unnamed Group of People, SD "staff"); the Philz baristas'
  Change.org petition against the move (Unnamed Group of People, SD "Philz
  baristas"); State Sen. Scott Wiener, D-San Francisco, "spoke out against
  the decision" (Named Person, Title "State Sen. D-San Francisco"); now rows
  6, 7, 8, old rows 6-8 -> 9-11. Row 2 (Sadarangani) Sourced Statement
  extended to the whole paragraph: "In the statement, Philz CEO Mahesh
  Sadarangani apologized for the decision to remove Pride flags..." added
  before the quotes (Note 12a).
- **Phase 4 Step 2 batch 1, held item, 2026-11-19.** `75` row 1
  ("...according to an NPR analysis conducted earlier this year."): Document
  -> Named Organization / NPR, Source Descriptors "analysis" -> blank (an
  outlet credited by name for its own analysis, v62 item 10 / Minor 22);
  Source Justification (the paragraph before, about redistricting experts)
  -> blank -- it is about the experts, not the analysis.
- **Phase 4 Step 2 batch 1, held item, 2026-11-19.** `56` row 5 (the
  confidential memo, Document): kept as Document; Source Justification
  "confidential memo issued by Sadarangani to Philz employees" -> "in a
  confidential memo to Philz employees" (the text's own words).
- **Phase 4 Step 2 batch 2a, original GT-II batch, 2026-11-19.** Approved
  item by item with full before-rows. `116`: Source Justification "null" ->
  blank (old rows 3, 4, 8); old row 1 SJ (three passages joined with "...")
  -> "The Sunday gathering was organized by Freedom250, a private group
  established by the Trump administration"; old row 2 SJ (event context and
  Hegseth) -> blank; old row 5 (the heckler who accused attendees of
  "supporting a pedophile") Unnamed Group of People / SD "attendees" ->
  Unnamed Person / SD "heckler", SJ (about the counterprotests) -> blank --
  the speaker is the heckler; old row 4 Weissman Title "Co-President, Public
  Citizen" -> "Public Citizen Co-President" (as written). 3 rows added:
  Freedom From Religion Foundation's truck "calling to “Celebrate Democracy,
  Not Theocracy.”" (Named Organization); federal officials "announced
  “Operation Summer Surge”..." (Unnamed Group of People, Title "federal
  officials"); "the administration says" the Interior fund will pay for
  beautification (Named Organization / Trump administration). Now 12 rows:
  old 1-4 -> 1-4, 5 -> 6, 6-9 -> 9-12; new rows 5, 7, 8. `63`: no change
  (row 5 "Many state and federal policymakers..." kept, user). `64`: row
  added for "...authorities said Saturday." (p2; Unnamed Group of People,
  Title "authorities"; now row 1, old rows 1-10 -> 2-11); old row 8 extended
  with the next sentence of its paragraph ("He was charged with battery and
  burglary..."), old row 9 with "He completed the program in 2024 and the
  charges were discontinued." (implied attribution, Notes 9/10); old rows 2,
  3 Maurer Title "Chief Deputy, Hillsborough County" -> "Hillsborough County
  Chief Deputy" (as written).
- **Phase 4 Step 2 batch 2b, original GT-II batch, 2026-11-19.** `63`: 2
  rows added for "Sen. Bernie Sanders, I-Vt., and Rep. Alexandria
  Ocasio-Cortez, D-N.Y., proposed a federal moratorium on new data centers
  in late March." -- one per lawmaker (Named Person / Bernie Sanders, Title
  "Sen. I-Vt."; Named Person / Alexandria Ocasio-Cortez, Title "Rep.
  D-N.Y."); now rows 9, 10, old rows 9-10 -> 11-12.
- **Story 67 (California_Ships_Whales) research rows, 2026-11-19 (Phase 4
  Step 2 batch 2b, decided with its twin 70).** Rows 10 and 11 ("Recent
  research out of Sonoma State University and the Marine Mammal Center in
  Sausalito estimates...") were the same sentence typed Named Organization
  twice (Sonoma State University; Marine Mammal Center in Sausalito) -> one
  row, Document, Source Descriptors "research" (user: we don't know the two
  organizations produced the work as their own; it may be affiliated
  researchers' work). Old rows 12, 13 -> 11, 12; now 12 rows. Row 12 (now
  11, NOAA "Federal scientists... believe...") kept.
- **Phase 4 Step 1 (text fidelity) on `67-California_Ships_Whales.csv`,
  2026-11-19.** Approved with full before-rows. Sourced Statements: line
  breaks at the end of the cell removed (rows 1, 2, 3, 4, 5, 7, 8, 9, 11,
  12); straight apostrophes -> curly as the text has them (rows 6, 9). Title
  of Source: "Director of marine resource protection, California Marine
  Sanctuary Foundation" -> "director of marine resource protection,
  California Marine Sanctuary Foundation" (rows 3, 5-9); "Federal
  Scientists" -> "federal scientists" (row 11) and "Researchers" ->
  "researchers" (row 4) -- capitals only from the start of the sentence;
  "South Bay Representative" -> "South Bay Rep." (rows 1, 12; elected
  official, as written). Same values as the twin file 70.
- **Phase 4 Step 2 on `67-California_Ships_Whales.csv` (dual pair with 70;
  technical fixes only), 2026-11-19.** Jessica Morten rows 3, 5, 6, 7, 8, 9:
  Title of Source "director of marine resource protection, California Marine
  Sanctuary Foundation" -> "director of marine resource protection at the
  California Marine Sanctuary Foundation" (as written); Source Justification
  "worked at the program for the last 10 years" -> "who has worked on the
  program for the last 10 years" (the text's words). Kept as AV annotated
  them (interpretive differences from SZ's 70, for ICR): row 1 Liccardo (the
  whole Save Willy Act paragraph as Named Person); no row for "The
  legislation introduced Wednesday by Sam Liccardo...".
- **Phase 4 Step 2 batch 2c (dual pairs; technical fixes only), original
  GT-II batch, 2026-11-19.** `74-AV_San_Ramon_Pride.csv`: Title of Source as
  written -- Anuradha Gupta rows 5, 6, 8 "Chapter President, PFLAG
  Danville-San Ramon Valley" -> "chapter president of PFLAG Danville-San
  Ramon Valley"; Robert Jweinat row 13 "San Ramon City Councilmember" ->
  "Councilmember". The rest of the difference from SZ's copy is the open
  AV/SZ Sourced Statement disagreement (joint discussion), left as is.
  `94-Met_Gala.csv`: Bolton row 6 extended with the lead-in sentence of his
  paragraph ("Andrew Bolton, the curator in charge of the Costume Institute,
  weighed in on why, exactly, “Costume Art” felt like the right choice for
  the gallery’s debut show.", Note 12a); old row 4 ("From that homemade
  tennis skirt to full-fledged fashion icon, Williams has come to...")
  removed -- the reporter's characterization, no attribution; straight
  apostrophes curled as in the text (Hollein Title "museum’s"; Williams SJ
  "year’s", "body’s"). Now 8 rows (old 5-9 -> 4-8). Row 7 ("As Williams
  pointed out, Bolton stressed...", typed Venus Williams) left as an ICR
  disagreement (user: Bolton is the source).
- **Minor 22 Document-name fixes after a re-read of the v61/v62 rules,
  2026-11-19.** `64` row 11 ("Hillsborough County Court records also
  showed..."): Name "Hillsborough County Court records" -> blank, Source
  Descriptors -> "Hillsborough County Court records" (court records
  credited; the court named in passing). `61-AV_Atlantic_Current.csv` row 9
  (the Science Advances research): Name "Science Advances" -> blank; Source
  Justification -> "published in the journal Science Advances; explored four
  different ways of using real-world observations to assess the models"
  (user: publication signals the research's scientific standing).
- **Follow-up, 2026-11-19.** `64` row 11 Source Descriptors "Hillsborough
  County Court records" -> "court records": v61 SD rule 4 keeps a place only
  for a person or group of people (an organization leaves out where it is
  based); a document's SD is its genre. The county stays in the Sourced
  Statement.
- **Phase 4 Step 2 batch 3a (dual pair 52), original GT-II batch,
  2026-11-19.** `52-Swalwell_Allegations.csv`: Source Descriptors "Livermore
  resident" (Julie Valentine) and "Hayward resident" (Melissa Estepa) ->
  blank (item 70: the text says "of Livermore", "of Hayward"; a Name of
  Place is not a descriptor). Title of Source for elected officials, as
  written: Eric Swalwell -> "Rep.", Ro Khanna -> "Rep. D-Santa Clara", Sam
  Liccardo -> "Rep. D-San José", Jared Huffman -> "Rep. D-California". Name
  of Source typos: "Alameda County District Attourney's Office" -> "Alameda
  County District Attorney’s Office", "House Commitee on Ethics" -> "House
  Committee on Ethics". 4 rows added: Julie Valentine "questioned the timing
  of the allegations" (her lead-in paragraph; the same sentence is still her
  quote row's SJ -- for the item 58 sweep); Swalwell "announced Sunday that
  he was dropping out..."; Named Organization / San Francisco Chronicle for
  the accuser's allegations from its published interviews (relayed through
  an outlet, item 73); Named Organization / CNN ("reported that three
  additional women..."). Now 17 rows: new rows 1, 5, 6, 7; old 1-3 -> 2-4,
  4-13 -> 8-17.
- **Phase 4 Step 2 batch 3c (dual pairs; technical fixes), original GT-II
  batch, 2026-11-19.** `112-Hantavirus_Ship.csv`: old row 14 ("The risk to
  the general public remains low, WHO and the CDC say.", Name "World Health
  Organization and Centers for Disease Control and Prevention") -> one row
  per organization (World Health Organization; US Centers for Disease
  Control and Prevention -- new row 15, old 15 -> 16); row 9 Name "Centers
  for Disease Control and Prevention" -> "US Centers for Disease Control and
  Prevention" (as row 7 and the text); Title of Source: David Fitter "CDC
  incident manager" -> "Dr., incident manager for the CDC’s hantavirus
  response", Stephen Kornfeld "Oregon oncologist" -> "Dr., Oregon
  oncologist" ("Dr., <position>" form). `109-Waymo_West_Valley.csv`: row
  added for "Some residents have expressed concerns about how autonomous
  vehicles could affect traffic safety in Los Gatos..." (Unnamed Group of
  People, SD "residents"; now row 8, old 8-16 -> 9-17); row 5 Title "Elected
  Leaders in West Valley" -> "elected leaders in the West Valley" (as
  written, sentence-start capital lowercased).
- **Phase 4 Step 2 batch 3c, held items, 2026-11-19.**
  `109-Waymo_West_Valley.csv`: Rob Moore row 7 (p8) -> "Moore said Waymo
  previously told him... Still, he said he sees potential safety benefits in
  autonomous vehicle technology." -- his two sentences in the paragraph,
  joined; the residents' sentence between them is its own row (Note 12a).
  (His "Still, he said..." sentence was added.)
- **Phase 4 Step 2 batch 4c (dual pair 122; technical fixes), original GT-II
  batch, 2026-11-21.** `122-SJ_Climate.csv` (now 26 rows, in text order):
  v61 Minor 23 (a collective term the story defines by naming its members
  resolves to those sources, one row each; p2 names the five candidates) --
  old row 1 ("...all of the candidates expressed broad support...", Unnamed
  Group of People "candidates") -> five Named Person rows (Scott Hughes,
  Mike Hennessy, Gordon Chester, Genny Altwer, Rick Ator; Title, SD and SJ
  as on each one's other rows), now rows 1-5; old row 5 ("But Altwer and her
  fellow candidates also voiced concern...", Unnamed Group of People) ->
  four Named Person rows for the fellow candidates (Hughes, Hennessy,
  Chester, Ator), now rows 9-12 after Altwer's row 8. Anika Khanna (now rows
  24-26): Title "event moderator" -> "event’s moderator" (as written),
  Source Descriptors blank -> "sophomore, member" (p21 "a sophomore at
  Lynbrook High School", p22 "a Silicon Valley Youth Climate Action
  member"). Old rows 2-4, 6-19 keep their order (now 6-8, 13-26).
- **Phase 4 Step 2 batch 5a (dual pair 108; technical fixes), original GT-II
  batch, 2026-11-22.** `108-Youth_POC_Climate.csv` (now 23 rows): Title of
  Source as written: Aniya Butler "YVA organizer" -> "YVA’s hip-hop and
  climate justice initiative coordinator" with Source Descriptors
  "organizer" (p8 "YVA organizer Aniya Butler", p29 "Aniya is YVA’s hip-hop
  and climate justice initiative coordinator"); Hannah Estrada "YVA
  education and organizing coordinator" -> "YVA’s education and organizing
  coordinator"; Sim Bilal Title -> blank, Source Descriptors "organizer"
  (non-credentialing; "an organizer with Youth Climate Strike in Los
  Angeles"). Youth vs. Apocalypse's p7 suggestion row -> "One suggestion:
  Use this upcoming Earth Day, April 22, to plan activities at your school.
  YVA suggests young people offer to do presentations..." (sentences 2-3 of
  the paragraph). ("One suggestion..." added; Bilal Name "Sam Bilal" -> "Sim
  Bilal", Title was "LA Youth Climate Strike Organizer".) Diane Feinstein
  Title "California Senator" -> "California senator" (as written). Row 2
  ("It wasn’t until Youth vs. Apocalypse (YVA) came to her class that it
  clicked for her: “They’re poisoning our lungs...”") Named Organization /
  Youth vs. Apocalypse -> Named Person / Lizbeth Ibarra (her realization, in
  her words; SJ as on her other rows). Row added for Ibarra's p5 quote
  ("“You can’t talk about the [climate] problem as though it’s about polar
  bears and melting ice caps...” says Ibarra...") -- now row 4, old rows
  4-22 -> 5-23.
- **Phase 4 Step 2 batch 5b (77), original GT-II batch, 2026-11-22.**
  Approved item by item with full before-rows. `77` (now 22 rows): Daniel
  Lurie Title "San Francisco Mayor" -> "Mayor" and Paul Miyamoto Title
  "Sherriff" -> "Sheriff" (as written); old row 15 removed ("Donald Kempf,
  40, was curious about how much help the center could provide." -- a
  reported feeling, v62 item 21); Matt Dorsey row trimmed to "He later
  called back to say the sheriff assured him people can be detained at the
  center." (the next sentence, "...Dorsey again hung up to call the mayor’s
  homelessness chief...", is the reporter's narration). Rows added:
  "Officials are pitching it not as a jail but as a “third place” — an
  alternative to jails and hospitals." (Unnamed Group of People, Title
  "officials"); "Homeless people who were near the RESET Center on
  Thursday... said they would take advantage of its showers and comfortable
  chairs..." (Unnamed Group of People, SD "homeless").
- **Phase 4 Step 2 batch 5c (87), original GT-II batch, 2026-11-22.**
  Approved item by item with full before-rows. `87` (now 20 rows): row 1
  (Trump administration) extended to the full sentence -- "The Trump
  administration is trying to do that: It announced it had successfully
  guided two US vessels out of the strait Monday." (v61 Note 12a); removed:
  Scott Bessent "He hopes Project Freedom will be able to get that oil to
  the market soon." (a reported hope, v62 item 21), "But that is a largely
  symbolic gesture, because the strait remains closed." (the reporter's
  analysis, not Bessent's), and the oil analysts row "Project Freedom falls
  well short of a full reopening... a long process that oil analysts believe
  will be required..." (a general reported belief inside the reporter's
  sentence, item 21). Rows added: Named Person / Donald Trump, President,
  for "Energy prices did not tumble after President Donald Trump announced
  the new US effort to “guide” ships through the Strait of Hormuz."; Named
  Organization / United Arab Emirates for "...in an attack that officials
  there blamed on Iranian drones." (as the file's "Officials in Iran" ->
  Iran).
- **Title of Source format: titles separated by "; ", "Dr." as its own title
  (v62 item 24), 2026-11-22.** `112-Hantavirus_Ship.csv` "Dr., <position>"
  -> "Dr.; <position>": Stephen Kornfeld (rows 1-5, "Dr.; Oregon
  oncologist") and David Fitter (rows 11, 13, 16, "Dr.; incident manager for
  the CDC’s hantavirus response").
- **Phase 4 Step 2 batch 5e (dual pair 55; technical fixes), original GT-II
  batch, 2026-11-22.** `55-Pentagon_Lab.csv` (now 30 rows): Name "Janya
  McGruder" -> "Jayna McGruder"; Jerry Whirley SJ "South Shore Home Owner"
  (not in the text) -> "heard that a $9 billion quantum-computing campus was
  coming a few blocks from his South Shore home" (item 70); Torrence row
  trimmed to "Torrence also spoke to what he said were larger politics at
  play." (the next sentence is the reporter's background); old row 8 ("The
  state’s quantum bet is also about jobs, and supporters say the federal
  subsidies are necessary. Residents said the numbers are small — and
  lopsided.") split one row per source -- residents keep the second
  sentence; new row for the first (Unnamed Group of People, SD
  "supporters"). Rows added: "Since then, they have committed hundreds of
  millions in taxpayer money... to help build what they say will be the
  country’s first large-scale quantum computer." (Unnamed Group of People,
  Title "state and city officials"); "The petition met the signature
  threshold, but in January the Chicago Board of Elections voted to strike
  it from the March ballot, agreeing with attorneys who argued..." (Named
  Organization / Chicago Board of Elections -- its own ruling; the
  attorneys' argument is nested). Row added: "All of which, advocates argue,
  are being concentrated in already overburdened communities." (Unnamed
  Group of People, SD "advocates", as the SZ copy).
- **Phase 4 Step 2 batch 5e, held item 5e.3, 2026-11-22.**
  `55-Pentagon_Lab.csv`: Title of Source as written -- JB Pritzker
  "Governor" -> "Gov." ("Gov. JB Pritzker"); Brandon Johnson "Mayor" ->
  "Chicago Mayor" (first formal introduction, p10: "The office of Chicago
  Mayor Brandon Johnson..."; v61 elected-office rule, constituency as
  written); Eliza Glezer -> "assistant deputy director for the Illinois
  Department of Commerce and Economic Opportunity; state official" (p9 and
  p18; titles joined with "; ", v62 item 24).
- **Phase 4 Step 2 batch 6c (89), original GT-II batch, 2026-11-22.** `89`
  (now 29 rows): Source Justification "null" -> blank on 26 rows (all but
  the row with a real SJ). Rows added: Named Organization / Santa Clara
  County for "The county said about 265 of those positions are currently
  filled, or roughly 40%, but Williams said he is hoping to avoid any
  layoffs." (two sources in one sentence; James Williams keeps his row);
  Named Person / Matt Mahan, Title "Mayor" (as written), for "San José,
  whose Mayor Matt Mahan heavily touted his work to spend more of the city’s
  affordable housing funds..."
- **Phase 4 Step 3 (merges) batch 3-1, original GT-II batch, 2026-11-24.**
  Same source, same paragraph -> one row (v61 Note 12a); other fields from
  the first row; no differing SJs. `89` (now 26 rows): James Williams rows
  merged -- "“We will be attempting very vigorously...” he said." + "What
  exactly happens to those employees...", "Williams said the county has been
  aiming to preserve or expand services..." + "He pointed to plans for new
  “satellite clinics...”", "Williams said the county has “moved
  mountains”..." + "But he called directly on the governor and
  legislature...". `108` (now 22 rows): Lizbeth Ibarra "“Still, growing up,
  we’d talk about Chevron in a good way...” Ibarra recalls." + "It wasn’t
  until Youth vs. Apocalypse (YVA) came to her class...". `109` (now 16
  rows): Kitty Moore "Cupertino Mayor Kitty Moore said city officials are
  also approaching Waymo cautiously..." + "She said Waymo representatives
  told city officials...".
- **Phase 4 Step 3 (merges) batch 3-2, original GT-II batch, 2026-11-24.**
  `121` (now 8 rows): The New York Times "The New York Times reported
  Wednesday that some employees had signed petitions..." + "The Times also
  reported that workers exchanged messages remotely..." merged. `122` (now
  23 rows): Rick Ator "“My concern is making sure we’re not pushing
  unexpected costs onto existing homeowners,” Ator told San José
  Spotlight..." + "“If the code is structured to apply only at the natural
  point of replacement...”" merged; Scott Hughes "Meanwhile, Hughes said he
  has supported efforts to bring more trees..." + "That includes a $20,000
  rebate program..." merged; Gordon Chester duplicate row "“We don’t need
  these data centers. I am okay with seeing them be far away...”" removed
  (the row "“I’m not the biggest fan of data centers, generally,” Chester
  said at the forum..." already holds the whole paragraph).
- **Phase 4 Step 3 batch 3-3, original GT-II batch, 2026-11-24.** `112` (now
  one row fewer): "Federal officials are working with medical staff to
  “ensure that they have the appropriate disposition,” the officials said in
  the agency’s first briefing..." (Unnamed Group of People / Title "federal
  officials") merged into the US Centers for Disease Control and Prevention
  row "However, the American passengers who are in quarantine in Nebraska
  are “encouraged to stay”..., officials with the CDC said Wednesday." (Note
  8: "the agency" is the CDC; same source, same paragraph).

### GT-I update

Changes to the GT-I files (`benchmarking/GT data/GT-2026/`, stories 1-43) from Phase 4, which started 2026-11-18. The same changes are also recorded under item 76 in `development-of-v59.md`.

- **Batch fidelity pass, 32 files, 226 rows, 2026-11-18.** Sourced
  Statements made to match the story text exactly: (A) 217 rows had
  spaces or line breaks at the start or end, now removed (stories 2-7,
  9-16, 18-20, 22-25, 27, 29, 31-34, 38-40, 42); (B) 5 rows had a double
  space inside where the text has one (16 rows 12, 13; 33 row 14; 34 rows
  4, 5); (C) 3 rows began with a quote mark the text doesn't have (37
  rows 1, 3; 40 row 2) -- removed, and those texts are to be checked
  against the PDF in case the text dropped it; (D) `6-OUSD-basic-job.csv`
  row 15 (Marina Muñoz) skipped the sentence "Muñoz spoke in Spanish and
  had her comments translated by OUSD interpreter Rebeca Walker-Marquez."
  between her two quotes and had a straight quote mark where the text has
  a curly one -> now the full continuous passage from the text. (E)
  `22-Judge-rules-breona-taylor.csv` had duplicate row numbers (1-4, then
  2-13) -> renumbered 1-16.

- **Whitespace in all fields, 39 files, 2026-11-18.** Spaces or line breaks
  at the start or end removed: Sourced Statements 47 more rows (missed
  by the first pass), Name of Source 34, Title of Source 30, Source
  Justification 119. Double spaces and line breaks inside a field
  collapsed to one space in 2 Titles (23 row 14 "hearing officer,
  Georgia Public Service Commission"; 11 row 1 "Outreach specialist,
  Nebraska AIDS Project") and 7 Source Justifications (27 row 18; 11 rows
  16-20; 41 row 1). Sourced Statements keep their inner line breaks where
  the text has them (17 rows 6, 19 -- a hard-wrapped text).

- **`7-District7-Candidate-Sues.csv` row 1 (Tonya Love), 2026-11-18.** The
  Sourced Statement ran across two paragraphs -> split into two rows, one
  per paragraph, same fields: "According to a post by Love on Instagram
  earlier this hour, the judge did not rule in Love’s favor." and "As a
  result, Love said that she would not be running for the District 7
  seat." Now 22 rows.

- **`29-tim-scott-deregulate.csv` row 2 (Document, the ROAD to Housing
  Act), 2026-11-18.** Same: split into "Nevertheless, the bill does
  include at least one idea to increase housing supply." and "That
  includes a repeal of the federal regulation requiring that manufactured
  housing sit on a permanent steel chassis." Now 18 rows.

- **Story texts 35 and 36 re-flowed, 2026-11-18 (text files, no CSV
  change).** Both were hard-wrapped (cut at a PDF's line width); now one
  paragraph per line, using the PDFs the user supplied. Only whitespace
  changed. All rows in `35-Trump-Tariffs-Rattling-Meta.csv` (7) and
  `36-Whats-Wrong-With-Apple.csv` (16) match the text word for word. In
  36 the photo caption moved to the header and a newsletter promo line was
  removed (neither had a row). Details in
  `extracted_articles_boilerplate/REFLOW_2026-11.md`. Story 26 turned out
  not to be wrapped (left as is); 17 still needs a PDF.

- **`1-Harris-Poll.csv`, Phase 4, 2026-11-18.** Rows 1 and 2: the final
  period, cut off, restored from the text. Row 16 removed ("The latest
  poll's survey period partially overlapped with the Aug. 19-22 Democratic
  National Convention... it remains to be seen whether the same level of
  enthusiasm for Harris will continue." -- the reporter's analysis, not a
  poll finding). Aimee Allison's Source Justification "A liberal group..."
  -> "a liberal group..." as in the text. Now 16 rows.

- **`2-Best-DDR-player.csv`, Phase 4, 2026-11-18.** One row per source per
  paragraph: Chris Chike's rows 1-2 merged into "“It wasn't anything
  serious,” he told Gazetteer SF over FaceTime from his home in Hayward.
  “I wasn't extremely good at the game or anything.”" (both had been
  quote fragments without quote marks or attribution); his rows 9-10
  merged into the full paragraph "“I don’t think I would have any other
  reason... like me.”"; Roger Clark's rows 12-13 merged into the full
  paragraph "“Whenever you meet someone who plays DDR... working in the
  tech industry here." Rows 3 (comments, Unnamed Group of People) and 4
  (one commenter, Unnamed Person) each held half of one sentence -> both
  the full sentence "Multiple contemporaneous comments called him a
  “god”; one commenter heralded the task as “the pinnacle of human brain
  potential.”" Added: Clark, "Clark recalled the song required about 800
  steps at 220 beats per minute for two minutes, a feat of agility not
  entirely dissimilar to an Olympic sprint."; Chike, "He describes it as
  a blessing in disguise." Chike's Source Justification on all his rows
  -> "Chris Chike is, by most accounts, the greatest Dance Dance Revolution
  player in the world." (was two versions, one with "he told Gazetteer SF
  over FaceTime from his home in Hayward"). Now 19 rows.

- **`3-SF renewed-sweeps.csv`, Phase 4, 2026-11-18.** Row 2 removed
  (Document, "Despite Public Works’ policy and procedure for documenting,
  bagging, and tagging most belongings" -- half a sentence; the policy is
  context here, and its rules are quoted in its own row); row 3 (Antonio
  Robinson) -> the full sentence "Despite Public Works’ policy... Robinson
  said he was unable to find any of his property...". Joey O’Brien's row
  "More than anything, however, he begged..." now starts at "O’Brien was
  polite and contrite, admitting that he had broken several windows in
  the past while suffering a mental breakdown." (attributed, same
  paragraph); its Source Justification, which was that sentence -> his
  Source Justification from his other rows. Hazel Williams row (started
  mid-sentence, "as a recent batch of public records...") -> the full
  sentence "Despite promises from law enforcement and city officials that
  this property will be preserved, there is plenty of evidence... taking
  them to the city’s storage facility on Cesar Chavez St." Robinson's row
  8 ("“All they do is tell me to go back to the Hunters Point Navigation
  Center...”"): Source Justification "“I had bad experiences,”" (a piece
  of the row's own quote) -> his opening-paragraph Source Justification
  ("On a grey Thursday morning last week, Antonio Robinson watched...").
  His other local Source Justifications left for the SJ sweep. Now 24
  rows.

- **`5-Humboldt-SF-homeless-bus.csv`, Phase 4, 2026-11-18.** Removed row 1
  (Named Organization / The Standard, "County supervisors learned of this
  from reporting in The Standard" -- narration of how they found out).
  The letter's two rows merged into the full paragraph ("“We are
  concerned...,” Humboldt’s letter says. “We urge you...”"); Paul Boden's
  two quote rows merged into the full paragraph ("“The aim isn’t to
  achieve a goal...,” Boden said. “The No. 1 answer...”"). Added: Named
  Organization / San Francisco mayor’s office, "When asked for comment,
  the San Francisco mayor’s office said it had not received the letter."
  Rex Bohn's Source Justification "said at Tuesday’s board meeting." ->
  blank (attribution only). Now 10 rows.

- **`6-OUSD-basic-job.csv`, Phase 4, 2026-11-18.** One row per source per
  paragraph -- merged: Max Simmons (2 rows), Stuart Loebl (2), Valarie
  Bachelor (2; one had lost its period, the other its opening quote), the
  report (2: "At the 40 schools... according to the report. Stage
  three... every five years."). "School sites will continue to be tested
  in five stages, district leaders said": final period restored. Removed:
  "Community members, many of whom raised concerns two weeks ago, were
  skeptical." (an attitude, not a statement). Title of Source: Simmons
  "OUSD Student Director" / "OUSD’s student directors" -> "OUSD’s student
  director" (the text's "one of OUSD’s student directors", singular);
  Loebl "Sixth-grade science teacher, Frick United Academy of Language" ->
  "sixth-grade science teacher at Frick" (as written); Bachelor "Director"
  -> "Director; chair of OUSD’s facilities committee", Source
  Justification "chair of OUSD’s facilities committee" -> blank. Now 20
  rows.

- **`7-District7-Candidate-Sues.csv`, Phase 4, 2026-11-18.** Full sentence
  for two rows that started mid-sentence: Kaufman Legal Group ("Love’s
  lawyers, from the political law firm Kaufman Legal Group, said in their
  petition that Love had gathered 67 signatures...") and Tonya Love ("In
  an Instagram post last week, Love told her followers..."). Quote marks
  restored on the lawyers' brief, Judge Michael Markman's "My
  understanding..." and Love's "We are moving forward..."; final period
  restored on "The lawyers asked the court...". Name "Voter's Choice Act"
  -> "Voter’s Choice Act" (as in the text). Removed: "The message was
  cheered on by several community members and organizations such as the
  East Oakland Collective nonprofit." (a reaction). Added: David Newton,
  "In an interview with The Oaklandside last week, Newton was resigned to
  his failure to gather enough signatures–he picked up 37–but said he
  would work with whoever came out ahead...", Source Descriptors
  "nephew", Source Justification "One person who tried but failed to
  gather enough signatures to run for the district seat was David Newton,
  a nephew of Black Panther Party founder Huey Newton." Row 20 ("After
  Love’s lawyers submitted their petition on Aug. 20, ... but lawyers for
  the county asked for the hearing to be moved up to today...") retyped
  Named Organization / Kaufman Legal Group -> Named Organization /
  Alameda County (the attribution is to the county's lawyers, as in "The
  county attorneys countered in a brief..."); Source Descriptors
  "political law firm" -> blank. Now 22 rows.

- **`8-primary-recount-complete.csv`, Phase 4, 2026-11-18.** Quote marks
  restored on John McGuire's and Bob Good's statements; Governor Glenn
  Youngkin's two rows merged into the full paragraph. Title of Source as
  written: McGuire "State Senator, (R-Goochland)" -> "State Senator
  (R-Goochland)"; Good "Representative, (R-Farmville)" -> "Rep.
  (R-Farmville); Chair for the House Freedom Caucus" (the chair moved from
  his Source Justification, now blank). McGuire's Source Justification
  "declared his victory in a recent statement" -> blank (attribution
  only). Added: Document (Source Descriptors "law"), "Under Virginia law,
  localities relative to this election are required to cover the cost of
  a recount when the margin of error is at or below .5 percent." Now 5
  rows.

- **`9-steph-curry-gold.csv`, Phase 4, 2026-11-18.** Stephen Curry's two
  rows from one paragraph merged into the full paragraph ("“Everything
  happens for a reason...,” the 36-year-old Curry said Wednesday... “But
  this summer’s been amazing...”"). Now 9 rows.

- **`10-Nebraska-voting-rights.csv`, Phase 4, 2026-11-18.** Title of
  Source capitals not in the text lowercased: TJ King "Outreach
  specialist, Black and Pink" -> "outreach specialist, Black and Pink";
  Nicole Porter "Senior director of advocacy, The Sentencing Project" ->
  "senior director of advocacy, The Sentencing Project". 8 rows.

- **`11-Nebraska-felons-crim-justice.csv`, Phase 4, 2026-11-18.** Name of
  Source "Brennan Center for Justice." -> "Brennan Center for Justice".
  Title of Source, capitals not in the text lowercased: King "outreach
  specialist, Nebraska AIDS Project", Patrick Berry "counsel, Democracy
  program at the Brennan Center", Porter "director of advocacy, The
  Sentencing Project", Margaret Love "co-founder and director, Collateral
  Consequences Resource Center"; Kathy Wilcot's two versions -> "member of
  the University of Nebraska Board of Regents" (as written). Steve Smith:
  Title "Civic Nebraska" (an organization name, no position) -> blank;
  Source Justification -> "of Civic Nebraska, part of a large coalition
  of groups supporting the measure". Love's row: "At the time Nebraska was
  in step with other states." taken off the front (her attribution is on
  the next sentence). King's Source Justification on two rows "King, 51,
  fought addiction for years and spent five years in prison after being
  convicted; volunteers at various organizations;" -> "King, 51, fought
  addiction for years and spent five years in prison after being
  convicted of possessing the party drug Ecstasy and theft by deception,
  ending probation last August.; volunteers at various organizations".
  Added: Steven Scott, "He sees the two-year wait period as one link in a
  long chain of barriers for those trying to reenter society." (it had
  been inside his Source Justification, which now keeps "Steven Scott,
  33, was paroled in 2015... including 2020."). Now 21 rows.

- **Step 1 of Phase 4 for GT-I: settled mechanical fixes, 2026-11-18 (user
  approved as one table).** Rules applied: one sentence is never cut (a
  Sourced Statement that started or ended mid-sentence becomes the full
  sentence); one row per source per paragraph (merged); a Title capital
  that isn't in the text (or only comes from the start of a sentence)
  lowercased, except titles written before a name; a leading "a/an/the"
  dropped from Titles. Includes two merges missed in the 1-12 pass (1, 3).
  - `1-Harris-Poll.csv`: one source, one paragraph -> rows merged: rows 2+3 (Reuters/Ipsos).
  - `3-SF renewed-sweeps.csv`: one source, one paragraph -> rows merged: rows 1+2 (Antonio Robinson).
  - `6-OUSD-basic-job.csv`: Title of Source as in the text: group row "Teachers" -> "teachers".
  - `13-pilot-curb-evictions.csv`: Sourced Statement made the full sentence from the text (quote marks/period/attribution restored): row 7 (Deborah Henderson). Title of Source as in the text: group row "Judge" -> "judge"; Deborah Henderson "Judge" -> "judge".
  - `14-many-police-calls.csv`: Title of Source as in the text: Jim Baker "Consultant, state public safety enhancement initiative" -> "consultant, state public safety enhancement initiative"; group row "Authorities" -> "authorities"; Libby Bennett "Executive director, Groundworks" -> "executive director, Groundworks"; Daniel Quipp "Chair of the selectboard" -> "chair of the selectboard".
  - `15-Newsom-declares-emergency.csv`: Title of Source as in the text: Jim Hanggi "Spokesperson, Southern California Edison" -> "spokesperson, Southern California Edison".
  - `16-America's-Sleeping-Giant.csv`: Sourced Statement made the full sentence from the text (quote marks/period/attribution restored): row 5 (Poor People’s Campaign). one source, one paragraph -> rows merged: rows 2+3 (Robert Paul Hartley); rows 5+6 (Poor People’s Campaign); rows 7+8 (William Barber).
  - `17-homelessness-santa-clara.csv`: Title of Source as in the text: Annya Artigas "the director of Social Emotional Learning at the Alum Rock district" -> "director of Social Emotional Learning at the Alum Rock district"; Maureen Smith "Professor, San Jose State University" -> "professor, San Jose State University".
  - `18-wyoming-primary-dems.csv`: Sourced Statement made the full sentence from the text (quote marks/period/attribution restored): row 5 (Becky Blackburn); row 6 (Becky Blackburn). Title of Source as in the text: Ross Diercks "Former Democratic state representative" -> "former Democratic state representative".
  - `19-oakland-sues-SW-airlines.csv`: Sourced Statement made the full sentence from the text (quote marks/period/attribution restored): row 9 (Southwest Airlines); row 12 (Measure FF). one source, one paragraph -> rows merged: rows 4+5 (lawsuit). Title of Source as in the text: Eric Swygert "Elected ramp representative with the Transport Workers Union of America Local 555" -> "elected ramp representative with the Transport Workers Union of America Local 555".
  - `20-bill-easier-access.csv`: Title of Source as in the text: group row "Program officials" -> "program officials".
  - `22-Judge-rules-breona-taylor.csv`: Sourced Statement made the full sentence from the text (quote marks/period/attribution restored): row 1 (Charles R. Simpson III). Title of Source as in the text: Christopher 2X "Founder, Game Changers" -> "founder, Game Changers"; Donald Trump "Former President" -> "former President".
  - `23-black-landowners-11mins.csv`: Title of Source as in the text: Jason Shaw "Chairman of the board, Georgia Public Service Commission" -> "chairman of the board, Georgia Public Service Commission"; Benjamin Tarbutton III "Owner, Sandersville Railroad" -> "owner, Sandersville Railroad".
  - `24-what-happens-to-migrants.csv`: Sourced Statement made the full sentence from the text (quote marks/period/attribution restored): row 1 (Spanish-speaking readers). Title of Source as in the text: Donald Trump "Former President" -> "former President".
  - `25-ballot-access-trans.csv`: Title of Source as in the text: Lucas Cameron-Vaughn "Attorney, ACLU of Tennessee" -> "attorney, ACLU of Tennessee"; group row "Politicians" -> "politicians"; group row "Experts" -> "experts".
  - `28-scranton-ukrainian-weapons.csv`: one source, one paragraph -> rows merged: rows 6+7 (Jim Jarick). Title of Source as in the text: group row "Officials" -> "officials"; Bob Durkin "President and chief executive officer of the Scranton Chamber of Commerce" -> "president and chief executive officer of the Scranton Chamber of Commerce"; Danielle Weinschenk "Lead Public Affairs Specialist at the Tobyhanna Army Depot" -> "lead Public Affairs Specialist at the Tobyhanna Army Depot"; Jim Jarick "Senior technology center engineer at the Tobyhanna Army Depot" -> "senior technology center engineer at the Tobyhanna Army Depot"; Mark Cancian "Senior advisor at the Center for Strategic and International Studies" -> "senior advisor at the Center for Strategic and International Studies".
  - `31-SV-GT-vermont_bill.csv`: Sourced Statement made the full sentence from the text (quote marks/period/attribution restored): row 4 (Aaliyah Wilburn).
  - `34-human-ca_guanranteed_income.csv`: Title of Source as in the text: Sean Kline "Director, Stanford Basic Income Lab" -> "director, Stanford Basic Income Lab"; Amy Castro "Director, Penn's Center for Guaranteed Income Research" -> "director, Penn's Center for Guaranteed Income Research"; Brian Arredondo "Coach" -> "coach"; group row "Researchers" -> "researchers"; Ely Fournier "Director, Strength Based Community Change" -> "director, Strength Based Community Change".
  - `35-Trump-Tariffs-Rattling-Meta.csv`: Title of Source as in the text: Eric Seufert "an independent mobile advertising analyst" -> "independent mobile advertising analyst".
  - `36-Whats-Wrong-With-Apple.csv`: Title of Source as in the text: Michael Gartenberg "Technology Analyst" -> "technology analyst".
  - `37-DogWalker.csv`: Title of Source as in the text: group row "Investigators" -> "investigators".
  - `38-UC-enrollment.csv`: Title of Source as in the text: David Alvarez "The chair of the Assembly’s budget subcommittee on education finance & a Democrat from Chula Vista" -> "chair of the Assembly’s budget subcommittee on education finance & a Democrat from Chula Vista".
  - `40-SeattleProtest.csv`: Sourced Statement made the full sentence from the text (quote marks/period/attribution restored): row 6 (Ana Marie Cauce).

- **Step 2 (judgment) batch 1, stories 13-24, 2026-11-18.**
  - `13-pilot-curb-evictions.csv`: removed "But Henderson has been happy
    with the results since then." (a feeling). 13 rows.
  - `14-many-police-calls.csv`: Robin Joy's two Title forms -> "Crime
    Research Group’s statistical director" (as written) on all her rows;
    "State consultants" -> "state consultants" (capital from sentence
    start); the study row's Source Justification "study by the Vermont
    nonprofit Crime Research Group" (a piece of its own row) -> blank; the
    "Those who spoke said..." row's Source Justification "the latest
    selectboard meeting Tuesday night." -> the full sentence "Bennett was
    one of several social service leaders to attend the latest selectboard
    meeting Tuesday night."; Jim Baker's "recounted at a recent public
    meeting." kept (user: speaking at the public meeting gives him standing
    in the story). Added: Named Organization / Groundworks Collaborative,
    "Groundworks Collaborative, for example, helps local people without
    permanent housing (its outreach team reports contact with 159 such
    individuals so far this year)." 24 rows.
  - `15-Newsom-declares-emergency.csv`: removed "Impacted residents, many of
    whom are older... have experienced a range of difficult emotions..."
    (feelings). Gavin Newsom Title "Governor" -> "Gov." (as written).
    Added: officials (Unnamed Group of People, Title "officials"), "The
    moves come as officials raised safety concerns about downed power
    lines, and equipment damaged by the landslide, which they said could
    cause fires."; John Cruikshank, "Cruikshank has called on state and
    federal officials to declare a state of emergency — so far, that step
    has not been taken." 16 rows.
  - `16-America's-Sleeping-Giant.csv`: William Barber Title "Rev.Dr." ->
    "Rev. Dr.". Added: Named Organization / Poor People’s Campaign, "This
    week in Washington at the National Press Club the Poor People’s
    Campaign, under the leadership of Barber and Rev. Dr. Liz Theoharis,
    announced their plans for a mass mobilization..." 11 rows.
  - `17-homelessness-santa-clara.csv`: removed Juanita Campa's "Campa’s next
    goal? To get her son... enrolled in kindergarten." and "Campa also hopes
    to move her family into a more permanent living situation." (a goal, a
    hope). Annya Artigas's two Title forms -> "director of Social Emotional
    Learning at the Alum Rock district" (as written). 18 rows.
  - `18-wyoming-primary-dems.csv`: Chuck Gray Title "Wyoming Secretary of
    State" -> "Wyoming’s Republican secretary of state" (as written); his
    Source Justification (the attribution sentence) -> blank; Pat Jordan's
    first row Source Justification (a piece of its own row) -> blank (his
    second row keeps it). Added: Named Organization / Associated Press,
    "There were none with such a low percentage of Democratic registrations
    in the presidential election years of 2012, 2008 and 2004, according to
    the AP data." 14 rows.
  - `20-bill-easier-access.csv`: Dave Cortese Title "Senator, (D-San Jose)"
    -> "Sen. (D-San Jose)" (as written); Cathy Kennedy "Registered Nurse,
    and President, California Nurses Assocation" -> "Registered Nurse and
    president of the CNA" (as written). 8 rows.
  - `22-Judge-rules-breona-taylor.csv`: removed the "Castle Doctrine" row
    ("— negating Walker’s constitutional rights..." -- the reporter's
    clause, now inside the judge's full-sentence row). Hawk Newsome's two
    Title forms -> "co-founder of Black Lives Matter of Greater New York and
    Black Opportunities" (as written). 15 rows.
  - `23-black-landowners-11mins.csv`: removed "The construction would also
    have an impact on other Black residents in the community, who believe
    it would affect their quality of life." (a group's belief, far from a
    standard attribution form). 15 rows.
  - `24-what-happens-to-migrants.csv`: added Document / Project 2025
    (Source Descriptors "plan"), "Additional anti-asylum measures in
    Project 2025 include increasing the credible fear threshold..." 12
    rows.

- **Step 2 (judgment) batch 2, stories 26-29, 31, 32, 35-43, 2026-11-18.**
  - `26-biden-harris-11M.csv`: added three Named Organization / U.S.
    Department of Education rows: "On Tuesday, the U.S. Department of
    Education (Department) announced over $11 million in grants..." (also
    Miguel Cardona's Source Justification), "The Department projects that
    these grantees will serve approximately 3,700 participants...", "The
    Department also announced $3.7 million in grant funding for the Native
    American Language (NAL@ED) projects." 4 rows.
  - `27-general-assembly-landlords.csv`: added Named Organization / MLK50
    (the outlet's own finding), "But, in a sample of 1,090 recent eviction
    cases, MLK50 found that 145 (about 13%) appeared on the court’s schedule
    for a second time..." 20 rows.
  - `28-scranton-ukrainian-weapons.csv`: added Tobyhanna Army Depot ("...
    estimates that the company has a $2.4 billion dollar impact on the
    area..."), Congressman Matt Cartwright ("...pointed to a $277 million
    Raytheon contract to illustrate this point."), Ukraine ("Ukraine has said
    that it will need more ammunition..."), Biden administration ("The Biden
    administration announced this month it would be sending an additional
    $250 million..."). Olksandr Popov Title "Colonel" -> "Col." (as
    written). 14 rows.
  - `31-SV-GT-vermont_bill.csv`: Saudia LaMont's Source Justifications
    "daughter was reacting to several instances of students touching and
    petting her hair without her consent" (a piece of its own row) and
    "citing surveys" (plain attribution) -> blank. Kept: "said during a
    preliminary vote on the bill" (speaking in a legislative proceeding).
    21 rows.
  - `35-Trump-Tariffs-Rattling-Meta.csv`: removed "The reasons for Meta’s
    slide are less obvious. But close watchers of the social networking and
    metaverse company know it is just as vulnerable..." (generic "close
    watchers" who "know"). 6 rows.
  - `39-84pct-coral-reefs.csv`: added Donald Trump (Title "President"),
    "The group’s update comes as President Donald Trump has moved
    aggressively in his second term to boost fossil fuels and roll back
    clean energy programs, which he says is necessary for economic growth."
    8 rows.
  - `40-SeattleProtest.csv`: Victor Balta Title "UW Spokesperson" -> "UW
    spokesperson"; Name "Mendal Weingarten" -> "Mendel Weingarten" (as in
    the text). Added Named Organization / UW, "Students United for
    Palestinian Equality and Return (SUPER UW) held a rally on campus
    Thursday, the day after UW announced it was suspending 21 students for
    the destructive protest."; Ana Marie Cauce, "University President Ana
    Marie Cauce quickly condemned the protest, which is estimated to have
    caused at least $1 million in damage." (it had been the Source
    Justification of her statement row -- and stays its Source
    Justification: the lead-in to her quote). 11 rows.
  - `41-Christians-K12.csv`: row numbers had a gap (no row 4) ->
    renumbered. Added Document (Source Descriptors "lawsuit"), "In July, a
    Black Los Angeles-area mother filed a lawsuit on behalf of her son,
    D.W., who endured years of racist bullying... including being called a
    “monkey” and “snitch.”" 9 rows.
  - `42-Uber-Lyft-CA.csv`: the two Castellanos rows' Names -> "Castellanos
    v. State of California". 13 rows.
  - `43-Chicago-immigration.csv`: Jennifer Peyton's row trimmed -- "Peyton
    is still unsure why she was terminated." taken off (the row keeps "She
    said she wakes up at night wondering why."). Added Carla Espinoza,
    "Espinoza has asked for a reason for her termination and specifically
    whether her gender, sex or heritage played a role." (it had been the
    Source Justification of her "“I am waiting for a response,”" row,
    which keeps it as its Source Justification -- the lead-in to her
    quote). 16 rows.

- **Step 2 (judgment) batch 3, stories 25 and 30, 2026-11-18.**
  - `25-ballot-access-trans.csv`: Quinn Diaz's row (only the paragraph's
    second sentence) -> the full paragraph ("To Quinn Diaz, a public
    policy associate at advocacy group Equality Florida, the failure of
    most anti-LGBTQ+ bills... But the new gender marker regulation...,
    they argued."); Diaz's two Title forms -> "public policy associate at
    advocacy group Equality Florida" (as written). Removed (feelings):
    Hazel Krebs's "Hazel Krebs... felt the weight of that increased
    scrutiny..." and "Krebs, who plans to vote in November, is worried...
    But she is determined..." -- the latter added to her closing quote's
    Source Justification after "; " (the lead-in to that quote). Rose
    Saxe's Source Justification (another source's statement, "politicians
    supporting these policies frame them as protecting cisgender women.")
    -> blank. Name "Center of Democracy and Civic Engagement" -> "Center
    for Democracy and Civic Engagement" (as in the text). Added: advocates
    (Unnamed Group of People), "The ripple effect could extend beyond
    trans people, these advocates warn."; Named Organization / Department
    of Highway Safety and Motor Vehicles, "In January, a Florida agency
    announced it would no longer update a trans person’s driver’s license
    with their correct gender identity."; advocacy groups and lawmakers
    (Unnamed Group of People), "Some advocacy groups and lawmakers in
    Florida also argue the state’s new regulations conflict with the
    federal Real ID Act..." Lucas Cameron-Vaughn's Source Justification
    "Those that intentionally target government identification that
    people use in everyday life, like driver’s licenses, are particularly
    challenging." -> blank (after a second look at all his rows: the
    sentence is about the laws, not his standing; nothing in the text ties
    him to the ACLU's lawsuit). 38 rows.
  - `30-la-lower-rent-hikes.csv`: Hugo Soto-Martinez Title "City Council
    Member" -> "Councilmember" (as written); his "has already called for
    capping increases at 3%" row's Source Justification blank -> "one of
    the council members who requested the report be commissioned"; his
    email row's Source Justification "In an email reacting to the Economic
    Roundtable report" (plain channel) -> blank. Title "L.A. City Council
    members" -> "L.A. city council members" (as written). Name "Laist" ->
    "LAist". Added (Unnamed Group of People): landlord advocates,
    "Landlord advocates strongly disputed the report’s conclusions.";
    landlords, "The city faced strong criticism from landlords who argued
    too little was being done to help property owners." The report row
    "Over the course of 193 pages, the report offers an extensive
    analysis..." kept (user). 38 rows.

- **`17-homelessness-santa-clara.csv` and its story text, 2026-11-18.**
  The text was hard-wrapped; re-flowed to one paragraph per line from the
  PDF the user supplied (only whitespace changed). Two Sourced Statements
  held the old line breaks (rows 4 "Three other counties in the Bay
  Area..." and 17 "Back on the East Side, Campa said...") -> spaces. All
  18 rows match the text. Step 1 and Step 2 re-run on the re-flowed text:
  no further changes.

- **Step 2 (judgment) batch 4, stories 33 and 34, 2026-11-18 (three SJ
  items in 33 and two in 34 held for the user).**
  - `33-human-border_patrol_towers.csv`: removed a duplicate row (a second
    copy of "Customs and Border Protection says the surveillance equipment
    is necessary..."). "In the San Diego area, border agents have been
    telling migrants... the system will detect anyone who tries to walk
    away, advocates say." retyped from Title "Border agents" to Source
    Descriptors "advocates" (the advocates are the attributed source; the
    agents' telling is nested). Added: state officials (Unnamed Group of
    People, Title "state officials"), the same sentence as the border
    advocates' row ("Border advocates are urging..., but state officials
    say they can’t afford to while facing a budget deficit."). Names "Dept
    of Homeland Security" -> "Department of Homeland Security", "Newsom's
    Office" -> "Newsom’s office". The Border Chronicles podcast row kept as
    is (user). Source Justifications blanked: the federal-contracts row's
    "The spending comes as the Biden Administration is facing increased
    political pressure..." (about the politics, not the source); the
    privacy and human rights advocates' "Some of the towers can track
    movement miles away." (user: inferring that the towers' reach is why
    the advocates are voiced is a leap). "The Department of Homeland
    Security “has also acknowledged the need for better coordination,
    communication, and oversight,” it said in its 2021 report on its border
    technology." retyped Document (Source Descriptors "report") -> Named
    Organization / Department of Homeland Security, Source Justification
    "it said in its 2021 report on its border technology." -> "its 2021
    report on its border technology" (user: "its" makes it the
    organization's own report). 28 rows.
  - `34-human-ca_guanranteed_income.csv`: Ely Fournier Title "director,
    Strength Based Community Change" (not in the text) -> blank; Sean
    Kline's Source Justification "director of the Stanford Basic Income
    Lab. Which studies..." (not verbatim) -> "Stanford Basic Income Lab,
    which studies guaranteed income programs across the country"; "Penn's"
    -> "Penn’s" (Name and Amy Castro's Title). Removed Kipp Kahlia's
    "Kahlia thought if she was going to meet clients..., showing up in an
    old, dinged up car wouldn’t do a lot for her image." (a reported
    thought). The advocates row's Source Justification "In 2021 child
    poverty decreased to 9% from 17.6% in 2019." -> blank. Kept (user):
    the opponents row's Source Justification "some opponents" and Brian
    Arredondo's "a coach for program participants." 26 rows.

### Checklist — Category 2 findings, to resolve during the normal Phase
2.5 (Named Organization) pass on these specific files, not forgotten in
the meantime

While investigating the "Not Defined" cases above, a broadened scan
turned up a different, more serious issue in two other files: named
organizations that are completely missing their own row, not merely
mistyped. Logging these here as a checklist for whenever Phase 2.5 runs
on `175-Indigenous_Health_COVID.csv` and `52-East_Bay_Voters.csv`
specifically, plus exactly how the scan found them so it can be
reproduced fresh if needed rather than re-derived from memory.

**How this was found:** a Python scan across all 90 GT-II CSVs, checking
every row typed Unnamed Group of People or Unnamed Person for the
presence of a broad list of institution-type nouns (hospital, company,
agency, university, council, board, commission, department, etc.) in its
Sourced Statement text, then manually checking each hit's full row and
surrounding sentence to rule out false positives (most hits were fine —
a person or group of people correctly typed as the source, with an
institution word merely mentioned in passing).

- **`175-Indigenous_Health_COVID.csv` rows 20 and 21** share one Sourced
  Statement: "The U.S. Civil Rights Commission, National Indian Health
  Board, Government Accountability Office, congressional committees and
  tribal leaders warned for decades that Native American health care was
  anemic and primed for catastrophe." Only 2 rows exist for this sentence
  (row 20: Title "congressional committees", UGOP; row 21: Title "tribal
  leaders", UGOP) — the three actually-named organizations (U.S. Civil
  Rights Commission, National Indian Health Board, Government
  Accountability Office) have no rows of their own at all. When Phase 2.5
  runs on this file, add 3 new Named Organization rows for this same
  Sourced Statement (per Note 12's joint-attribution splitting), one per
  named entity.
- **`175` row 57**: "The National Indian Health Board and tribal leaders
  contend the constraints are unrealistic..." — only "tribal leaders"
  (UGOP) has a row; National Indian Health Board has none. Same fix:
  add a Named Organization row for the Board.
- **`52-East_Bay_Voters.csv` row 8**: "The Manhattan District Attorney's
  Office has also reportedly opened an investigation into the 2024
  allegation, which the accuser said occurred in New York." The existing
  row (Unnamed Person, SD "accuser") correctly captures the second half
  of this sentence, but the first half — the Manhattan DA's Office
  opening an investigation, a clear Named Organization fact — has no row
  at all. Add one when Phase 2.5 runs on this file.

Also logged the same day, a separate and unrelated finding in
`72-Quantum_Campus.csv` (rows 8, 23, 33, 38): common-noun descriptive
text sitting in Name of Source for Unnamed Group of People/Unnamed Person
rows (e.g., Name = "State regulators", "advocates and residents") —
this is exactly what Phase 2.1/2.2's credentialing-test population passes
already exist to catch, so no separate checklist needed here; it'll be
handled automatically whenever those phases run on this file.

~~**New entry (2026-09-17, found during the Sep17-batch Phase 1 sweep):**
`181-US_Canada_Dairy.csv` is missing a row...~~ **RETRACTED same day —
false positive.** Row 24 already captures this exact statement ("The
senior Trump administration official who briefed reporters..."),
correctly typed Unnamed Person. It wasn't caught by the Phase 1 sweep
because that sweep only scanned Anonymous-Source-typed rows, and this
row is a different type — that's a real scope note for future sweeps
(a check for missing SS candidates isn't the same thing as Phase 1's
Anonymous Source recheck), but there's no actual missing row in this
file. Caught by the user re-checking against the source XLSx.

### Checklist — non-standard Type of Source values found corpus-wide, PRIORITY RAISED 2026-09-18

Found 2026-09-18 while discussing the "Not Defined" fallback's field
handling (128 row 1) — the user asked to verify no other non-standard
Type of Source values exist in the migrated corpus, expecting the
canonicalization pass to have already caught everything. It mostly did
(Sep17 batch: zero non-standard values, clean), but a full corpus-wide
scan of GT-2026 + all of GT-II found 3 more, all in older, pre-Sep17
files that predate this session's canonicalization sweep. **User's
reaction: priority for fixing these just went up** — not clear yet how
they were missed, since a canonicalization sweep was supposedly run as
part of the earlier migration batches.

**How this was found (reproducible):** scan every CSV's Type of Source
column (`benchmarking/GT data/GT-2026/*.csv` + `benchmarking/GT
data/GT-II/*.csv`) with a `collections.Counter`, flag any value outside
the 7 canonical values (6 schema types + "Not Defined").

- **`58-Louisiana_Shooting.csv` rows 1, 3, 5, 6, 11** — `"Unnamed group of
  people"` (lowercase), the same casing-typo class already normalized
  elsewhere this migration. Mechanical fix, not a judgment call.
- **`116-Trump_Christian_Independance_Day.csv` row 6** — the entire row
  is unclassified (Type, Name, Title, SD, and SJ all empty, not just
  Type). SS: "And Interior and other federal agencies have awarded
  no-bid contracts to firms reportedly favored by Trump..." Needs real
  annotation work, not a value fix — "Interior" is a named federal
  agency (Department of the Interior), so this likely wants Named
  Organization for Interior, possibly split per Note 12 with "other
  federal agencies" as an unnamed remainder.
- **`62-D4vd_Murder_Charges.csv` row 4** — Type "Named Group" (not a
  canonical value). Name of Source is "Blair Berk; Marilyn Bednarski;
  Regina Peter" — three named individuals in one row. Looks like a
  straightforward Note 12 violation (one row per named source) rather
  than a real "what type is this" question — probably three separate
  Named Person rows.

## status-2026-09-14

Full cross-reference of everything delivered so far (`~/Documents/GT-II-finishedfiles-XLSx/`), run before the next batch of XLSx files comes down.

**a) All XLSx → CSV conversion: ✅ Complete.**
80 XLSx files in the folder, minus 2 deliberately excluded (133, 134 — confirmed different experiment) = 78. All 78 have a corresponding CSV in `benchmarking/GT data/GT-II/`, 1:1, zero missing, zero count mismatches.

**b) No-URL stories itemized in housekeeping list: ✅ Confirmed.**
Of the 71 distinct story numbers with an XLSx delivered, 19 fall in the no-URL bucket (53, 61, 65, 66, 91, 95, 101–107, 110, 111, 123, 124, 135, 136) — all already itemized in item 2 above. (Story 96 is also on that no-URL list but has no XLSx delivered yet, so it's a non-issue for this batch.)

**c) Dual-coded files converted: ✅ Confirmed.**
9 story numbers have 2 XLSx files each (52, 55, 76, 94, 108, 109, 112, 122, 125) — all 9 have exactly 2 corresponding CSVs. All 9 also have real article text and were fully carried through Phase 1/2 of the migration.

  *Update 2026-10-23:* a **13th dual-coded pair** was found while
  computing talk statistics: `67-California_Ships_Whales.csv` (AV side,
  `#67 - California Ships Whales.xlsx`; listed under the Aashka/AV
  column of the GT Expansion List) and `70-South_Bay_Whales.csv` (SZ,
  `70 - South Bay Whales.xlsx`; Suha/SZ column) are the same KQED
  article ("California Asks Ships to Hit the Brakes for Whales", same
  URL and text). It was delivered under two story numbers, so it was
  never flagged as an overlap. Treat 67/70 as a dual-coded pair from
  now on (for counting, SZ's 70 is used). The full list is now 52, 55,
  61, 67/70, 74, 76, 94, 98, 108, 109, 112, 122, 125.

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
