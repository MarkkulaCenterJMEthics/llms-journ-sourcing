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
  specific rows called out in that commit (168 row 28, 172 row 17, 171
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
