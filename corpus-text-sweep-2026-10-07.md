# Corpus text sweep — 2026-10-07 (item 50 retroactive audit)

Read-only sweep of all article texts in `extracted_articles_boilerplate/`
(160 files) against all GT CSVs (`GT-2026/` 43 + `GT-II/` 137 = 180), run
after the upgraded `storypdf_to_text.py` checks (see `development-of-v59.md`
item 50). Nothing was changed by the sweep itself; fixes are tracked in the
Status column as they land.

## Method

1. **Annotation-vs-text match (all 180 CSVs).** Every Sourced Statement
   normalized (quotes/dashes/whitespace/zero-width chars/brackets) and
   checked for an exact substring match in its story's text; non-matches
   scored with `rapidfuzz.partial_ratio`. Works regardless of how the text
   was obtained (only 40 of 160 texts came from a PDF via the converter).
   Result: 170 CSVs had text, 3,222 rows, 140 non-exact (~96% exact).
   Rows < 90 were inspected by hand against the closest text passage.
2. **Upgraded PDF check on all 54 source PDFs** (`dedupe_chars()` duplicate
   count + `find_overlapping_glyphs()`), across the SZ, AV, and
   solidarity-initiative folders. 34 flagged; each flagged line was then
   traced into the corpus text to see whether merged text actually landed
   there. Only 191 and 210 (plus the already-fixed 195/197/200/205) did.
3. **Text-side scan of all 160 texts**: garble signs (doubled-letter
   words, mid-word capitals, runs of 1–3-letter lines), site clutter in
   the last ~1,200 chars of the body, and inline link/URL/page-footer
   patterns anywhere.

## Findings

### Tier 1 — corpus text broken (LLMs read garbage here)

| Story | Problem | Fix | Status |
|---|---|---|---|
| 173 | Article ends ~line 35; then ~2,440 lines (~95KB) of the page's `window.__INITIAL_STATE__` JSON | Cut everything after the article | **DONE** — 98.9KB -> 5.3KB, 24/24 rows exact |
| 72 | Inline link URLs (`residents (https://…) packed`); reversed-text print headers mid-sentence at page breaks (`ihc//:sptth( ETANOD )… 9/17/26, 12:43 PM …`); 9 rows < 90 | Re-extract or hand clean | **DONE** — 5 page-break blocks + 14 inline URLs removed, orphaned punctuation rejoined; 25 -> 32/38 exact (remaining 6 are annotation-side: quote style/case, AV elisions in r14/r26) |
| 210 | Guardian "Your offer is expiring / Support us / Maybe later" pop-up x6, some mid-sentence (breaks rows 1, 2, 9); one doubled video caption | Remove pop-up lines | **DONE** — 18 pop-up lines + doubled caption + video-embed labels removed; a 2nd garbled line found (`The paidY aoduverr…`) and recovered by font separation; 18/18 exact |
| 191 | Line 26 garbled: `repSokript etod ,c ownhteincth` ("reported, which" + "Skip to content") | Font-separation recovery from PDF | **DONE** — recovered "reported, which advocated…"; row 2 now exact |
| 36 | NYT page footer mid-sentence (`…by the https://www.nytimes.com/…html 1/6 political…`) | Remove | **DONE** — 5 page footers removed; 16/16 exact |
| 76, 77, 78 | Inline "(opens in new tab)" link labels | Remove | **DONE** — 17 labels removed; 76 10 -> 13/13, 78 15 -> 18/19 (r9 punctuation only). 77 r7 drops because the GT row itself copied the label — see Tier 4 |
| 192 | Doubled video caption line (`OOppeennAAII…`) + player timestamp | Remove | **DONE** — 6 player-clutter lines removed; 20/20 exact |

### Tier 2 — annotated content missing/different in our text (likely article-version drift, as with 212/193)

| Story | Rows | Note |
|---|---|---|
| 113 | 42, 43 | Sentences absent |
| 124 | 10 (+3) | "A draft DHS memo…" absent; row 3 "Congo" vs text "Democratic Republic of Congo" |
| 135 | 5, 12, 21 | Different wording throughout ("a person" vs "the rescued miner") |
| 164 | 1 | "said Tuesday" + different sentence ending |
| 55 | 1, 20 | Sentences absent |
| 122-SZ | 21 | Sentence absent |
| 30, 33, 150, 184 | 1 each | Reworded — could be annotator paraphrase instead |

Needs a live-page comparison or annotator confirmation of which version they used.

**Investigation 2026-10-08** (live pages fetched for every story with a URL; Internet Archive for 135):
- **A. Our text stale — DONE:** 184 replaced with the live version (9 -> 12/12 exact).
- **B. Our text missing page content.** Decided: photo captions count as article text (they sometimes carry attribution). **Captions DONE:** all real captions added in place for 55 (7), 122 (1), 135 (1) — 55 r1/r20, 122-SZ r21, 135 r13 now exact. **Scrollytelling DONE:** 113's 3 panels inserted in place (r42/r43 now exact). **124 r10 resolved by a new rule** (item 51 in `development-of-v59.md`): subtitles/summary lines aren't annotatable, so the row was removed; subtitles moved out of the body corpus-wide.
- **C. DONE 2026-10-08 — Annotator mixed two versions:** 135 r5/r12/r21 match the Internet Archive's earliest May 29 snapshot exactly; the other 16 rows match the later version we have. Rewritten to the later version's sentences (all 3 now exact).
- **D. DONE 2026-10-08 — Annotator rewording (live = our text):** 124 r3, 33 r20, 150 r3 restored to the verbatim article sentence; 30 r10 removed instead (reworded duplicate of r11 pointing to a chart). See `development-of-v59.md` item 52 and SZ's housekeeping log.
- **E. DONE 2026-10-08:** 164 r1 — matched neither our text nor live CNN; no usable archive copy, and the annotator couldn't locate the version used. Replaced with our copy's opening sentence word for word (logged under SZ).

### Tier 3 — annotations of visual elements (policy question)

5 r7 (table caption), 65 r15 (chart description), 112-SZ r6 (map graphic),
135 r13 (photo caption), 113 r84/85 (emails quoted from screenshots). Not
present as text, so a text-only LLM can never find them — keep in GT or not?

**Tier 3 status (2026-10-08) — OPEN, graphics rule undecided.** Traced so far: 65 r15 = Datawrapper chart description (embed iA... `iiZQd`); 112-SZ r6 = CNN flight-map graphic intro (credits Spanish Health Ministry, Oceanwide Expeditions; fact not in body; also 2 orgs in one row, Note 12); 5 r7 = table title (no URL to trace); 113 r84/r85 = emails in screenshots, pure images (no alt text). Graphic labels reaching model input: 68 (3 NOAA/AirNow chart source lines), 205 ("Chart: CBS News Data Team • Source: Freddie Mac via FRED"). Body sentences that describe a graphic (12 r2, 164 r9) are body text, fine. Leading proposal (not yet approved): text benchmark covers only the text payload; move the 5 graphic rows to a side file (`GT-II/graphics-only-rows.csv`) rather than delete; move 68/205 chart labels to a `Graphic:` header field. **Before deciding:** the user asked for a verified check (the scans above were keyword-based): for the ~108 URL stories, enumerate every embedded graphic/figure on the live page, extract its text, and test (1) whether it's in our article files / model input and (2) whether any GT row annotates it; for the ~50 PDF-only stories, list chart-like text blocks for visual review.

**Tier 3 verified graphics check — DONE 2026-10-08.** Live pages fetched for all 108 URL stories (80 direct + 11 via Internet Archive = 91 checkable; NYT/Reuters/WaPo blocks covered by their PDFs instead). Every embedded graphic inventoried (Datawrapper, Flourish/Infogram, newsroom graphics frames, inline chart/map figures) in 11 stories (65, 75, 76, 77, 78, 105, 112, 113, 168, 170, 175); each embed fetched and its text compared against the model input and every GT row. PDF-derived files (47) scanned for chart-like blocks (source/note footers, label runs, numeric axis runs). **Results:** (1) graphic text reaching the models: only **68** (Mapbox map controls/credit, an air-quality table, 3 NOAA/AirNow source lines) and **205** (mortgage-rate chart axis values + "Chart: CBS News Data Team • Source: Freddie Mac via FRED") — neither annotated; 113's scrollytelling panels are in deliberately. (2) GT rows annotating graphic text: **65 r15** and **112-SZ r6** (both verified against the embeds), **5 r7** (self-labeled "TABLE:", untraceable — no URL/PDF), plus **113 r84/r85** (emails as screenshots). Apparent matches in 65 r7, 168 r3, 170 r20/r28 were false positives (shared org names; an embed repeating the article's own lead as a teaser). (3) Not checkable: the 34 legacy GT-2026 files (no URL/PDF; keyword scan only, which found just 5 r7) and 104, 152 (no page, no PDF). Other PDF-side hits were site navigation/ads (Tier 5). **DONE 2026-10-08 (approved):** 5 GT rows moved to `GT-II/graphics-only-rows.csv`; 68/205 graphic lines moved to a `Graphic:` header field; rule recorded (`development-of-v59.md` item 53).

## Tiers 4–6 refreshed (2026-10-08) — DETAILED FINDINGS + SUGGESTED ORDER (nothing applied yet)

Re-run on current files and model input (`article_body()`), after Tiers 1/2/3/7 and the body-only rule. Original Tier 4–6 notes below are superseded by this section.

**Tier 4 — annotations not matching the article exactly: 82 of 3,328 rows.**
- ~55 trivial (trailing "." where the article continues with ","; curly vs straight quotes; a missing space; "WHO and CDC" vs "WHO and the CDC"). Leave — fuzzy evaluation absorbs these.
- **Clear errors (7 rows) — DONE 2026-10-08 (`development-of-v59.md` item 54):** 58 r5 (first sentence pasted twice); 159 r14, r15 (mojibake "Iranâ€™s" -> "Iran's"); 121 r5 ("Al tools" with lowercase L -> "AI tools"); 101 r5 (annotator inserted "DoDMA" before "Spokesperson Chipiliro Khamula says…" — inference, not in the sentence); 22 r7, GT-2026 ("212 Black lives" vs article "214"); 112-SZ r1 ("he told CNN's Erin Burnett" vs article "he said on CNN's 'Erin Burnett OutFront'").
- **Pattern needing a rule (~11 rows) — attribution stitching:** where a multi-sentence quote is interrupted by its attribution, the annotator cut the attribution and stitched the quote together, or skipped a sentence: 19 r9 (drops `the spokesperson said in a statement`), 7 r5/r12/r20 (`the lawyers stated in their brief`, `he said`, `she wrote`), 13 r7 (`Henderson told a WLOK-AM audience earlier this summer`), 31 r10 (`, she said,`), 6 r15 and 63 r1 (skip a whole sentence between two passages), 72 r14/r26 (AV, skip/add words). Question for the user: must a Sourced Statement be one contiguous verbatim span? Recommendation: yes — restore to include the attribution (same as Tier 2 D fixes). Mostly GT-2026 rows.
- Small trims (leave or decide case by case): 100 r9 drops "(RDDT.N)"; 124 r1 drops "(WHO)"; 58 r7 "The Times" vs "The New York Times".
- Already decided/pending: 193 r23 (awaiting SZ fresh Newsweek capture); 191 r10–12 "Nividia" (leave); 194 r9 ş vs ș; 68 r5 (article misspells "Synder", annotator corrected).

**Tier 5 — web clutter still in model input: 45 files flagged** (some false positives, e.g. "sign up for an easy time" in a quote). Heavy (10–16 lines): 200, 203, 206, 208, 209 (Reuters site footer "Latest / Browse / Home / World…"), 205 (CBS footer), 191 (Independent footer), 204 (CNN footer). Light (1–7 lines): "Advertisement" (56, 121); "Getting your Trinity Audio player ready…" (109, 122); "Read More:" / "RELATED ARTICLES" (17, 23, 55, 98); newsletter pitches (16, 29, 36, 66, 74, 100, 108, 111, 168, 212); photo-gallery chrome "Read More" / "Purchase Licensing Rights" (123, 151, 153, 181); "CLICK HERE TO DOWNLOAD THE FOX NEWS APP" (159); WSJ reprint notice (196). Page numbers mid-sentence: 50 ("1/5", "3/5", one inside annotated row 7). No annotation affected; every model reads it. Plan: hand-trim with the user file by file, heavy first.

**Tier 6 — unusual characters.** Model input nearly clean: 21 modifier apostrophes "ʼ" in 7 files (35, 36, 58, 68, 82, 106, 129); 13 non-breaking hyphens (41, 55); 1 non-breaking space (32) — mechanical normalize to ' and -. In GT CSVs: 159 mojibake (in Tier 4 list); 13 zero-width characters (123, 150, 156); 5 "ﬁ" ligatures (123); non-breaking hyphens (55-Pentagon_Lab, 41, GT-2026-document-rows.csv); 1 "ʼ" (102) — same class as the SZ-batch Pass 0 strip.

**Suggested order (agreed to revisit):** (1) ~~Tier 4 clear errors, 7 rows~~ DONE; (2) ~~attribution stitching~~ rule = verbatim/contiguous (item 55); Groups A/B/C DONE (6 r15 no change; 63 r1, 72 r26 split at paragraph breaks); (3) ~~Tier 6 mechanical cleanup~~ DONE (item 56); (4) Tier 5 trims file by file, heavy first — Reuters five (200, 203, 206, 208, 209) DONE 2026-10-08: cut everything after "Our Standards: The Thomson Reuters Trust Principles." (55/76/96/89/84 lines: site directory, suggested topics, author bios, Read Next teasers), removed top "Learn more about"/"Subscribe" and a mid-body "Subscribe" (200), removed newsletter promos (incl. 2-line Daily Docket promos in 203/208/209), stripped "Purchase Licensing Rights"/"Read more" from photo captions (captions kept). 423 lines removed; all GT rows in the five stories still exact. Remaining: 205, 191, 204, then the light files.

### Tier 4 — minor annotator edits (text fine)

13 r7 (attribution elided mid-quote), 100 r9 ("(RDDT.N)" dropped), 191
r10–12 ("Nividia" -> "Nvidia", already decided: leave), 6 r15 and 63 r1
(not yet examined), 72 r14/r26 (AV skipped/added a few words).

**GT-side clutter — DONE 2026-10-07:** `77` row 7 Sourced Statement and Source Justification, and row 8 Source Justification, contain a copied link label: "legislation(opens in new tab) inside City Hall".

### Tier 5 — site clutter at story endings (27 of 160 files)

Mostly untrimmed SZ-batch Reuters/CBS/AP files: 191, 200 (still has
"Latest / Browse / Advertise" footer), 203, 204, 205, 206, 208, 209, 211.
Older files with leftovers: 23, 29, 32, 35, 36, 55, 56, 66, 74, 110, 121,
129, 153, 159, 176.

### Tier 6 — unusual punctuation characters (harmless, optional)

Modifier-letter apostrophe (U+02BC), non-breaking hyphen (U+2011), "ﬁ"
ligature in 106, 55, 123, 94 — account for most 90–99 near-misses.

### Tier 7 — CSVs with no article text (10 CSVs, 9 stories)

- ~~PDFs now exist in the SZ corpus folder, never converted: 53, 54, 57, 58, 62, 91, 98.~~ **DONE 2026-10-07** — all 7 converted with the upgraded converter, hand-trimmed (page numbers, "Listen" player lines, duplicate URL lines; 91's doubled-label map graphic; 98's ads, nav, product block, and reader comments), Author/Date filled from each byline. Match: 62 20/20, 91 17/17, 54 15/16, 57 14/15, 98 10/11, 58 13/15, 53 7/14 (53's misses are trailing "," vs "." except row 1). New findings: 53 r1 reworded vs our copy (RESOLVED 2026-10-08: SZ supplied the annotated version; r1 is its subtitle) ("American warships… acting as a 'net'" vs "Navy ships"; PDF headline also differs from the Expansion List's) -> Tier 2; 58 r5 Sourced Statement contains its first sentence twice (GT paste error) and r7 "The Times" vs text "The New York Times" -> Tier 4.
- 84: **DONE 2026-10-07** — the PDF labeled 84 in SZ's folder is a different article (mislabeled); fetched the correct article directly from the Expansion List URL (site blocks the default script user-agent; a browser user-agent works), extracted with trafilatura, all 7 GT rows exact. Original note: Expansion List batch I has 84 = "Fire crews complete prescribed burns near Paradise Lake for forest management" (SZ, 2news.com URL); batch II's 84 slot is empty. PDF "84" is "Astronomers detect an atmosphere around a mini Pluto"; `84-Paradise_Lake_Fire.csv` is about a fire.
- 127: no PDF anywhere — Expansion List marks it "omitted - see 'GT for Influencer/Videos'" (a video, not an article), yet `GT-II/127-Everest_Climbers.csv` exists. **DONE 2026-10-07** — the real 127 is an influencer video ("Fetterman Secret Israel Handler EXPOSED"), excluded per user. The Everest file numbered 127 turned out to be a mislabeled duplicate of SZ's story-125 annotation (all 6 SS identical), so its CSV was removed from `GT-II/` with nothing lost. Logged in the standing exclusion list in `gt-ii-student-housekeeping.md`.
