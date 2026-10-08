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
- **E. Unresolved:** 164 r1 — matches neither our text nor live CNN; no usable archive copy.

### Tier 3 — annotations of visual elements (policy question)

5 r7 (table caption), 65 r15 (chart description), 112-SZ r6 (map graphic),
135 r13 (photo caption), 113 r84/85 (emails quoted from screenshots). Not
present as text, so a text-only LLM can never find them — keep in GT or not?

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

- ~~PDFs now exist in the SZ corpus folder, never converted: 53, 54, 57, 58, 62, 91, 98.~~ **DONE 2026-10-07** — all 7 converted with the upgraded converter, hand-trimmed (page numbers, "Listen" player lines, duplicate URL lines; 91's doubled-label map graphic; 98's ads, nav, product block, and reader comments), Author/Date filled from each byline. Match: 62 20/20, 91 17/17, 54 15/16, 57 14/15, 98 10/11, 58 13/15, 53 7/14 (53's misses are trailing "," vs "." except row 1). New findings: 53 r1 reworded vs our copy ("American warships… acting as a 'net'" vs "Navy ships"; PDF headline also differs from the Expansion List's) -> Tier 2; 58 r5 Sourced Statement contains its first sentence twice (GT paste error) and r7 "The Times" vs text "The New York Times" -> Tier 4.
- 84: **DONE 2026-10-07** — the PDF labeled 84 in SZ's folder is a different article (mislabeled); fetched the correct article directly from the Expansion List URL (site blocks the default script user-agent; a browser user-agent works), extracted with trafilatura, all 7 GT rows exact. Original note: Expansion List batch I has 84 = "Fire crews complete prescribed burns near Paradise Lake for forest management" (SZ, 2news.com URL); batch II's 84 slot is empty. PDF "84" is "Astronomers detect an atmosphere around a mini Pluto"; `84-Paradise_Lake_Fire.csv` is about a fire.
- 127: no PDF anywhere — Expansion List marks it "omitted - see 'GT for Influencer/Videos'" (a video, not an article), yet `GT-II/127-Everest_Climbers.csv` exists. **DONE 2026-10-07** — the real 127 is an influencer video ("Fetterman Secret Israel Handler EXPOSED"), excluded per user. The Everest file numbered 127 turned out to be a mislabeled duplicate of SZ's story-125 annotation (all 6 SS identical), so its CSV was removed from `GT-II/` with nothing lost. Logged in the standing exclusion list in `gt-ii-student-housekeeping.md`.
