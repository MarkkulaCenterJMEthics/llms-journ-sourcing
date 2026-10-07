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
| 173 | Article ends ~line 35; then ~2,440 lines (~95KB) of the page's `window.__INITIAL_STATE__` JSON | Cut everything after the article | open |
| 72 | Inline link URLs (`residents (https://…) packed`); reversed-text print headers mid-sentence at page breaks (`ihc//:sptth( ETANOD )… 9/17/26, 12:43 PM …`); 9 rows < 90 | Re-extract or hand clean | open |
| 210 | Guardian "Your offer is expiring / Support us / Maybe later" pop-up x6, some mid-sentence (breaks rows 1, 2, 9); one doubled video caption | Remove pop-up lines | open |
| 191 | Line 26 garbled: `repSokript etod ,c ownhteincth` ("reported, which" + "Skip to content") | Font-separation recovery from PDF | open |
| 36 | NYT page footer mid-sentence (`…by the https://www.nytimes.com/…html 1/6 political…`) | Remove | open |
| 76, 77, 78 | Inline "(opens in new tab)" link labels | Remove | open |
| 192 | Doubled video caption line (`OOppeennAAII…`) + player timestamp | Remove | open |

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

### Tier 3 — annotations of visual elements (policy question)

5 r7 (table caption), 65 r15 (chart description), 112-SZ r6 (map graphic),
135 r13 (photo caption), 113 r84/85 (emails quoted from screenshots). Not
present as text, so a text-only LLM can never find them — keep in GT or not?

### Tier 4 — minor annotator edits (text fine)

13 r7 (attribution elided mid-quote), 100 r9 ("(RDDT.N)" dropped), 191
r10–12 ("Nividia" -> "Nvidia", already decided: leave), 6 r15 and 63 r1
(not yet examined).

### Tier 5 — site clutter at story endings (27 of 160 files)

Mostly untrimmed SZ-batch Reuters/CBS/AP files: 191, 200 (still has
"Latest / Browse / Advertise" footer), 203, 204, 205, 206, 208, 209, 211.
Older files with leftovers: 23, 29, 32, 35, 36, 55, 56, 66, 74, 110, 121,
129, 153, 159, 176.

### Tier 6 — unusual punctuation characters (harmless, optional)

Modifier-letter apostrophe (U+02BC), non-breaking hyphen (U+2011), "ﬁ"
ligature in 106, 55, 123, 94 — account for most 90–99 near-misses.

### Tier 7 — CSVs with no article text (10 CSVs, 9 stories)

- PDFs now exist in the SZ corpus folder, never converted: 53, 54, 57, 58, 62, 91, 98.
- 84: numbering clash — PDF "84" is "Astronomers detect an atmosphere around a mini Pluto"; `84-Paradise_Lake_Fire.csv` is about a fire.
- 127: no PDF anywhere.
