# v59 Worked Examples — Raw Material for Few-Shot Prompt Examples

This file collects the worked examples cited while drafting v59 prompt revisions
(see `development-of-v59.md`'s Prompt Updates checklist). Purpose: a future pass
will convert these into correct-annotation / incorrect-annotation JSON pairs for
the OpenRouter API completions payload (exact JSON format TBD — user will
specify it later). Each entry below is written so that pass can construct,
without re-deriving anything from scratch:

1. The full correct annotation tuple — all six v59 CSV fields (Sourced
   Statements, Type of Source, Name of Source, Title of Source, Source
   Descriptors, Source Justification).
2. The full incorrect companion tuple (same six fields, with the wrong
   value(s) that were actually found in GT before the fix, where real).
3. Any story-text material beyond the Sourced Statement/Source Justification
   themselves that's needed to derive the correct annotation — quoted
   verbatim, with a source file + line reference, not paraphrased.

Only real GT cases (marked **REAL**) should end up in the final few-shot set
without further scrutiny. Cases marked **HYPOTHETICAL** are illustrative only —
no real GT row backs them; swap in a real case if one turns up later, or keep
them clearly marked as invented if used for illustration.

---

## Example 1 — Title of Source carry-forward (illustrates item 11)

**REAL.** `GT-2026/25-ballot-access-trans.csv`, row 6. Source: Donald Trump.

Full correct tuple:
| Field | Value |
|---|---|
| Sourced Statements | "He described Democratic vice presidential nominee Tim Walz recently as "very heavy into the transgender world."" |
| Type of Source | Named Person |
| Name of Source | Donald Trump |
| Title of Source | former President |
| Source Descriptors | (null) |
| Source Justification | "Anti-transgender rhetoric was front and center at the Republican National Convention in July, and Trump has taken to verbally targeting transgender people in his campaign." |

Full incorrect tuple as originally found in GT (before this migration's fix):
| Field | Value |
|---|---|
| Sourced Statements | (same as above, unchanged) |
| Type of Source | Named Person |
| Name of Source | **former President** *(a title sitting where a name belongs)* |
| Title of Source | **(null)** *(never captured)* |
| Source Descriptors | (null) |
| Source Justification | (same as above, unchanged) |

Supporting story-text material needed to derive the correct annotation (not
in this row's own Sourced Statement or Source Justification — found several
paragraphs earlier in the same article):
> "...served as breeding ground for the binary vision of the country embraced
> by **former President Donald Trump** and U.S. Sen. JD Vance, his running mate
> on the Republican ticket."
— `extracted_articles_boilerplate/25-antitrans_laws.txt`, line 16.

Illustrates: (a) a title must not be captured as if it were the source's name;
(b) Title of Source is a global, article-wide value for the source — it
should carry forward to a later sourced statement even when that specific
sentence only uses a pronoun ("He") and the name/title were established
earlier in the piece.

---

## Example 2 — Title of Source: no contextual inference (illustrates item 16, violation half)

**REAL.** `GT-2026/43-Chicago-immigration.csv`, row 11 (one of 5 rows affected;
all 5 had the identical error). Source: Carla Espinoza.

Full correct tuple:
| Field | Value |
|---|---|
| Sourced Statements | "Espinoza, who was sworn in as a judge in 2023, said at least 20 other judges who stood alongside her during that ceremony have been terminated." |
| Type of Source | Named Person |
| Name of Source | Carla Espinoza |
| Title of Source | judge |
| Source Descriptors | (null) |
| Source Justification | "Espinoza was fired on July 11 by email while sitting on the bench, with no reason given." |

Full incorrect tuple as originally found in GT (before this migration's fix):
| Field | Value |
|---|---|
| Sourced Statements | (same as above, unchanged) |
| Type of Source | Named Person |
| Name of Source | Carla Espinoza |
| Title of Source | **immigration judge** *(unsupported inference — see below)* |
| Source Descriptors | (null) |
| Source Justification | (same as above, unchanged) |

Supporting story-text material showing why "immigration judge" is NOT
supported for Espinoza specifically, even though it's the story's general
subject and is explicitly true of a different named source in the same piece:
> "Espinoza, who was sworn in as a **judge** in 2023..." — this is the entirety
> of what the article ever states about her title; the word "immigration" is
> never attached to her specifically anywhere in the piece.
— `extracted_articles_boilerplate/43-chicago_immigration_judges_fired_by_trump_white_ho.txt`, line 16.

> By contrast, a *different* named source in the same story is given the
> fuller title explicitly: "**Jennifer Peyton, a former assistant chief
> immigration judge** in Chicago, said she has watched more than 100 judicial
> colleagues be abruptly terminated..."
— same file, line 8.

Illustrates: never borrow a qualifier from the story's general topic or a
different source's separately-stated title, however plausible it seems. This
exact mistake was made twice in a row during this migration: once by the
original human GT annotator, and once more by an LLM (this session) that
trusted the existing GT value via carry-forward instead of checking it
against Espinoza's own literal introduction — caught only via a user
follow-up question.

---

## Example 3 — Title of Source: coreference resolution IS allowed (illustrates item 16, carve-out half)

**REAL.** `GT-2026/32-openai_board.csv`, row 17. Source: Ilya Sutskever.

Full correct tuple:
| Field | Value |
|---|---|
| Sourced Statements | "At 5:15 a.m. on Monday, he posted on X, formerly Twitter, that "I deeply regret my participation in the board's actions."" |
| Type of Source | Named Person |
| Name of Source | Ilya Sutskever |
| Title of Source | board member |
| Source Descriptors | (null) |
| Source Justification | (null) |

Full "would-be-incorrect-if-over-restricted" tuple (a plausible but wrong
reading of the item-16 guardrail, if it were applied without the carve-out —
not an actual state GT was ever in for this row, included for contrast):
| Field | Value |
|---|---|
| Sourced Statements | (same as above, unchanged) |
| Type of Source | Named Person |
| Name of Source | Ilya Sutskever |
| Title of Source | **(null)** *(wrongly left empty on the theory that bare "member" isn't credentialing and nothing more specific is stated in this row's own text)* |
| Source Descriptors | **member** *(wrongly routed here instead)* |
| Source Justification | (null) |

Supporting story-text material needed to derive the correct annotation (not
in this row's own Sourced Statement — found in an earlier paragraph, one
sentence before this source is first named):
> "At one point, Mr. Altman, the chief executive, made a move to push out
> **one of the board's members** because he thought a research paper she had
> co-written was critical of the company."
— `2025_input_stories/32-openai_board.txt`, line 12.
> [Next sentence, same article:] "Another **member**, Ilya Sutskever, thought
> Mr. Altman was not always being honest when talking with the board."

Illustrates: resolving "another member" back to the antecedent named one
sentence earlier ("the board's members") is ordinary coreference resolution,
not inference from general context — this is different in kind from Example 2
(Espinoza), where no antecedent anywhere points specifically at her. The
guardrail in Example 2 bans borrowing from general topic/other sources'
titles; it does not ban resolving a plain "another one"/"she"/"the other
member" to something the immediately surrounding text already names for that
specific source.

---

## Example 4 — Source Descriptors carry-forward, accumulate-by-default (illustrates item 15)

**HYPOTHETICAL — no real GT case of this specific gap has turned up in
GT-2026 yet.** Swap in a real case if one surfaces during the 50-150 batch
migration.

Scenario: a source is called "protestor" in an early sourced statement, and
"artist" in a later one, in the same story.

Full correct tuple (later row):
| Field | Value |
|---|---|
| Sourced Statements | *(invented)* "Speaking after the demonstration, [Name] said she planned to turn the day's events into a new series of paintings." |
| Type of Source | Named Person |
| Name of Source | *(invented)* [Name] |
| Title of Source | (null) |
| Source Descriptors | **protestor, artist** *(accumulated — see below)* |
| Source Justification | *(invented)* "[Name], a protestor at Tuesday's rally, said..." *(from the earlier row, carried as context)* |

Full incorrect tuple (later row, if replacement were treated as the default
instead of accumulation):
| Field | Value |
|---|---|
| Sourced Statements | (same as above) |
| Type of Source | Named Person |
| Name of Source | (same as above) |
| Title of Source | (null) |
| Source Descriptors | **artist** *(wrongly overwrites "protestor" instead of adding to it)* |
| Source Justification | (same as above) |

Illustrates: a reporter revealing a second facet of the same source later in
a story does not usually mean the earlier facet stopped being true —
accumulation should be the default ("protestor, artist"), with full
replacement reserved for the narrower case where the source's actual status
demonstrably changed between mentions (e.g., "candidate" becoming
"councilmember" after an election result reported later in the same piece).

---

## Example 5 — Multi-descriptor accumulation within a single row (related precedent for item 15's underlying principle; itself illustrates item 10)

**REAL.** `GT-2026/2-Best-DDR-player.csv`, row 5. Source: Roger Clark.

Full correct tuple:
| Field | Value |
|---|---|
| Sourced Statements | ""He's the GOAT," according to friend Roger Clark, a San Francisco programmer and longtime DDR enthusiast. "Everybody knows it. No one would deny it."" |
| Type of Source | Named Person |
| Name of Source | Roger Clark |
| Title of Source | (null) |
| Source Descriptors | friend, programmer, DDR enthusiast |
| Source Justification | "longtime DDR enthusiast. San Francisco programmer" |

(No incorrect companion needed here — this is a positive precedent example
showing three descriptors captured together in one row, in order of
appearance, none dropped in favor of the others. Included because it's the
closest real precedent to Example 4's cross-row accumulation principle, just
scoped to a single row instead.)

---

## Example 6 — Non-credentialed source's org affiliation goes to Source Justification, not Source Descriptors (illustrates item 17)

**REAL.** `GT-2026/16-America's-Sleeping-Giant.csv`, row 13. Source: Janice Guzman.

Full correct tuple:
| Field | Value |
|---|---|
| Sourced Statements | ""I work for MassHealth as a Personal Care Attendant, helping to take care of people with disabilities and who are sick and need help with their daily activities. But I do not have health insurance myself," Guzman told reporters. "I am an essential worker living paycheck-to-paycheck and I have to make decisions every day. Do I put gas in my car or do I pay my bills? Or do I put food on my table?," said Janice Guzman, an organizer with the. "This is why I am organizing with the Massachusetts Poor People's Campaign. We have got to get our power as people, get back the mic, raise our voices and register voters. Forward together!"" *(note: "an organizer with the." is a genuine glitch in the original article's own text, not a GT/extraction error)* |
| Type of Source | Named Person |
| Name of Source | Janice Guzman |
| Title of Source | (null) |
| Source Descriptors | organizer |
| Source Justification | "an organizer with the Massachusetts Poor People's Campaign and SEIU 1199" |

Full incorrect tuple as originally found in GT (before this migration's fix):
| Field | Value |
|---|---|
| Sourced Statements | (same as above, unchanged) |
| Type of Source | Named Person |
| Name of Source | Janice Guzman |
| Title of Source | **Organizer, Massachusetts Poor People's Campaign** *(non-credentialing word "organizer" wrongly routed here, with a proper-noun org name comma-attached)* |
| Source Descriptors | (null) |
| Source Justification | **(null)** *(the org affiliation and a second org, SEIU 1199, were never captured anywhere)* |

Supporting story-text material needed to derive the correct annotation (not
in this row's own Sourced Statement — found one paragraph earlier, in Guzman's
clean, ungarbled introduction sentence):
> "Dozens of local leaders from the state chapters of the Poor People's
> Campaign were on hand in person and virtually for the high spirited kick off
> including **Janice Guzman, an organizer with the Massachusetts Poor People's
> Campaign and SEIU 1199**."
— `2025_input_stories/16-america_sleeping_giant.txt`, line 28 (exact line
number as of the 2026-09 audit; re-verify if the file changes).

Illustrates: Source Descriptors holds only the bare non-credentialing word
("organizer"); the organizational affiliation — including a second org
(SEIU 1199) that had never been captured anywhere before this fix — belongs
in Source Justification instead. Note the national parent org ("Poor
People's Campaign") already existed elsewhere in this same GT file as its own
Named Organization row, but the *specific state chapter* and the *second org*
did not exist anywhere in the data until this fix recovered them into SJ.

---

## Example 7 — Real professional role that is nonetheless non-credentialing (illustrates item 18)

**REAL.** `GT-2026/18-wyoming-primary-dems.csv`, row 3. Source: Becky Blackburn.

Full correct tuple:
| Field | Value |
|---|---|
| Sourced Statements | ""Normally I just roll my eyes and walk away because I'm fighting a losing battle and I'm fully aware of that," she said. "Maybe that is why I'm well-liked, because I keep my mouth shut 10 times more than I want to."" |
| Type of Source | Named Person |
| Name of Source | Becky Blackburn |
| Title of Source | (null) |
| Source Descriptors | paralegal |
| Source Justification | "A paralegal for the Republican county attorney, Blackburn hears a lot of right-wing views around town." |

Full incorrect tuple as initially proposed during this migration pass (not
what original GT had — GT had left both Title of Source and Source
Descriptors null here; this is the *tempting wrong fix* that was considered
and rejected before landing on the correct one above):
| Field | Value |
|---|---|
| Sourced Statements | (same as above, unchanged) |
| Type of Source | Named Person |
| Name of Source | Becky Blackburn |
| Title of Source | **Paralegal for the Republican county attorney** *(wrongly treated as credentialing — a paralegal doesn't hold the licensed authority the attorney they work for has)* |
| Source Descriptors | (null) |
| Source Justification | (same as above, unchanged) |

Supporting story-text material: none needed beyond this row's own Source
Justification — the reasoning here is about correctly classifying the
*meaning* of "paralegal" (real, defined job; not independently licensed),
not about recovering missing context from elsewhere in the article.

Illustrates: a word can name a real, defined professional position and still
be non-credentialing if the role doesn't carry independent licensed
authority — paralegals assist attorneys but cannot independently represent
clients in court or give formal legal advice. Same reasoning extends to
other legal-adjacent support roles (legal secretary, law clerk) as a category
distinct from the licensed practitioners they support.

---

## Example 8 — UGOP Source Descriptors as an institutional/convened-body label, not just an identity word (illustrates item 13)

**REAL.** `GT-2026/8-primary-recount-complete.csv`, row 1. Type: Unnamed Group of People (the recount panel's judges — originally miscategorized as Named Organization with "three-panel recount court" as Name of Source; the reclassification itself is a separate fix, this example is about the Source Descriptors value once correctly typed as UGOP).

Full correct tuple:
| Field | Value |
|---|---|
| Sourced Statements | "Later that evening, the judges confirmed that McGuire had indeed defeated Good by 370 votes—only four fewer than what was originally determined on election night." |
| Type of Source | Unnamed Group of People |
| Name of Source | (null — always null for this type) |
| Title of Source | judges |
| Source Descriptors | three-panel recount court |
| Source Justification | "On August 1, a three-panel recount court reviewed the election returns that officials from 24 localities transported to the Goochland County Circuit Court" |

(No incorrect companion needed — this is a positive precedent example. The
"incorrect" state this row was actually in before its fix was a Type-of-Source
misclassification, not a Source Descriptors error; see `development-of-v59.md`
finding D for that separate fix. This example exists purely to demonstrate
what a good Source Descriptors value looks like for this flavor of UGOP.)

Illustrates: Source Descriptors for Unnamed Group of People can name an
institutional or convened body defined by what it was formed to do (the panel
was convened specifically to conduct the recount), not just an informal
identity/action word like "protestors" or "attendees" — the definition
already covers this, the example list just didn't demonstrate it before this
addition.

---

## Example 9 — Named Org Source Descriptors: category word is correct, function language is not (illustrates item 14)

**REAL.** `GT-2026/42-Uber-Lyft-CA.csv`, row 8. Source: Gridwise.

Full correct tuple:
| Field | Value |
|---|---|
| Sourced Statements | "Even as average rideshare prices rose over 7% from 2023 to 2024, Uber driver earnings fell 3.4% and Lyft driver earnings dropped 13.9%, according to Gridwise, an app that helps drivers track mileage and optimize earnings." |
| Type of Source | Named Organization |
| Name of Source | Gridwise |
| Title of Source | (null — never valid for Named Organization) |
| Source Descriptors | app |
| Source Justification | "an app that helps drivers track mileage and optimize earnings" |

Full incorrect tuple as initially proposed during this migration pass (not
what original GT had — GT had left Source Descriptors null; this is the
*tempting wrong fix* that was considered and walked back before landing on
the correct one above):
| Field | Value |
|---|---|
| Sourced Statements | (same as above, unchanged) |
| Type of Source | Named Organization |
| Name of Source | Gridwise |
| Title of Source | (null) |
| Source Descriptors | **an app that helps drivers track mileage and optimize earnings** *(the full functional clause, wrongly compressed into this field instead of "app")* |
| Source Justification | (null) |

Supporting story-text material: none needed beyond this row's own text —
Gridwise is mentioned exactly once in the entire article, only in this
sentence (`extracted_articles_boilerplate/42-california's_uber_and_lyft_drivers_may_soon_be_for.txt`,
line 24).

Illustrates: "app" is a bare category word (functions the same way "tech
company" would) and belongs in Source Descriptors; the fuller functional
elaboration belongs in Source Justification instead, even though it's fully
duplicative of text already visible in the Sourced Statement for this row —
same overlap principle already established elsewhere (SD/SJ may legitimately
share text).

---

## Example 10 — Named Org Source Descriptors: no bare category word present, correctly left null (illustrates item 14)

**REAL.** `GT-2026/25-ballot-access-trans.csv`, row 5. Source: Movement Advancement Project.

Full correct tuple:
| Field | Value |
|---|---|
| Sourced Statements | "At least nine states in the past two years have explicitly regulated gender in this way, according to a tally by the Movement Advancement Project (MAP), which tracks LGBTQ+ policy." |
| Type of Source | Named Organization |
| Name of Source | Movement Advancement Project |
| Title of Source | (null) |
| Source Descriptors | (null) |
| Source Justification | "the Movement Advancement Project (MAP), which tracks LGBTQ+ policy." |

(No incorrect companion needed — this is a positive precedent example
showing the correct null result. A tempting-but-wrong fix would be forcing
"which tracks LGBTQ+ policy" — a function description, not a bare category
word — into Source Descriptors.)

Illustrates: when the text gives only function/mission language and no bare
category word for the organization, Source Descriptors is correctly left
null — the function/mission text belongs in Source Justification, not
squeezed into Source Descriptors just because it's the only characterizing
language available.

---

## Example 11 — Source Justification carry-forward across non-adjacent paragraphs, and what does NOT count as SJ (illustrates Note 22/24's SJ carry-forward rule)

**REAL test-annotation output, not GT.** RAI-004 (2026-09 CNBC test-annotation,
"From Silicon Valley to DC, the tech world is suddenly obsessed with one
concept in AI: Distillation"). Source: Anthropic (Named Organization), rows
17-19 of three non-adjacent paragraphs.

Row 17 (first appearance — establishes the Source Justification):
| Field | Value |
|---|---|
| Sourced Statements | "However, Anthropic has a different view, because the company sees how its models are being used and has a burgeoning business to protect. In February, the company said its Claude capabilities were being distilled on an "industrial scale" by China's DeepSeek, Moonshot, and MiniMax, which used about 24,000 fake accounts, generating 16 million exchanges." |
| Type of Source | Named Organization |
| Name of Source | Anthropic |
| Title of Source | (null — never valid for Named Organization) |
| Source Descriptors | (null — no bare category word for Anthropic is stated in this text) |
| Source Justification | "because the company sees how its models are being used and has a burgeoning business to protect" |

Row 18 (next paragraph — no new SJ-qualifying text of its own):
| Field | Value |
|---|---|
| Sourced Statements | "Anthropic, which is valued at close to $1 trillion and has aspirations of going public in the near future, said stopping illicit distillation was a matter of national security." |
| Type of Source | Named Organization |
| Name of Source | Anthropic |
| Title of Source | (null) |
| Source Descriptors | (null) |
| Source Justification | "because the company sees how its models are being used and has a burgeoning business to protect" *(carried forward from row 17 — "valued at close to $1 trillion and has aspirations of going public" is company background, not a stakeholdership/relevance signal, so it does NOT independently qualify as SJ and does not replace the carried-forward value)* |

Row 19 (a further paragraph later — still no new SJ-qualifying text):
| Field | Value |
|---|---|
| Sourced Statements | ""Anthropic and other US companies build systems that prevent state and non-state actors from using AI to, for example, develop bioweapons or carry out malicious cyber activities," the company said in its February post. And stopping it requires "rapid, coordinated action among industry players, policymakers, and the global AI community."" |
| Type of Source | Named Organization |
| Name of Source | Anthropic |
| Title of Source | (null) |
| Source Descriptors | (null) |
| Source Justification | "because the company sees how its models are being used and has a burgeoning business to protect" *(carried forward again, same reasoning as row 18)* |

Supporting story-text material: none needed beyond the rows' own text — the
question here is not about missing context, it's about correctly judging
which candidate phrases qualify as SJ (per its own definition) versus which
are just company background, and whether the earlier SJ should still carry
forward when a later row doesn't independently qualify.

Illustrates: (a) Note 22/24's SJ carry-forward rule working correctly across
non-adjacent paragraphs, not just an immediately-following one; (b) a
judgment call on the *boundary* of valid SJ — "valued at close to $1 trillion
and has aspirations of going public" describes the company but doesn't
explain why Anthropic is quoted specifically about distillation, so it's
correctly excluded even though it appears in the same sentence as an
attribution; (c) the earlier, still-relevant SJ value keeps carrying forward
rather than being nulled out just because the current row's own text doesn't
independently support a new one.

---

## Notes for whoever builds the final JSON few-shot set

- Examples 1, 2, 3, 5, 6, 7, 8, 9, and 10 are real, verified GT cases as of this
  writing — safe to use directly, field values pulled straight from the
  actual GT-2026 CSVs and cross-checked against the source article text
  files.
- Example 4 is explicitly hypothetical/invented — do not present it as a real
  case; either keep it clearly marked as illustrative, or replace it with a
  real GT row if one turns up.
- Example 7's "incorrect" tuple is not what original GT had (GT simply left
  both fields null) — it's the tempting-but-wrong fix that was proposed and
  rejected mid-pass. Worth keeping as a few-shot pair anyway since it
  demonstrates the specific reasoning error (mistaking a defined job for a
  licensed one), but flag this provenance difference if the JSON format
  distinguishes "actual GT error" from "proposed-and-rejected error."
- Example 11 has a different provenance from the others: it's drawn from a
  live v60 test-annotation run (RAI-004, a CNBC article not part of GT-2026),
  not from an existing GT CSV. Field values were produced by actually applying
  the current system_prompt_v60/user_prompt_v60 instructions to the article
  text, not pulled from human-annotated ground truth — flag this distinction
  if the JSON format needs to track example provenance.
- More examples will be appended here as drafting continues on further Prompt
  Updates/Development checklist items in `development-of-v59.md`.
