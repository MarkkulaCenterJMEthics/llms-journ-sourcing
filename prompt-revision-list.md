# Prompt revision list (next: v61)

Current prompt version: **v60** (`new_prompts/system_prompt_v60.*`, `new_prompts/user_prompt_v60_csv.*`). This list collects what goes into the next revision, split by size:

- **Minor:** wording that clarifies or extends an existing rule. No schema change, no new design decision.
- **Major:** a new schema field, a new design principle, or a behavior change that still needs a design decision or sign-off before it can be drafted.

Item numbers in parentheses point to `development-of-v59.md`: "Prompt Updates" items (refinements to existing rules) and "Prompt Development" items (new schema/design questions). The initial minor/major split below follows those two sections and is open for review.

## Minor (v61)

1. **Body-only annotation scope.** *Added 2026-10-08. Reminder set for 2026-10-09: decide the exact prompt wording and where it goes.* Rule adopted 2026-10-08 (`development-of-v59.md` items 51 and 53). The current user prompt's step 1 only says "ignore the Subtitle"; it doesn't cover captions, scrolling-graphic text panels, summary boxes, graphics, or images of documents. The human-facing instruction below is final and in use. For the LLM prompt, replace the subtitle sentence in step 1 (and the "(Ignore the Subtitle text if it exists)" note in the opening line) with a short version, e.g.: *"Annotate only the body of the article: its paragraphs, photo captions, and any article-style text panels. Ignore the headline, subtitle, summary boxes ('Summary', 'Key points', etc.), and any text from charts, maps or tables."* Since extraction now strips subtitles, summary boxes and graphic text before the model sees the article (`article_body()` in `v10-extract-multiple-LLMs.py`), this line is a backstop for anything that slips through.

   Human-facing instruction (as adopted):

   > **What to annotate: the body of the article**
   >
   > Annotate sourced statements **only in the body of the article**. The body includes:
   > - every paragraph of the article's main text, from first to last
   > - **photo captions** (the text under a photo), because they sometimes say where information came from
   > - **text panels in a scrolling graphic** (sometimes called "scrollytelling"), when they read as ordinary article paragraphs
   >
   > **Do not annotate:**
   > - the **headline** or the **subtitle** (the summary line just under the headline)
   > - **summary boxes**, such as "Summary," "Key points," "In summary," "Overview," or "Need to know"
   > - anything that belongs to a **chart, map, table, or other data graphic**: its title, description, labels, notes, or "Source:" line
   > - text that appears **only inside an image**, such as a screenshot of an email or document
   > - **bylines, datelines, author bios,** newsletter sign-ups, ads, "Read more" links, and other page navigation
   >
   > **One thing that is body text:** a sentence in the article that describes a graphic and says where its data comes from, such as *"The map below shows which ZIP codes have the most older homes, according to Census data."* Annotate it as usual.
   >
   > **Why:** Subtitles and summary boxes repeat what the body already says. Graphics and images aren't part of the plain text the models receive. Annotating only the body means human and model annotations are measured against the same text.
   >
   > **If a statement seems important but you only find it in a subtitle, summary box or graphic,** look for the same information in the body. If the body has it, annotate the body version. If it doesn't, leave it out.

2. Title of Source qualifier-trim notes, co-referenced (Prompt Updates 20).
3. Source Descriptors: the missing counterpart to the org-affiliation-stripping paragraph (Prompt Updates 21).
4. Source Descriptors captures the reporter's characterization, not a source's self-description inside a quote (Prompt Updates 22).
5. Note 12 carve-out: a poll/survey/research consortium credited under one combined name stays one row (Prompt Updates 23).
6. Document vs. Named Organization: the organization's own name must be manifestly stated (Prompt Updates 24).
7. Note 8: formal representative capacity vs. individual expert judgment affiliated with an organization (Prompt Updates 25).
8. A bare sovereign-region name used as shorthand for its government is a valid Named Organization (Prompt Updates 26).
9. Extend the Source Descriptors carry-forward rule to Named Organization's category word (Prompt Updates 27).
10. Title of Source carry-forward: say "backfill" explicitly; the current wording only says "later" (Prompt Updates 28).
11. "Teacher" / "instructor" (K-12 context) as valid Title of Source examples (Prompt Updates 29).
12. Org-affiliation stripping applies to a named person too, not just a named organization (Prompt Updates 30).
13. Generic sourcing-role words ("interviewee," "source") aren't meaningful Source Descriptors (Prompt Updates 31).

## Major

1. Secondary-source signal for Named Organization ("reported earlier by," "first reported by") — schema or detection-rule question (Prompt Development 20).
2. Document vs. Named Organization when an organization's own report is used to source a claim about itself — needs more real examples (Prompt Development 21).
3. An unnamed spokesperson attribution resolving backward to a later-named spokesperson — needs guardrail sign-off (Prompt Development 22).
4. Detecting an implied document attribution (a verbatim quoted phrase in otherwise unattributed reporter text) (Prompt Development 31).

Related but not a prompt item: Prompt Development 25 (extraction dropping chart captions) is settled by the graphics rule (item 53): graphic text is out of scope for the text benchmark.
