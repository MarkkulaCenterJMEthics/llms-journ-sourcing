# Few-Shot Examples for the v60 API Completions Payload

This folder holds the few-shot example bank and a reference script for
building the actual OpenRouter chat-completions payload for v60 sourcing
annotation, plus the design reasoning behind the shape of both. It's meant to
be read alongside `new_prompts/system_prompt_v60.txt`/`.md` and
`new_prompts/user_prompt_v60_csv.txt`/`.md`, and `development-of-v59.md` /
`v59-worked-examples.md` for the fuller design history behind each rule these
examples demonstrate.

## Files

- `examples_bank.json` — 13 real worked examples (see below), each with a
  correct 6-field annotation, an incorrect companion where one exists, and a
  short explanation. Source of truth for the few-shot content.
- `build_payload.py` — reference implementation that assembles an actual
  OpenRouter request from the bank + the v60 prompts + a target article. Run
  it directly (`python3 build_payload.py path/to/article.txt [model_name]`)
  to see the exact JSON payload shape, or import `build_messages()` from it.

## Why the payload is shaped the way it is

### 1. OpenRouter's chat completions API is OpenAI-compatible, and works the same across providers

There's no dedicated "few-shot examples" field in the API. `messages` is
just an array of `{role, content}` objects (`system`/`user`/`assistant`/
`developer`/`tool`), and it's explicitly documented as multi-turn: "This
enables few-shot examples... by alternating user, assistant, and tool
messages in sequence." This is uniform across every model on OpenRouter,
including Anthropic Claude and Qwen — OpenRouter "normalizes the schema
across models and providers so you only need to learn one." Unsupported
parameters for a given provider are just silently dropped, not errors.

### 2. Why the real query comes *after* all the few-shot examples

The `messages` array is a literal chronological conversation history. Few-shot
examples work by faking a *prior* exchange ("a user asked this before, you
replied like this"), and the real, current request has to be the **last**
message — that's what "last" means in a conversation. There's no coherent
alternative ordering: putting the instructions in their own early turn before
the examples would require an artificial "OK, ready" filler assistant reply
with nothing to actually respond to yet, which wastes tokens for no benefit.

Concretely: `user_prompt_v60_csv.txt`'s first line is `"Read the attached
news article carefully %s..."` — the article was always meant to be
substituted directly into the *same* message as the instructions. So the
full 12-step procedural text from the user prompt is attached only once,
to the final real turn, not repeated on every few-shot pair (which would
just bloat the cached prefix with identical text N times for no benefit —
see caching below). The few-shot `user` turns carry just the bare article
excerpt; the system prompt already carries every substantive classification
rule, and steps 1-11 of the user prompt are explicitly silent scratch-work
steps ("no need to print anything yet") — the only observable input→output
mapping is *excerpt in, CSV out*, which is exactly what a bare-excerpt
few-shot pair demonstrates.

### 3. Correct-only, not contrastive — a deliberate cost tradeoff

Every few-shot example repeats on **every single API call**. A contrastive
format (showing the wrong answer plus an explanation, or a fake
self-correction turn) roughly doubles the token footprint of each example,
for every request. Given the priority was minimizing token cost, we went
correct-only: each example bank entry still *keeps* an `incorrect_annotation`
and `explanation` field for human documentation and QA, but the actual
request builder never sends those fields to the model.

Why this doesn't lose much precision: the reason contrastive framing can help
is that it closes off a specific plausible-but-wrong path more explicitly
than a positive example alone. But `system_prompt_v60` already does this job,
for free, once — every Note that includes a "not X, because Y" contrast
(Note 4's disclosure test, Note 19's judge/immigration-judge contrast, Note
13's police-department example) is exactly this technique, just paid for
once as part of the system prompt's fixed, cached cost rather than repeated
per few-shot example. If a specific model turns out to still get a specific
distinction wrong in testing, `build_payload.py`'s design makes it easy to
add an inline caution sentence to that one example's `user` turn later
(pulling from its existing `explanation` field) without restructuring
anything — an option to reach for selectively if needed, not a default.

### 4. Prompt caching: explicit `cache_control` marker required for Anthropic and Qwen

Confirmed against OpenRouter's docs (2026-09):

- **Automatic caching, no marker needed:** OpenAI, Grok, Moonshot AI, Groq,
  DeepSeek, Z.AI, Google Gemini 2.5.
- **Requires an explicit `cache_control` marker:** **Anthropic Claude and
  Qwen** — without it, no caching discount applies at all for these two.

The marker is `{"type": "text", "text": "...", "cache_control": {"type":
"ephemeral"}}` — note `content` has to switch from a plain string to an
array of content-block objects on the one message carrying the marker.

A single breakpoint, placed on the **last** few-shot `assistant` turn's
content block, covers the whole stable prefix (system prompt + every
example) in one shot — a cache_control breakpoint caches everything *up to
and including* the block it's on, so there's no need for a second breakpoint
on the system message separately. `build_payload.py` applies this marker
**unconditionally on every request**, regardless of target model: it's
required for Anthropic/Qwen, and simply ignored (harmless) for providers
that cache automatically — one code path works everywhere.

Cost structure: Anthropic charges cache *reads* at ~0.1x normal input
pricing but cache *writes* at 1.25x-2x (the first call in a session "pays
extra" to populate the cache; every repeat call within the cache window — 5
minutes by default, 1 hour if opted into — reads cheaply). This lines up
with the ~40-50% savings already observed in production logs for a workload
that re-sends the same system+few-shot prefix across many articles.
OpenRouter also uses "sticky routing" to keep repeat requests on the same
backend provider so the cache actually stays warm across a batch run.

## The 13 examples in the bank

Each demonstrates a specific v59→v60 rule or a real segmentation/annotation
edge case. Full reasoning and citations for each are in
`v59-worked-examples.md` (same numbering minus the excluded ones below):

1. Title of Source carry-forward (Trump/Tim Walz)
2. Title of Source: no contextual inference (Espinoza/immigration judge)
3. Title of Source: coreference resolution IS allowed (Sutskever/board member)
4. Multi-descriptor accumulation in one row (Roger Clark: friend, programmer, DDR enthusiast)
5. Non-credentialed source's org affiliation → Source Justification, not Source Descriptors (Guzman)
6. Real professional role that's still non-credentialing (Blackburn/paralegal)
7. UGOP Source Descriptors as institutional/convened-body label (three-panel recount court)
8. Named Org Source Descriptors: category word correct, function language is not (Gridwise)
9. Named Org Source Descriptors: no category word present, correctly null (Movement Advancement Project)
10. Source Justification carry-forward across non-adjacent paragraphs (Anthropic, 3 rows)
11. Non-contiguous same-source rows interrupted by a different source (Waymo/Rob Moore, 4 rows)
12. Anonymous Source disclosure test, correctly applied (Hormuz)
13. Title of Source embedded inside Source-Justification-shaped text (Venus Williams)

**Deliberately excluded from the bank right now:** the Waymo/Sandy Karp
"unnamed spokesperson resolved backward to a name given later in the story"
case (Example 1 in the original "1-Examples for API completion" doc). This
pattern is confirmed intentional on journalism-practice grounds (a company
won't typically put two different spokespeople on the same story at once),
but the exact guardrail scope hasn't been signed off yet — logged as item 22
in `development-of-v59.md`, pending that decision. Also excluded: any
HYPOTHETICAL example (invented, no real case backing it) — the bank is
real-cases-only for now.

## Extending the bank

Add new entries to `examples_bank.json` following the existing schema
(`example_id`, `source`, `illustrates`, `article_excerpt`,
`correct_annotation`, `incorrect_annotation` or `null`, `explanation`).
`build_payload.py` will pick up new entries automatically — no code changes
needed unless the schema itself changes.
