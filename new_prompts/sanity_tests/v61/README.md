# v61 prompt sanity test

The v61 annotation prompts and two news stories with human ground truth (GT), for a first check that the prompts behave as intended before the full benchmark run. The files stay where they live in the repo; this README points to them. The copy sent to students (2026-11-06) bundles the same files into `prompts/`, `stories/` and `ground_truth/` folders.

## What's here

- Prompts (in `new_prompts/`)
  - [`system_prompt_v61.txt`](../../system_prompt_v61.txt) and [`user_prompt_v61_csv.txt`](../../user_prompt_v61_csv.txt) are the prompts to send to a model. The `.md` versions have the same text, formatted for reading.
- Stories (in `extracted_articles_boilerplate/`)
  - [`53-iranian_sea_trade_blocked_as_six_ships_forced_to_t.txt`](../../../extracted_articles_boilerplate/53-iranian_sea_trade_blocked_as_six_ships_forced_to_t.txt): Iranian sea trade blocked (13 GT rows)
  - [`58-what_we_know_about_the_killing_of_8_children_in_lo.txt`](../../../extracted_articles_boilerplate/58-what_we_know_about_the_killing_of_8_children_in_lo.txt): the Shreveport, Louisiana killings (21 GT rows)
- Ground truth (in `benchmarking/GT data/GT-II/`): the human annotation for each story, one CSV per story.
  - [`53-Iran_Sea_Trade_Blocked.csv`](<../../../benchmarking/GT data/GT-II/53-Iran_Sea_Trade_Blocked.csv>)
  - [`58-Louisiana_Shooting.csv`](<../../../benchmarking/GT data/GT-II/58-Louisiana_Shooting.csv>)

When sending a story to a model, keep the Title line and the text after the second `=====` line, and drop the rest of the header block (Subtitle, URL, Author, Date, Source PDF). The benchmark run sends only the title and body, so this keeps the test the same.
