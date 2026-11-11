"""
eval_multi_llm_article_v2.py -- v59-schema evaluator.

A v2 of eval_multi_llm_article.py that scores the 6-column v59 schema (the
added `Source Descriptors` field) and fixes several defects that matter
specifically for v59 output. The matching methodology is otherwise unchanged --
fuzz_split statement matching at the configured threshold, semantic
title/justification rules at 0.55, exact Type match, Name at fuzz > 80 -- so
numbers stay comparable to the historical benchmark runs.

What is new
-----------
* Reads and scores `Source Descriptors` on both sides. Match rule is null-aware
  (an empty GT cell equals the literal word "null" a model writes) and
  order-insensitive across comma-separated multi-values.
* Lenient descriptor scoring alongside the strict one (additive): partial
  credit per row (recall/precision/F1 over the label set) and near-miss credit
  per label via whole-word containment, fuzz >= 80, or semantic >= 0.65.
  Reported as Descr(soft) / Descr(softPrec); strict Descr(present) unchanged.
* Accepts both schemas: `--schema v59` (6 columns) and `--schema v57`
  (5 columns, Descriptors treated as absent).
* Command line driven. The original required editing main() before every run.
* `Unresolved` is a recognised Type of Source, since user_prompt_v59 step 2
  explicitly allows it. It maps to its own bucket, so it never matches a GT type.

Optional matching fixes (see `class fixes`) -- ALL OFF BY DEFAULT
-----------------------------------------------------------------
By default the matching behaves exactly like eval_multi_llm_article.py, so the
only difference in the numbers is the added Source Descriptors metric. The
fixes below change which rows pair up and therefore change the scores; they
are opt-in via --apply-fixes and documented here so their effect can be
audited (run with and without the flag and diff the summaries).

* exclusive_claims -- one LLM row can no longer be claimed by several GT rows.
  Previously the Round 1 search never excluded already-claimed rows, so on
  repeated statement text a single LLM row satisfied several GT rows and both
  recall and precision were inflated.
* pair_on_name -- when several LLM rows carry identical statement text (which
  v59 Note 12 requires for a statement attributed to several sources), the
  partner is chosen by closest Name rather than by file order. Previously the
  first row always won, so the wrong source's Name/Title was graded and correct
  answers scored wrong.
* fix_round2_tuple -- Round 2 compared against the `(index, sentence)` tuple
  instead of the sentence, so its guard scored against a stringified tuple and
  its containment test could never fire.
* bidirectional_containment -- the containment guard only tested
  LLM-inside-GT. That forgives a model splitting a statement but penalises one
  merging several, which the v59 prompt actively asks for.
* Rows are renumbered after empty-statement rows are dropped. The original kept
  the old labels while using them as list positions, which misaligned sentences
  whenever any row was dropped.
* Field rates no longer crash when a story produces zero matches; they report
  NaN (not measurable) instead.

Usage
-----
    python eval_multi_llm_article_v2.py         --gt-dir "benchmarking/GT data/GT-2026"         --pred-glob "../rai-scu-kite/sourcing_data/v59_gt_benchmark/story*_v59.csv"         --out-dir benchmarking/metrics/v59

    # opt in to the matching fixes (numbers no longer comparable to history)
    python eval_multi_llm_article_v2.py ... --apply-fixes

Source Justification grading -- --justification-mode legacy | best-match
------------------------------------------------------------------------
Default `legacy`. See SJ_Review_Proposal.md for the evidence behind each item.

* `legacy` -- every distinct Justification sharing a Name is glued into one
  comma-joined blob on both sides and the blobs are compared whole-string.
  A source the human explained twice and the model explained once therefore
  scores the model's one answer against both glued together, which dilutes a
  correct answer below the 0.55 bar (measured: 0.7398 -> 0.2819 on one real
  row, and the glue is copied onto every row of that source). Blank-vs-blank
  scores 1.0 and counts as correct.
* `best-match` -- the per-source justifications are kept as a LIST instead of
  glued, and the model's answer is compared against each of the human's
  separately, keeping the best score. The original generosity is preserved
  (the model's justification may come from any of its rows for that source),
  but a thorough annotator no longer sinks a correct answer. Applies to both
  Rate A (Justification_Match_Rate) and Rate B
  (Title_Justification_Join_Match_Rate) so the A->B gap stays readable.

Blank vs. blank counts as CORRECT in both modes, by design: an empty GT
justification means the article carries no justification for that source, so a
model that also writes nothing has given the right answer. Legacy arrives there
by accident (MiniLM embeds two empty strings identically, so
semantic_match("", "") == 1.0 clears the 0.55 bar); best-match states it
outright so it cannot change silently if the embedding model is swapped.

Applied unconditionally (neither changes which rows pair up):

* `fill_with_same_person` now sorts before joining. The values were collected
  in a `set`, so the join order followed Python's per-process hash seed and no
  justification figure this evaluator ever produced was exactly reproducible
  (measured: story 1 / cl-son5 scored 18.8% on PYTHONHASHSEED=0 and 87.5% on
  PYTHONHASHSEED=1, with Name, Title and Recall unchanged). Sorting removes
  the randomness only; it picks no new answer. Title was latently exposed to
  the same effect and is fixed by the same line.
* The per-pair dump (--dump-comparisons) now also carries `*_Match_Score_Whole`
  (the whole-string ruler the reported rates actually use) and the recorded
  `Justification_Pass` / `Justification_Join_Pass` verdicts, so a published
  percentage can be reconciled by hand against the spreadsheet. The existing
  `*_Match_Score` columns keep the sentence-split ruler because
  `Title_Match_Score` and `Title_and_Justification_Match_Score` are inputs to
  Title_Match_Rate and All_Attribute_Accuracy -- repointing them would have
  changed those published numbers, which this item is not meant to do.
"""

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import argparse
import glob
from functools import lru_cache
import logging
import re
import sys
from collections import defaultdict
import textwrap
import nltk
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

nltk.download("punkt")
import os

import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

from configure import eval_config
import mismatch_log
import sj_eval

run_mode = "debug"  # run or develop


class fixes:
    """
    Optional matching fixes. ALL DEFAULT OFF: out of the box this script scores
    exactly like eval_multi_llm_article.py, plus the Source Descriptors column.
    Pass --apply-fixes (or call enable_all()) to turn them on; results are then
    NOT comparable to historical benchmark numbers.
    """

    exclusive_claims = False
    pair_on_name = False
    fix_round2_tuple = False
    bidirectional_containment = False

    @classmethod
    def enable_all(cls):
        cls.exclusive_claims = True
        cls.pair_on_name = True
        cls.fix_round2_tuple = True
        cls.bidirectional_containment = True

    @classmethod
    def use_legacy(cls):
        cls.exclusive_claims = False
        cls.pair_on_name = False
        cls.fix_round2_tuple = False
        cls.bidirectional_containment = False

    @classmethod
    def describe(cls):
        return ", ".join(
            f"{name}={getattr(cls, name)}"
            for name in (
                "exclusive_claims",
                "pair_on_name",
                "fix_round2_tuple",
                "bidirectional_containment",
            )
        )
class justification:
    """
    How Source Justification is graded. DEFAULT LEGACY: out of the box the
    justification rates reproduce eval_multi_llm_article.py exactly (modulo the
    unconditional sort in fill_with_same_person, which only removes run-to-run
    randomness). Pass --justification-mode best-match to turn on the fixes for
    dilution-by-glue and blank-counts-as-correct; results are then NOT
    comparable to historical benchmark numbers.
    """

    LEGACY = "legacy"
    BEST_MATCH = "best-match"
    CHOICES = (LEGACY, BEST_MATCH)

    mode = LEGACY

    @classmethod
    def best_match(cls):
        return cls.mode == cls.BEST_MATCH

    @classmethod
    def describe(cls):
        return cls.mode


error_matched = defaultdict(int)

# debug variables
all_gt_count_all = 0
both_found_count_all = 0
b_typeok_count_all = 0
b_typenook_nameok_count_all = 0
b_typeok_nameok_count_all = 0
b_typenook_namenook_count_all = 0
b_typeok_nameok_titleok_count_all = 0
b_typeok_namenook_titleok_count_all = 0
b_typenook_namenook_titleok_count_all = 0
b_typenook_nameok_titleok_count_all = 0
b_typeok_namenook_titlenook_count_all = 0
b_typenook_namenook_titlenook_count_all = 0
b_typenook_nameok_titlenook_count_all = 0

all_matched_global_df_list = []

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

SourceTypeMapping = {
    "named individual": "Named_Person",
    "namedindividual": "Named_Person",
    "named": "Named_Person",
    "named person": "Named_Person",
    "namedperson": "Named_Person",
    "named person sources": "Named_Person",
    "named_persons": "Named_Person",
    "named_person": "Named_Person",
    #############################################
    "organization": "Named_Organization",
    "named organization": "Named_Organization",
    "namedorganization": "Named_Organization",
    "organizations": "Named_Organization",
    "named organization sources": "Named_Organization",
    "named_organization": "Named_Organization",
    #############################################
    "unnamed": "Anonymous_Source",
    "unnamed individual": "Anonymous_Source",
    "unnamedindividual": "Anonymous_Source",
    "and “anonymity” = y": "Anonymous_Source",
    "anonymous source": "Anonymous_Source",
    "anonymous_source": "Anonymous_Source",
    "unnamed people": "Anonymous_Source",
    #############################################
    "unnamed person": "Anonymous_Person",
    #############################################
    "anonymous_persons": "Anonymous_Groups",
    "unnamed group of people": "Anonymous_Groups",
    "unamed group of people": "Anonymous_Groups",
    "unnamed group": "Anonymous_Groups",
    "groups": "Anonymous_Groups",
    "unnamed_group_of_people": "Anonymous_Groups",
    "anonymous sources": "Anonymous_Groups",
    #############################################
    "document": "Documents",
    "documents": "Documents",
    "document sources": "Documents",
    # v59 user prompt step 2 allows "Unresolved" when no definition fits; map it
    # to its own bucket so it never matches a GT type (always scored wrong).
    "unresolved": "Unresolved",
    "": "",
}


# Custom function to perform mapping with error handling
def map_with_error(x):
    x_lower = x.lower()
    if x_lower in SourceTypeMapping:
        return SourceTypeMapping[x_lower]
    else:
        raise ValueError(f"Unmapped source type: '{x}'")


def clean_text(text):
    if pd.isna(text) or text == "":
        return ""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def split_into_sentences(text):
    return sent_tokenize(text)


def split_into_sub_sentences(text: str) -> list[str]:
    import re

    sentences = sent_tokenize(text)
    split_by_commas = []
    for sentence in sentences:
        split_by_commas.extend(re.split(r",\s*", sentence))
    return [s.strip() for s in split_by_commas if s.strip()]


def keep_origin_sentence(text):
    return [text]


def fuzzy_compare(row, gt_col, llm_col):
    return fuzz.ratio(clean_text(row[gt_col]), clean_text(row[llm_col]))


def semantic_compare(row, gt_col, llm_col, using_split=True):
    if using_split:
        return semantic_compare_split_sentence(
            clean_text(row[gt_col]), clean_text(row[llm_col])
        )
    else:
        return semantic_match(clean_text(row[gt_col]), clean_text(row[llm_col]))


def fuzzy_compare_split_row(row, gt_col, llm_col):
    return fuzzy_compare_split_sentence(
        clean_text(row[gt_col]), clean_text(row[llm_col])
    )


def fuzzy_compare_split_sentence(ground_truth_text, llm_generated_text):
    # Split the cleaned texts into sentences
    gt_sentences = split_into_sub_sentences(ground_truth_text)
    llm_sentences = split_into_sub_sentences(llm_generated_text)

    # Calculate pairwise similarity scores
    pairwise_scores = [
        fuzz.ratio(clean_text(gt_sentence), clean_text(llm_sentence))
        for gt_sentence in gt_sentences
        for llm_sentence in llm_sentences
    ]

    # Return the maximum similarity score or 0 if no scores are available
    return max(pairwise_scores, default=0)


def semantic_compare_split_sentence(gt_text, llm_text):
    # Split the cleaned texts into sentences
    gt_sentences = split_into_sub_sentences(gt_text)
    llm_sentences = split_into_sub_sentences(llm_text)

    # Calculate pairwise similarity scores
    pairwise_scores = [
        semantic_match(clean_text(gt_sentence), clean_text(llm_sentence))
        for gt_sentence in gt_sentences
        for llm_sentence in llm_sentences
    ]

    # Return the maximum similarity score or 0 if no scores are available
    return max(pairwise_scores, default=0)


def exact_match(row, gt_col, llm_col):
    return "Yes" if clean_text(row[gt_col]) == clean_text(row[llm_col]) else "No"


@lru_cache(maxsize=65536)
def _encode(text):
    """
    Cached MiniLM encode. Pure memoisation -- identical inputs give identical
    vectors, so no score changes. Matters because best-match mode compares
    every GT justification against every model justification for a source, and
    the same strings recur across a source's rows.
    """
    return embedding_model.encode(text, convert_to_tensor=True)


def semantic_match(sentence_one, sentence_two):
    embedding_1 = _encode(sentence_one)
    embedding_2 = _encode(sentence_two)

    sim_score = util.pytorch_cos_sim(embedding_1, embedding_2)
    return float(sim_score.cpu()[0][0])


def find_best_match(sentence, sentences, match_method="fuzz"):
    if match_method == "fuzz":
        best_match = max(
            sentences, key=lambda x: fuzz.ratio(clean_text(sentence), clean_text(x))
        )
        score = fuzz.ratio(clean_text(sentence), clean_text(best_match))
    elif match_method == "fuzz_split":
        best_match = max(
            sentences,
            key=lambda x: fuzzy_compare_split_sentence(
                clean_text(sentence), clean_text(x)
            ),
        )
        score = fuzzy_compare_split_sentence(
            clean_text(sentence), clean_text(best_match)
        )
    elif match_method == "semantic_split":
        best_match = max(
            sentences,
            key=lambda x: semantic_compare_split_sentence(
                clean_text(sentence), clean_text(x)
            ),
        )
        score = semantic_compare_split_sentence(
            clean_text(sentence), clean_text(best_match)
        )
    elif match_method == "semantic":
        best_match = max(
            sentences,
            key=lambda x: semantic_match(clean_text(sentence), clean_text(x)),
        )

        score = semantic_match(clean_text(sentence), clean_text(best_match))
    else:
        raise Exception(f"The method {match_method} is not supported")
    return best_match, score


# Function to fill target_column with combined values for the same person
def fill_with_same_person(df, target_column):
    # Create a mapping of Name -> list of target_column values
    name_set = defaultdict(set)

    for index, row in df.iterrows():
        name = row.get("Name", "")
        value = row.get(target_column, "")
        if name and value:  # Ensure both fields are non-empty
            name_set[name].add(value)

    # Update target_column in the DataFrame
    for index, row in df.iterrows():
        name = row.get("Name", "")
        if name in name_set:
            # sorted(), not list(): name_set is a set, so list() followed the
            # per-process hash seed and the joined blob's word order -- and
            # therefore its embedding and its score -- changed between runs.
            df.at[index, target_column] = ",".join(sorted(name_set[name]))

    return df


def collect_by_person(df, target_column):
    """
    Same pooling as fill_with_same_person, but kept as a LIST in a new
    `{target_column}_All` column instead of glued into one string, and without
    overwriting the row's own value. Lets the scorer compare the model's answer
    against each of the human's separately rather than against all of them
    concatenated (see SJ_Review_Proposal.md, Problem 1).

    A row whose Name is empty is skipped by the pooling (as in
    fill_with_same_person), so its own value is appended explicitly -- otherwise
    nameless sources would lose their justification entirely.
    """
    by_name = defaultdict(list)
    for _, row in df.iterrows():
        name = row.get("Name", "")
        value = row.get(target_column, "")
        if name and value and value not in by_name[name]:
            by_name[name].append(value)

    collected = []
    for _, row in df.iterrows():
        values = list(by_name.get(row.get("Name", ""), []))
        own = row.get(target_column, "")
        if own and own not in values:
            values.append(own)
        collected.append(sorted(values))

    df[target_column + "_All"] = collected
    return df


def preprocess_dataframe(df, statement_column):
    # Normalize text columns
    text_columns = df.select_dtypes(include=["object"]).columns
    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    # Change Null/NaN/empty entries to empty string
    df = df.fillna("")

    # Remove rows where the Statement is empty
    df = df[df[statement_column] != ""]
    # v2: renumber after dropping rows. The original kept the old labels while
    # using them as list positions downstream, which silently misaligned
    # sentences whenever any row was dropped.
    df = df.reset_index(drop=True)

    df["SourceType"] = df["SourceType"].str.lower().map(map_with_error)

    # Must run BEFORE fill_with_same_person, which overwrites the per-row
    # Justification with the glued blob. Cheap and unconditional so the column
    # is always present in the dump; only best-match mode scores from it.
    collect_by_person(df, "Justification")
    # The row's own justification, before fill_with_same_person glues it.
    # sj_eval (--sj-eval) needs it to know WHICH row an SJ was written on.
    df["Justification_Row"] = df["Justification"]

    fill_with_same_person(df, "Title")  #
    fill_with_same_person(df, "Justification")  #

    # Keep origin statements
    df["Sentences"] = df[statement_column].apply(keep_origin_sentence)
    df["Title_and_Justification"] = df["Title"] + " " + df["Justification"]

    return df


def normalize_source_type(source_type):
    if pd.isna(source_type):
        return ""
    try:
        normalized = str(source_type).lower().strip()
        normalized = re.sub(r"\s+", "", normalized)
        return normalized
    except AttributeError:
        return str(source_type)


def load_and_preprocess_ground_truth(file_path):
    """
    Load a human ground-truth CSV. Accepts both schemas: the v59 6-column form
    with `Source Descriptors`, and the older 5-column form (in which case
    Descriptors is created empty so scoring still runs and simply reports
    everything as null-vs-null).
    """
    columns_to_keep = {
        "Sourced Statements": "SourcedStatement",
        "Type of Source": "SourceType",
        "Name of Source": "Name",
        "Title of Source": "Title",
        "Source Descriptors": "Descriptors",
        "Source Justification": "Justification",
    }

    gt_df = pd.read_csv(file_path, encoding="utf-8-sig")
    gt_df.columns = [str(c).replace("﻿", "").strip() for c in gt_df.columns]
    present = {k: v for k, v in columns_to_keep.items() if k in gt_df.columns}
    missing = set(columns_to_keep) - set(present) - {"Source Descriptors"}
    if missing:
        raise ValueError(f"{file_path}: ground truth is missing columns {sorted(missing)}")

    gt_df = gt_df[list(present)].rename(columns=present)
    if "Descriptors" not in gt_df.columns:
        gt_df["Descriptors"] = ""

    return preprocess_dataframe(gt_df, "SourcedStatement")


def load_predictions(file_path, schema="v59"):
    """Load a model prediction CSV under the requested schema."""
    if schema == "v59":
        return load_and_preprocess_llm_v59(file_path)
    if schema == "v57":
        return load_and_preprocess_llm(file_path)
    raise ValueError(f"unknown schema {schema!r}")


def load_and_preprocess_llm_v59(file_path):
    """
    Load a v59 prediction CSV (run_v59_gt_benchmark.py output): the 6-column
    table with 'Sourced Statement' etc. headers and the literal word null for
    empty fields.
    """
    columns_to_keep = {
        "Sourced Statement": "SourcedStatement",
        "Type of Source": "SourceType",
        "Name of Source": "Name",
        "Title of Source": "Title",
        "Source Descriptors": "Descriptors",
        "Source Justification": "Justification",
    }
    llm_df = pd.read_csv(file_path, encoding="utf-8-sig")
    llm_df.rename(columns=columns_to_keep, inplace=True)
    # v59 CSV output writes the literal word null for empty fields
    llm_df = llm_df.replace(
        to_replace=r"(?i)^\s*(null|`null`|none|n/a)\s*$", value="", regex=True
    )
    llm_df = preprocess_dataframe(llm_df, "SourcedStatement")
    return llm_df


def load_and_preprocess_llm(file_path):
    """
    process llm csv file.
    """
    columns_to_keep = {"Name_of_Source": "Name of Source"}
    try:
        llm_df = pd.read_csv(file_path, encoding="ISO-8859-1")
    except Exception as e:
        logging.info(f" read llm csv {file_path} with error {e}.")
        llm_df = pd.read_csv(file_path)

    llm_df.rename(columns=columns_to_keep, inplace=True)
    if "Descriptors" not in llm_df.columns:
        llm_df["Descriptors"] = ""
    llm_df = preprocess_dataframe(llm_df, "SourcedStatement")

    return llm_df


def contained(inner, outer):
    """
    Does one whole statement sit inside the other? The original only tested
    LLM-inside-GT, which forgives a model that splits a statement but penalises
    one that merges several -- and the v59 prompt asks models to merge
    contiguous same-source statements.
    """
    a, b = clean_text(inner), clean_text(outer)
    if not a or not b:
        return False
    if fixes.bidirectional_containment:
        return a in b or b in a
    return a in b


def resolve_row_index(candidates, best_text, gt_row, llm_df):
    """
    Turn the winning statement text back into a row label.

    find_best_match returns text rather than an index, so the row has to be
    looked up again. When several LLM rows carry identical statement text --
    which v59 Note 12 requires for one statement attributed to several sources
    -- the original always took the first, so the wrong source's Name and Title
    could be graded and a correct answer scored wrong. With pair_on_name the tie
    is broken by whichever candidate's Name is closest to this GT row's Name.
    """
    idxs = [idx for idx, sentence in candidates if sentence == best_text]
    if not idxs:
        return None
    if fixes.pair_on_name and len(idxs) > 1:
        gt_name = clean_text(str(gt_row.get("Name", "")))
        idxs.sort(
            key=lambda i: -fuzz.ratio(gt_name, clean_text(str(llm_df.loc[i, "Name"])))
        )
    return idxs[0]


def compare_attributions(gt_df, llm_df):
    # Prepare the result dataframe
    result_df = pd.DataFrame(
        columns=[
            "GT_Sentence",
            "LLM_Sentence",
            "Sentence_Match_Score",
            "Source_Type_GT",
            "Source_Type_LLM",
            "Source_Type_Match",
            "Name_GT",
            "Name_LLM",
            "Name_Match_Score",
            "Title_GT",
            "Title_LLM",
            "Title_Match_Score",
            "Justification_GT",
            "Justification_LLM",
            "Justification_Match_Score",
        ]
    )

    tie_df = pd.DataFrame(
        columns=[
            "GT_Sentence",
            "LLM_Sentence",
            "Sentence_Match_Score",
            "Source_Type_GT",
            "Source_Type_LLM",
            "Source_Type_Match",
            "Name_GT",
            "Name_LLM",
            "Name_Match_Score",
            "Title_GT",
            "Title_LLM",
            "Title_Match_Score",
            "Justification_GT",
            "Justification_LLM",
            "Justification_Match_Score",
        ]
    )

    # Flatten the sentences for easier comparison
    gt_sentences = [
        (idx, sentence)
        for idx, sentences in gt_df["Sentences"].items()
        for sentence in sentences
    ]
    llm_sentences = [
        (idx, sentence)
        for idx, sentences in llm_df["Sentences"].items()
        for sentence in sentences
    ]

    # ------------------------------------------------------------------
    # Round 1 -- each GT statement looks for its best LLM partner.
    # ------------------------------------------------------------------
    sent_by_llm_idx = dict(llm_sentences)
    sent_by_gt_idx = dict(gt_sentences)
    matching_cache = defaultdict(list)
    matched_llm_indexes = set()

    for gt_idx, gt_sentence in gt_sentences:
        # v2 (exclusive_claims): a partner already taken by an earlier GT row is
        # off the table. Without this one LLM row could satisfy several GT rows
        # whenever statement text repeats, inflating recall and precision.
        candidates = [
            (idx, sentence)
            for idx, sentence in llm_sentences
            if not fixes.exclusive_claims or idx not in matched_llm_indexes
        ]
        if not candidates:
            break

        best_match, score = find_best_match(
            gt_sentence, [s for _, s in candidates], eval_config.match_method
        )
        if score <= eval_config.match_threshold[eval_config.match_method]:
            continue

        llm_idx = resolve_row_index(candidates, best_match, gt_df.loc[gt_idx], llm_df)
        if llm_idx is None:
            continue

        # Safety gate: the statements must also be broadly similar as wholes, or
        # one must sit inside the other.
        llm_text = sent_by_llm_idx[llm_idx]
        global_score = fuzz.ratio(clean_text(gt_sentence), clean_text(llm_text))
        if global_score >= 30 or contained(llm_text, gt_sentence):
            matching_cache[gt_idx].append((llm_idx, score))
            matched_llm_indexes.add(llm_idx)

    # ------------------------------------------------------------------
    # Round 2 -- LLM statements nobody claimed get a second chance in reverse,
    # with a looser guard.
    # ------------------------------------------------------------------
    unmatched_llm = [
        (idx, s) for idx, s in llm_sentences if idx not in matched_llm_indexes
    ]

    final_unmatched_llm = set()
    for llm_idx, llm_sentence in unmatched_llm:
        best_match, score = find_best_match(
            llm_sentence, [s for _, s in gt_sentences], eval_config.match_method
        )
        if score <= eval_config.match_threshold[eval_config.match_method]:
            final_unmatched_llm.add(llm_idx)
            continue

        gt_idx = next(idx for idx, s in gt_sentences if s == best_match)
        if fixes.fix_round2_tuple:
            gt_text = sent_by_gt_idx[gt_idx]
            global_score = fuzz.ratio(clean_text(gt_text), clean_text(llm_sentence))
            inside = contained(llm_sentence, gt_text)
        else:
            # Legacy behaviour: compares against the (index, sentence) tuple, so
            # the score is taken against a stringified tuple and `inside` is an
            # element test that effectively never fires.
            legacy_target = gt_sentences[gt_idx]
            global_score = fuzz.ratio(
                clean_text(legacy_target), clean_text(llm_sentence)
            )
            inside = clean_text(llm_sentence) in legacy_target

        if global_score >= 20 or inside or len(matching_cache[gt_idx]) > 0:
            matching_cache[gt_idx].append((llm_idx, score))
        else:
            final_unmatched_llm.add(llm_idx)

    for gt_idx, gt_sentence in gt_sentences:
        gt_row = gt_df.loc[gt_idx]
        if gt_idx not in matching_cache:  # no any matching in LLM sentence.
            new_row = pd.DataFrame(
                {
                    "GT_Sentence": [gt_sentence],
                    "LLM_Sentence": [""],
                    "Sentence_Match_Score": [0],
                    "Source_Type_GT": [gt_df.loc[gt_idx, "SourceType"]],
                    "Source_Type_LLM": [""],
                    "Name_GT": [gt_df.loc[gt_idx, "Name"]],
                    "Name_LLM": [""],
                    "Title_GT": [gt_df.loc[gt_idx, "Title"]],
                    "Title_LLM": [""],
                    "Justification_GT": [gt_df.loc[gt_idx, "Justification"]],
                    "Justification_LLM": [""],
                    "Descriptors_GT": [gt_df.loc[gt_idx, "Descriptors"]],
                    "Descriptors_LLM": [""],
                    "Justification_All_GT": [gt_df.loc[gt_idx, "Justification_All"]],
                    "Justification_All_LLM": [[]],
                    "Title_and_Justification_GT": [
                        gt_df.loc[gt_idx, "Title_and_Justification"]
                    ],
                    "Title_and_Justification_LLM": [""],
                }
            )
        elif len(matching_cache[gt_idx]) == 1:
            llm_idx, score = matching_cache[gt_idx][0]
            llm_row = llm_df.loc[llm_idx]
            new_row = pd.DataFrame(
                {
                    "GT_Sentence": [gt_sentence],
                    "LLM_Sentence": [llm_sentences[llm_idx][1]],
                    "Sentence_Match_Score": [score],
                    "Source_Type_GT": [gt_row["SourceType"]],
                    "Source_Type_LLM": [llm_row["SourceType"]],
                    "Name_GT": [gt_row["Name"]],
                    "Name_LLM": [llm_row["Name"]],
                    "Title_GT": [gt_row["Title"]],
                    "Title_LLM": [llm_row["Title"]],
                    "Justification_GT": [gt_row["Justification"]],
                    "Justification_LLM": [llm_row["Justification"]],
                    "Descriptors_GT": [gt_row["Descriptors"]],
                    "Descriptors_LLM": [llm_row["Descriptors"]],
                    "Justification_All_GT": [gt_row["Justification_All"]],
                    "Justification_All_LLM": [llm_row["Justification_All"]],
                    "Title_and_Justification_GT": [gt_row["Title_and_Justification"]],
                    "Title_and_Justification_LLM": [llm_row["Title_and_Justification"]],
                }
            )
        else:
            llm_idx_candidates = [x[0] for x in matching_cache[gt_idx]]
            scores_candidates = [x[1] for x in matching_cache[gt_idx]]
            max_score = max(scores_candidates)

            indices = []
            for i, score in enumerate(scores_candidates):
                if score == max_score:
                    indices.append(i)

            if len(indices) == 1:
                # only one maximum:
                llm_idx = llm_idx_candidates[indices[0]]
                llm_row = llm_df.loc[llm_idx]
                merge_sentence = ""
                for i in llm_idx_candidates:
                    merge_sentence += llm_sentences[i][1]

                new_row = pd.DataFrame(
                    {
                        "GT_Sentence": [gt_sentence],
                        "LLM_Sentence": [merge_sentence],
                        "Sentence_Match_Score": [max_score],
                        "Source_Type_GT": [gt_row["SourceType"]],
                        "Source_Type_LLM": [llm_row["SourceType"]],
                        "Name_GT": [gt_row["Name"]],
                        "Name_LLM": [llm_row["Name"]],
                        "Title_GT": [gt_row["Title"]],
                        "Title_LLM": [llm_row["Title"]],
                        "Justification_GT": [gt_row["Justification"]],
                        "Justification_LLM": [llm_row["Justification"]],
                        "Descriptors_GT": [gt_row["Descriptors"]],
                        "Descriptors_LLM": [llm_row["Descriptors"]],
                        "Justification_All_GT": [gt_row["Justification_All"]],
                        "Justification_All_LLM": [llm_row["Justification_All"]],
                        "Title_and_Justification_GT": [
                            gt_row["Title_and_Justification"]
                        ],
                        "Title_and_Justification_LLM": [
                            llm_row["Title_and_Justification"]
                        ],
                    }
                )
            else:
                # multiple candidates: pick all with the max score
                pairs = matching_cache[gt_idx]  # list of (llm_idx, score)
                llm_idx_candidates, scores = zip(*pairs)  # tuples
                scores = np.asarray(scores)
                max_score_tie = scores.max()
                max_mask = scores == max_score_tie

                # the set of LLM indices tied for max score
                max_llm_indices = [
                    idx for idx, keep in zip(llm_idx_candidates, max_mask) if keep
                ]
                # deterministic representative row: first of the ties
                rep_llm_idx = max_llm_indices[0]
                llm_row = llm_df.loc[rep_llm_idx]

                # merge all sentences
                merge_sentence = ".".join(
                    str(llm_sentences[i][1]) for i in llm_idx_candidates
                )

                # optional diagnostics if tied items disagree on metadata
                if len(max_llm_indices) > 1:
                    src_types = set(llm_df.loc[max_llm_indices, "SourceType"].tolist())
                    src_names = set(llm_df.loc[max_llm_indices, "Name"].tolist())
                    if len(src_types) > 1 or len(src_names) > 1:
                        for tile_idx in max_llm_indices:
                            llm_row_tie = llm_df.loc[tile_idx]
                            tile_row = pd.DataFrame(
                                {
                                    "GT_Sentence": [gt_sentence],
                                    "LLM_Sentence": [llm_sentences[tile_idx][1]],
                                    "Sentence_Match_Score": [float(max_score_tie)],
                                    "Source_Type_GT": [gt_row["SourceType"]],
                                    "Source_Type_LLM": [llm_row_tie["SourceType"]],
                                    "Name_GT": [gt_row["Name"]],
                                    "Name_LLM": [llm_row_tie["Name"]],
                                    "Title_GT": [gt_row["Title"]],
                                    "Title_LLM": [llm_row_tie["Title"]],
                                    "Justification_GT": [gt_row["Justification"]],
                                    "Justification_LLM": [llm_row_tie["Justification"]],
                                    "Descriptors_GT": [gt_row["Descriptors"]],
                                    "Descriptors_LLM": [llm_row_tie["Descriptors"]],
                                    "Justification_All_GT": [gt_row["Justification_All"]],
                                    "Justification_All_LLM": [llm_row_tie["Justification_All"]],
                                    "Title_and_Justification_GT": [
                                        gt_row["Title_and_Justification"]
                                    ],
                                    "Title_and_Justification_LLM": [
                                        llm_row_tie["Title_and_Justification"]
                                    ],
                                }
                            )
                            tie_df = pd.concat([tie_df, tile_row], ignore_index=True)

                new_row = pd.DataFrame(
                    {
                        "GT_Sentence": [gt_sentence],
                        "LLM_Sentence": [merge_sentence],
                        "Sentence_Match_Score": [float(max_score)],
                        "Source_Type_GT": [gt_row["SourceType"]],
                        "Source_Type_LLM": [llm_row["SourceType"]],
                        "Name_GT": [gt_row["Name"]],
                        "Name_LLM": [llm_row["Name"]],
                        "Title_GT": [gt_row["Title"]],
                        "Title_LLM": [llm_row["Title"]],
                        "Justification_GT": [gt_row["Justification"]],
                        "Justification_LLM": [llm_row["Justification"]],
                        "Descriptors_GT": [gt_row["Descriptors"]],
                        "Descriptors_LLM": [llm_row["Descriptors"]],
                        "Justification_All_GT": [gt_row["Justification_All"]],
                        "Justification_All_LLM": [llm_row["Justification_All"]],
                        "Title_and_Justification_GT": [
                            gt_row["Title_and_Justification"]
                        ],
                        "Title_and_Justification_LLM": [
                            llm_row["Title_and_Justification"]
                        ],
                    }
                )
        # Row positions of the pairing, for sj_eval (--sj-eval). Additive:
        # no metric reads these columns.
        new_row["GT_Row"] = gt_idx
        new_row["LLM_Rows"] = [sorted(i for i, _ in matching_cache.get(gt_idx, []))]
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    for llm_idx in final_unmatched_llm:
        llm_row = llm_df.loc[llm_idx]
        llm_sentence = llm_sentences[llm_idx][1]
        new_row = pd.DataFrame(
            {
                "GT_Sentence": [""],
                "LLM_Sentence": [llm_sentence],
                "Sentence_Match_Score": [0],
                "Source_Type_GT": [""],
                "Source_Type_LLM": [llm_row["SourceType"]],
                "Name_GT": [""],
                "Name_LLM": [llm_row["Name"]],
                "Title_GT": [""],
                "Title_LLM": [llm_row["Title"]],
                "Justification_GT": [""],
                "Justification_LLM": [llm_row["Justification"]],
                "Descriptors_GT": [""],
                "Descriptors_LLM": [llm_row["Descriptors"]],
                "Justification_All_GT": [[]],
                "Justification_All_LLM": [llm_row["Justification_All"]],
                "Title_and_Justification_GT": [""],
                "Title_and_Justification_LLM": [llm_row["Title_and_Justification"]],
            }
        )
        new_row["GT_Row"] = -1
        new_row["LLM_Rows"] = [[llm_idx]]
        result_df = pd.concat([result_df, new_row], ignore_index=True)

    # Calculate match scores and exact match for SourceType
    result_df["Source_Type_Match"] = result_df.apply(
        exact_match, axis=1, gt_col="Source_Type_GT", llm_col="Source_Type_LLM"
    )
    for attr in ["Name"]:
        result_df[f"{attr}_Match_Score"] = result_df.apply(
            fuzzy_compare, axis=1, gt_col=f"{attr}_GT", llm_col=f"{attr}_LLM"
        )

    for attr in ["Title", "Justification", "Title_and_Justification"]:
        result_df[f"{attr}_Match_Score"] = result_df.apply(
            semantic_compare,
            axis=1,
            gt_col=f"{attr}_GT",
            llm_col=f"{attr}_LLM",
        )
        # ...and again on the ruler the reported rates actually use. The column
        # above splits both texts at sentence/comma boundaries and keeps the
        # best pair; process_*_match compares the WHOLE strings, so the two
        # disagree and a dump could never be reconciled with a published
        # percentage. Additive: the split column is left alone because
        # Title_Match_Score and Title_and_Justification_Match_Score feed
        # Title_Match_Rate and All_Attribute_Accuracy.
        result_df[f"{attr}_Match_Score_Whole"] = result_df.apply(
            semantic_compare,
            axis=1,
            gt_col=f"{attr}_GT",
            llm_col=f"{attr}_LLM",
            using_split=False,
        )

    return result_df, tie_df


def process_title_match(row):
    """
    Define a function to process each row for title matching logic
    """

    def is_subset(a, b):
        return a in b or b in a

    if row["Title_Match_Score"] > 0.55:
        return True

    if row["Name_Match_Score"] > 80:
        if is_subset(row["Title_GT"], row["Title_LLM"]):
            return True

        # Construct strings with organization names and titles
        gt_full = row["Justification_GT"] + " " + row["Title_GT"]
        score = max(
            semantic_match(clean_text(row["Title_LLM"]), clean_text(row["Title_GT"])),
            semantic_match(clean_text(row["Title_LLM"]), clean_text(gt_full)),
        )
        return score > 0.55
    else:
        return False


def process_title_match_a(row):
    score = semantic_match(clean_text(row["Title_LLM"]), clean_text(row["Title_GT"]))
    return score > 0.55


def process_title_match_b(row):
    gt_full = row["Justification_GT"] + " " + row["Title_GT"]
    score = max(
        semantic_match(clean_text(row["Title_LLM"]), clean_text(row["Title_GT"])),
        semantic_match(clean_text(row["Title_LLM"]), clean_text(gt_full)),
    )
    return score > 0.55


JUSTIFICATION_THRESHOLD = 0.55


def _justification_lists(row):
    """The row's per-source justification lists, cleaned and de-blanked."""
    gt_list = [clean_text(v) for v in row["Justification_All_GT"]]
    llm_list = [clean_text(v) for v in row["Justification_All_LLM"]]
    return [v for v in gt_list if v], [v for v in llm_list if v]


def process_justification_match_a(row):
    """
    Rate A -- Justification_Match_Rate. Always True/False.

    Blank on both sides counts as CORRECT: an empty GT justification means the
    article carries no justification for that source, so a model that also
    leaves it empty has given the right answer. Legacy reaches the same verdict
    only by accident -- semantic_match("", "") == 1.0 clears the 0.55 bar --
    which would flip silently if the embedding model were ever swapped or an
    empty-string guard added. Best-match states it outright.
    """
    if justification.best_match():
        gt_list, llm_list = _justification_lists(row)
        if not gt_list and not llm_list:
            return True  # neither side claims a justification -- agreement
        if not gt_list or not llm_list:
            return False
        return (
            max(semantic_match(llm, gt) for gt in gt_list for llm in llm_list)
            > JUSTIFICATION_THRESHOLD
        )

    score = semantic_match(
        clean_text(row["Justification_LLM"]), clean_text(row["Justification_GT"])
    )
    return score > JUSTIFICATION_THRESHOLD


def process_justification_match_b(row):
    """
    Rate B -- Title_Justification_Join_Match_Rate. As A, but each GT
    justification also competes glued to the GT Title, forgiving the
    human/model disagreement about which column a role belongs in. Fixed in
    step with A so the A->B gap stays a column-placement measure rather than an
    artefact of the two rates using different rulers. Blank vs. blank counts as
    correct, as in A.
    """
    if justification.best_match():
        gt_list, llm_list = _justification_lists(row)
        title_gt = clean_text(row["Title_GT"])
        candidates = list(gt_list)
        if title_gt:
            candidates += [gt + " " + title_gt for gt in gt_list] or [title_gt]
        if not candidates and not llm_list:
            return True  # neither side claims a justification -- agreement
        if not candidates or not llm_list:
            return False
        return (
            max(semantic_match(llm, gt) for gt in candidates for llm in llm_list)
            > JUSTIFICATION_THRESHOLD
        )

    gt_full = row["Justification_GT"] + " " + row["Title_GT"]
    score = max(
        semantic_match(clean_text(row["Justification_LLM"]), clean_text(gt_full)),
        semantic_match(
            clean_text(row["Justification_LLM"]), clean_text(row["Justification_GT"])
        ),
    )
    return score > JUSTIFICATION_THRESHOLD


NULLISH_DESCRIPTORS = {"", "null", "`null`", "none", "n/a"}


def descriptor_set(value):
    """
    Normalise a Source Descriptors cell to a set of labels. Null-aware -- an
    empty GT cell and the literal word "null" a model writes both become the
    empty set -- and order-insensitive across comma-separated multi-values.
    """
    raw = clean_text(value)
    if raw in NULLISH_DESCRIPTORS:
        return set()
    parts = [clean_text(part).strip() for part in raw.split(",")]
    return {part for part in parts if part and part not in NULLISH_DESCRIPTORS}


def process_descriptor_match(row):
    """v59 Source Descriptors match: equal label sets counts as correct."""
    return descriptor_set(row["Descriptors_GT"]) == descriptor_set(row["Descriptors_LLM"])


# ---------------------------------------------------------------------------
# Lenient (soft) descriptor scoring -- ADDITIVE. The strict set-equality
# metrics above are unchanged; these add partial credit and near-miss credit.
#
# Two labels count as "the same" if any of these holds, checked in order:
#   1. identical after clean_text
#   2. one is a whole-word substring of the other   ("tenant advocates" / "advocates")
#   3. fuzz.ratio >= DESC_FUZZY_THRESHOLD           ("participant" / "participants")
#   4. MiniLM cosine >= DESC_SEMANTIC_THRESHOLD      ("guitarist" / "musician")
# Then GT labels and model labels are aligned one-to-one (greedy, best first)
# and a row scores recall = matched/|GT|, precision = matched/|model|, F1.
# Thresholds were calibrated on real GT/model word pairs; 0.65 (not the
# team's usual 0.55) keeps "wife"/"mother" (0.62) apart.
# ---------------------------------------------------------------------------
DESC_FUZZY_THRESHOLD = 80
DESC_SEMANTIC_THRESHOLD = 0.65


def _whole_word_contains(a, b):
    return re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b) is not None


def descriptor_pair_matches(a, b):
    """Do two single descriptor labels count as the same label under the lenient rule?"""
    a, b = clean_text(a), clean_text(b)
    if not a or not b:
        return False
    if a == b:
        return True
    if _whole_word_contains(a, b) or _whole_word_contains(b, a):
        return True
    if fuzz.ratio(a, b) >= DESC_FUZZY_THRESHOLD:
        return True
    return semantic_match(a, b) >= DESC_SEMANTIC_THRESHOLD


def _pair_order_score(a, b):
    """Cheap ordering score for the greedy alignment (no embeddings)."""
    a, b = clean_text(a), clean_text(b)
    if a == b or _whole_word_contains(a, b) or _whole_word_contains(b, a):
        return 1.0
    return fuzz.ratio(a, b) / 100.0


def soft_descriptor_scores(gt_value, llm_value):
    """
    Partial-credit descriptor score for one matched row.
    Returns (recall, precision, f1), or None when both sides are empty.
    recall is nan when GT is empty; precision is nan when the model wrote nothing.
    Labels are sorted before alignment so the result is deterministic.
    """
    gt_labels = sorted(descriptor_set(gt_value))
    llm_labels = sorted(descriptor_set(llm_value))
    if not gt_labels and not llm_labels:
        return None
    if not gt_labels:
        return (float("nan"), 0.0, 0.0)
    if not llm_labels:
        return (0.0, float("nan"), 0.0)
    candidates = sorted(
        ((_pair_order_score(g, l), g, l) for g in gt_labels for l in llm_labels),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    used_gt, used_llm, matched = set(), set(), 0
    for _, g, l in candidates:
        if g in used_gt or l in used_llm:
            continue
        if descriptor_pair_matches(g, l):
            used_gt.add(g)
            used_llm.add(l)
            matched += 1
    recall = matched / len(gt_labels)
    precision = matched / len(llm_labels)
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    return (recall, precision, f1)




def apply_row_rule(df, fn):
    """df.apply(fn, axis=1) that stays a bool Series when df is empty."""
    if df.empty:
        return pd.Series(False, index=df.index, dtype=bool)
    return df.apply(fn, axis=1)


def calculate_performance_metrics(result_df):
    global \
        all_gt_count_all, \
        both_found_count_all, \
        b_typeok_count_all, \
        b_typeok_nameok_count_all, \
        b_typenook_nameok_count_all, \
        b_typenook_namenook_count_all, \
        b_typeok_nameok_titleok_count_all, \
        b_typeok_namenook_titleok_count_all, \
        b_typenook_namenook_titleok_count_all, \
        b_typenook_nameok_titleok_count_all, \
        b_typeok_namenook_titlenook_count_all, \
        b_typenook_namenook_titlenook_count_all, \
        b_typenook_nameok_titlenook_count_all

    # Filter sentences found by both human and LLM
    both_found = result_df[
        (result_df["GT_Sentence"] != "") & (result_df["LLM_Sentence"] != "")
    ]

    both_found = both_found.copy()
    both_found["title_match"] = apply_row_rule(both_found, process_title_match)
    both_found_with_type_match = both_found[both_found["Source_Type_Match"] == "Yes"]

    # ##
    # This is to understand the basic justification matching rule
    # both_found["title_match_a"] = both_found.apply(process_title_match_a, axis=1)
    # both_found["title_match_b"] = both_found.apply(process_title_match_b, axis=1)
    both_found["justification_match_a"] = apply_row_rule(
        both_found, process_justification_match_a
    )
    both_found["justification_match_b"] = apply_row_rule(
        both_found, process_justification_match_b
    )

    both_found_count = len(both_found)  # New count for sentences found by both

    # Record metrics related data
    # both found => data both sentence matched
    # both found + type matched
    # both found + type_matched + name matched
    # both found + type_matched + name matched + title_justification_matched
    all_gt_count_all += len(result_df[(result_df["GT_Sentence"] != "")])
    both_found_count_all += both_found_count
    b_typeok_count_all += len(both_found[both_found["Source_Type_Match"] == "Yes"])
    b_typeok_nameok_count_all += len(
        both_found[
            (both_found["Name_Match_Score"] > 80)
            & (both_found["Source_Type_Match"] == "Yes")
        ]
    )
    b_typenook_nameok_count_all += len(
        both_found[
            (both_found["Name_Match_Score"] > 80)
            & (both_found["Source_Type_Match"] == "No")
        ]
    )
    b_typenook_namenook_count_all += len(
        both_found[
            (both_found["Name_Match_Score"] <= 80)
            & (both_found["Source_Type_Match"] == "No")
        ]
    )
    b_typeok_nameok_titleok_count_all += len(
        both_found[
            (both_found["Name_Match_Score"] > 80)
            & (both_found["Source_Type_Match"] == "Yes")
            & (both_found["Title_and_Justification_Match_Score"] > 0.55)
        ]
    )

    b_typeok_namenook_titleok_count_all += len(
        both_found[
            (both_found["Name_Match_Score"] < 80)
            & (both_found["Source_Type_Match"] == "Yes")
            & (both_found["Title_and_Justification_Match_Score"] > 0.55)
        ]
    )

    b_typenook_namenook_titleok_count_all += len(
        both_found[
            (both_found["Name_Match_Score"] < 80)
            & (both_found["Source_Type_Match"] == "No")
            & (both_found["Title_and_Justification_Match_Score"] > 0.55)
        ]
    )

    b_typenook_nameok_titleok_count_all += len(
        both_found[
            (both_found["Name_Match_Score"] > 80)
            & (both_found["Source_Type_Match"] == "No")
            & (both_found["Title_and_Justification_Match_Score"] > 0.55)
        ]
    )

    b_typeok_namenook_titlenook_count_all += len(
        both_found[
            (both_found["Name_Match_Score"] < 80)
            & (both_found["Source_Type_Match"] == "Yes")
            & (both_found["Title_and_Justification_Match_Score"] < 0.55)
        ]
    )

    b_typenook_namenook_titlenook_count_all += len(
        both_found[
            (both_found["Name_Match_Score"] < 80)
            & (both_found["Source_Type_Match"] == "No")
            & (both_found["Title_and_Justification_Match_Score"] < 0.55)
        ]
    )

    b_typenook_nameok_titlenook_count_all += len(
        both_found[
            (both_found["Name_Match_Score"] > 80)
            & (both_found["Source_Type_Match"] == "No")
            & (both_found["Title_and_Justification_Match_Score"] < 0.55)
        ]
    )

    # source type match
    source_type_match_rate = (both_found["Source_Type_Match"] == "Yes").mean()

    if run_mode == "debug":
        # Check for non-matching rows
        non_matched_rows = both_found[both_found["Source_Type_Match"] == "No"]

        # Add corresponding types to the error set
        for _, row in non_matched_rows.iterrows():
            error_matched[f"{row['Source_Type_GT']}_to_{row['Source_Type_LLM']}"] += 1

    # Calculate Match Rates
    # All gt with names
    # Positive:  match_score > 80 & Type is correct
    name_match_rate = (both_found["Name_Match_Score"] > 80).mean()
    type_name_match_rate = (
        (both_found["Name_Match_Score"] > 80)
        & (both_found["Source_Type_Match"] == "Yes")
    ).mean()
    type_correct_name_match_rate = (
        both_found_with_type_match["Name_Match_Score"] > 80
    ).mean()

    title_match_rate = both_found["title_match"].mean()
    both_found["descriptor_match"] = apply_row_rule(both_found, process_descriptor_match)
    descriptors_match_rate = both_found["descriptor_match"].mean()

    # The single rate above is dominated by rows where both sides are null, and
    # GT descriptor population is still incomplete for some source types (see
    # development-of-v59.md punchlist item 2). Split it so the two behaviours are
    # visible separately: accuracy where GT actually has a descriptor, and how
    # often a model fills the field where GT leaves it empty.
    gt_has_descriptor = both_found["Descriptors_GT"].apply(
        lambda v: bool(descriptor_set(v))
    )
    present = both_found[gt_has_descriptor]
    absent = both_found[~gt_has_descriptor]
    descriptors_recall_present = (
        present["descriptor_match"].mean() if len(present) else float("nan")
    )
    descriptors_added_absent = (
        absent["Descriptors_LLM"].apply(lambda v: bool(descriptor_set(v))).mean()
        if len(absent)
        else float("nan")
    )
    # Lenient / partial-credit descriptor scores (additive; see soft_descriptor_scores)
    soft_scores = [
        soft_descriptor_scores(row["Descriptors_GT"], row["Descriptors_LLM"])
        for _, row in both_found.iterrows()
    ]
    model_has_descriptor = both_found["Descriptors_LLM"].apply(
        lambda v: bool(descriptor_set(v))
    ).tolist()
    _present_soft = [
        s for s, g in zip(soft_scores, gt_has_descriptor.tolist()) if g and s is not None
    ]
    _model_soft = [
        s for s, m in zip(soft_scores, model_has_descriptor) if m and s is not None
    ]
    descriptors_soft_recall_present = (
        float(np.mean([s[0] for s in _present_soft])) if _present_soft else float("nan")
    )
    descriptors_soft_f1_present = (
        float(np.mean([s[2] for s in _present_soft])) if _present_soft else float("nan")
    )
    descriptors_soft_precision = (
        float(np.mean([s[1] for s in _model_soft])) if _model_soft else float("nan")
    )

    justification_match_rate = both_found["justification_match_a"].mean()
    title_justification_join_match_rate = both_found["justification_match_b"].mean()

    # The verdicts the rates actually counted, written back so --dump-comparisons
    # can be reconciled against the published percentage by hand. Unmatched rows
    # are never scored and stay blank here.
    result_df["Justification_Pass"] = both_found["justification_match_a"]
    result_df["Justification_Join_Pass"] = both_found["justification_match_b"]
    # Same for Title and Descriptors, which mismatch_log.py reads to list misses.
    result_df["Title_Pass"] = both_found["title_match"]
    result_df["Descriptors_Pass"] = both_found["descriptor_match"]

    # Calculate total unique sentences
    total_gt_sentences = len(result_df[result_df["GT_Sentence"] != ""])
    total_llm_sentences = len(result_df[result_df["LLM_Sentence"] != ""])
    total_unique_sentences = len(result_df[result_df["GT_Sentence"] != ""]) + len(
        result_df[(result_df["GT_Sentence"] == "") & (result_df["LLM_Sentence"] != "")]
    )

    match_counts = len(
        both_found[
            (both_found["Source_Type_Match"] == "Yes")
            & (
                (
                    (
                        (both_found["Source_Type_LLM"] == "Named_Person")
                        | (both_found["Source_Type_LLM"] == "Named_Organization")
                        | (both_found["Source_Type_LLM"] == "Documents")
                    )
                    & (both_found["Name_Match_Score"] > 80)
                )
                | (
                    (
                        (both_found["Title_GT"] != "")
                        & (both_found["Title_LLM"] != "")
                        & (both_found["Title_Match_Score"] > 0.55)
                    )
                    | (
                        (both_found["Justification_GT"] != "")
                        & (both_found["Justification_LLM"] != "")
                        & (both_found["Justification_Match_Score"] > 0.55)
                    )
                    | (
                        (both_found["Justification_GT"] == "")
                        & (both_found["Justification_LLM"] == "")
                    )
                )
            )
        ]
    )

    # LLM unique discover counts
    llm_unique_discover_counts = len(
        result_df[(result_df["GT_Sentence"] == "") & (result_df["LLM_Sentence"] != "")]
    )

    # Calculate LLM Recall
    source_statement_match_rate = (both_found_count) / total_gt_sentences

    # Calculate LLM Precision
    llm_statement_match_rate = (both_found_count) / total_llm_sentences

    # Calculate LLM statement match rate with type correct
    source_type_match_rate_with_statement = (
        len(both_found[both_found["Source_Type_Match"] == "Yes"]) / total_gt_sentences
    )

    # Calculate LLM unique discover Rate
    llm_unique_discover_rate = llm_unique_discover_counts / total_unique_sentences

    # Calculate LLM Miss Rate
    human_unique_discover_counts = len(
        result_df[(result_df["GT_Sentence"] != "") & (result_df["LLM_Sentence"] == "")]
    )
    llm_miss_rate = human_unique_discover_counts / total_gt_sentences

    all_attribute_accuracy = (
        len(
            both_found[
                (both_found["Name_Match_Score"] > 80)
                & (both_found["Source_Type_Match"] == "Yes")
                & (both_found["Title_and_Justification_Match_Score"] >= 0.55)
            ]
        )
        / total_gt_sentences
    )

    # Compile results
    metrics = {
        "Statement_Match_Rate": source_statement_match_rate,
        "LLMStatement_Match_Precison": llm_statement_match_rate,
        "Type_Match_Rate": source_type_match_rate,
        "Combined_Statement_Type_Match_Rate": source_type_match_rate_with_statement,
        "Name_Match_Rate": name_match_rate,
        "Type_Name_Match_Rate": type_name_match_rate,
        "Type_Correct_Name_Match_Rate": type_correct_name_match_rate,
        "Title_Match_Rate": title_match_rate,
        "Descriptors_Match_Rate": descriptors_match_rate,
        "Descriptors_Recall_When_Present": descriptors_recall_present,
        "Descriptors_Added_When_Absent": descriptors_added_absent,
        "Descriptors_GT_Present_Rows": int(gt_has_descriptor.sum()),
        "Descriptors_Soft_Recall_When_Present": descriptors_soft_recall_present,
        "Descriptors_Soft_F1_When_Present": descriptors_soft_f1_present,
        "Descriptors_Soft_Precision": descriptors_soft_precision,
        "Justification_Match_Rate": justification_match_rate,
        "Title_Justification_Join_Match_Rate": title_justification_join_match_rate,
        "All_Attribute_Accuracy": all_attribute_accuracy,
        "LLM_Miss_Rate": llm_miss_rate,
        "LLM_Unique_Discover_Rate": llm_unique_discover_rate,
        "Both_Found_Count": both_found_count,  # Added for debugging
        "Match_Counts": match_counts,  # Added for debugging
        "LLM_Unique_Discover_Counts": llm_unique_discover_counts,  # Added for debugging
        "Total_Unique_Sentences": total_unique_sentences,  # Added for debugging
    }

    return metrics


def plot_performance_comparison(performance_df, metric, output_dir):
    plt.figure(figsize=(12, 6))
    sns.barplot(x="article_w_version", y=metric, hue="model_name", data=performance_df)
    plt.title(f"{metric} Comparison")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric.lower().replace(' ', '_')}_comparison.png")
    plt.close()


def plot_overall_comparison(performance_df, output_dir, metrics):
    # Compute average performance
    avg_performance = performance_df.groupby("model_name")[metrics].mean().reset_index()

    # Melt the dataframe for plotting
    melted_df = pd.melt(avg_performance, id_vars=["model_name"], value_vars=metrics)

    # Wrap x-label text
    melted_df["model_names"] = melted_df["model_name"].apply(
        lambda x: textwrap.fill(x, 20)
    )

    # Plot combined metrics
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(
        x="model_names",
        y="value",
        hue="variable",
        data=melted_df,
    )

    # Add percentage annotations
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{100 * height:.1f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.title("Overall Performance Comparison")
    plt.xlabel("Model Name")
    plt.ylabel("Average Score")
    # plt.xticks(rotation=45)
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_performance_comparison.png")
    plt.close()

    # Individual metric plots
    for metric_name in metrics:
        metric_df = (
            performance_df.groupby("model_name")[metric_name].mean().reset_index()
        )
        metric_df["model_names"] = metric_df["model_name"].apply(
            lambda x: textwrap.fill(x, 20)
        )

        plt.figure(figsize=(16, 8))
        ax = sns.barplot(
            x="model_names",
            y=metric_name,
            data=metric_df,
        )

        # Add percentage annotations
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(
                f"{100 * height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
            )
        plt.title(f"Overall Performance {metric_name} Comparison")
        plt.xlabel("Model Name")
        plt.ylabel(f"Average {metric_name} Score")
        # plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overall_{metric_name}_performance_comparison.png")
        plt.close()


def process_one_article(gt_file, llm_files, output_dir, model_name):
    """
    one article id handler for target model
    """
    # Load and preprocess human data
    human_df = load_and_preprocess_ground_truth(gt_file)

    article_name = gt_file.split("/")[-1][:-4]
    one_article_res = {}
    all_metrics = []  # List to store all metrics for averaging

    for llm_file in llm_files:
        llm_version_id = llm_file[:-4].split("-")[-2]
        try:
            llm_df = load_and_preprocess_llm(llm_file)
        except:
            print(llm_file)
            continue
        try:
            llm_results, tie_df = compare_attributions(human_df, llm_df)
            if run_mode == "debug":
                llm_results.to_csv(
                    f"{output_dir}/{article_name}-{model_name}-{llm_version_id}-comparison.csv",
                    index=False,
                )
                if tie_df.shape[0] > 0:
                    tie_df.to_csv(
                        f"{output_dir}/{article_name}-{model_name}-{llm_version_id}-tie-samples.csv",
                        index=False,
                    )
                type_error_metrics_df = llm_results[
                    (llm_results["Source_Type_Match"] == "No")
                    & (llm_results["Source_Type_GT"] != "")
                ]
                type_error_metrics_df.to_csv(
                    f"{output_dir}/{article_name}-{model_name}-{llm_version_id}-type_miss_match.csv",
                    index=False,
                )
                llm_unique_metrics_df = llm_results[
                    (llm_results["GT_Sentence"] == "")
                    & (llm_results["LLM_Sentence"] != "")
                ]
                llm_unique_metrics_df.to_csv(
                    f"{output_dir}/{article_name}-{model_name}-{llm_version_id}-llm_unique.csv",
                    index=False,
                )
            llm_metrics = calculate_performance_metrics(llm_results)
            one_article_res[f"{article_name}_v{llm_version_id}"] = llm_metrics
            all_matched_global_df_list.append(llm_results)
            all_metrics.append(llm_metrics)
        except Exception as e:
            print(article_name, model_name, llm_version_id)
            print(e)

    # Convert the list of dictionaries to a DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Calculate the average of each column
    avg_metrics = metrics_df.mean()
    one_article_res[f"{article_name}_avg"] = avg_metrics.to_dict()

    return one_article_res


def process_articles_one_model(
    gt_files_by_id, llm_files_by_id, model_name, valid_article_ids, output_dir
):
    model_metrics = {}
    for article_id in valid_article_ids:
        model_metrics[article_id] = process_one_article(
            gt_files_by_id[article_id],
            llm_files_by_id[article_id],
            output_dir,
            model_name,
        )

    return model_metrics


def dict_to_df(metric_res):
    rows = []
    for _, metrics in metric_res.items():
        for version, values in metrics.items():
            row = {"article_w_version": version}
            row.update(values)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main_legacy_llm_results():
    """Original main(): walks llm_results/ per the hard-coded paths. Kept for reference; not used by the CLI."""
    human_gt_dir = "benchmarking/GT data/20250904/"
    llm_base_dirs = [
        # "llm_results/llms_20241211233801",
        # "llm_results/llms_20241212222327",
        # "llm_results/llms_20241215110531",
        # "llm_results/llms_20241215124335",
        # "llm_results/llms_20250209213013",
        "llm_results/all_llm_results_20250905102307/"
    ]
    output_dir = "benchmarking/metrics/09_05"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    total_version_num = 2

    model_names = [
        # "chatgpt-4o-latest",
        "gemini-pro-1.5",
        # "claude-3.5-sonnet",
        # "llama-3.1-70b-instruct",
        # "llama-3.1-405b-instruct",
        # "gemini-2.5-flash",
        # "deepseek-r1"
    ]

    llm_files_by_id = {v: defaultdict(list) for v in model_names}
    for llm_folder in llm_base_dirs:
        for model_name in model_names:
            for article_folder in glob.glob(f"{llm_folder}/{model_name}/*"):
                article_name = article_folder.split("/")[-1][:-4]
                article_id = article_name.split("-")[0]
                llm_files_by_id[model_name][article_id] += list(
                    glob.glob(f"{article_folder}/*.csv")
                )

    gt_files_by_id = {}
    for gt_file in glob.glob(f"{human_gt_dir}/*.csv"):
        article_name = gt_file.split("/")[-1][:-4]
        article_id = article_name.split("-")[0]
        gt_files_by_id[article_id] = gt_file

    valid_article_ids = set()
    for article_id in gt_files_by_id:
        is_valid = True
        for model_name in model_names:
            if (
                article_id not in llm_files_by_id[model_name]
                # or len(llm_files_by_id[model_name][article_id]) != total_version_num
                or len(llm_files_by_id[model_name][article_id]) == 0
            ):
                is_valid = False
                print(f"error valid num with {model_name} for the {article_id} ")
        if is_valid:
            valid_article_ids.add(article_id)

    valid_article_ids = set(["13", "16", "27", "31", "32"])
    # valid_article_ids = set(['4'])
    print("valid article ids is ", valid_article_ids)
    print("total valid article id number is ", len(valid_article_ids))

    combined_avg_metrics_dfs = []
    for model_name in model_names:
        model_metrics = process_articles_one_model(
            gt_files_by_id,
            llm_files_by_id[model_name],
            model_name,
            valid_article_ids,
            output_dir,
        )
        model_metrics_df = (
            dict_to_df(model_metrics)
            .sort_values(by="article_w_version", ascending=True)
            .reset_index(drop=True)
        )

        model_metrics_df.to_csv(f"{output_dir}/{model_name}_all_metrics.csv")

        avg_metrics_df = model_metrics_df[
            model_metrics_df["article_w_version"].str.contains("_avg")
        ]
        avg_metrics_df.to_csv(f"{output_dir}/{model_name}_avg_metrics.csv")

        avg_metrics_df = avg_metrics_df.copy()
        avg_metrics_df["model_name"] = model_name

        cols = avg_metrics_df.columns.tolist()
        cols = [cols[0]] + [cols[-1]] + cols[1:-1]
        avg_metrics_df = avg_metrics_df[cols]
        combined_avg_metrics_dfs.append(avg_metrics_df)

    # Combine all average metrics dataframes into one
    performance_df = pd.concat(combined_avg_metrics_dfs, ignore_index=True)
    performance_df.to_csv(f"{output_dir}/all_performance_metrics.csv", index=False)

    # Plot individual metric comparisons
    metrics = [
        "Statement_Match_Rate",
        "LLMStatement_Match_Precison",
        "Type_Match_Rate",
        "Combined_Statement_Type_Match_Rate",
        "Name_Match_Rate",
        "Type_Name_Match_Rate",
        "Title_Match_Rate",
        "Justification_Match_Rate",
        "Title_Justification_Join_Match_Rate",
        "All_Attribute_Accuracy",
    ]
    for metric in metrics:
        plot_performance_comparison(performance_df, metric, output_dir)

    # Plot overall comparison
    plot_overall_comparison(performance_df, output_dir, metrics)

    # Print summary statistics
    over_all_performance = performance_df.groupby("model_name")[metrics].agg(
        ["mean", "std"]
    )
    over_all_performance.to_csv(f"{output_dir}/over_all_performance_metrics.csv")
    print(over_all_performance)

    print(error_matched)

    # TODO: need to remove after debugging
    all_matched_global_df_combined = pd.concat(
        all_matched_global_df_list, ignore_index=True
    )
    all_matched_global_df_combined.to_csv("all_matched_global_df_combined.csv")


HEADLINE_METRICS = [
    ("Statement_Match_Rate", "Stmt Recall"),
    ("LLMStatement_Match_Precison", "Stmt Prec"),
    ("Type_Match_Rate", "Type"),
    ("Name_Match_Rate", "Name"),
    ("Title_Match_Rate", "Title"),
    ("Descriptors_Match_Rate", "Descr(all)"),
    ("Descriptors_Recall_When_Present", "Descr(present)"),
    ("Descriptors_Added_When_Absent", "Descr(added)"),
    ("Descriptors_Soft_Recall_When_Present", "Descr(soft)"),
    ("Descriptors_Soft_Precision", "Descr(softPrec)"),
    ("Justification_Match_Rate", "Justif"),
]

DEFAULT_PRED_REGEX = "story(?P<story>[0-9]+)_(?P<model>.+)_v59[.]csv$"


def score_one(gt_file, pred_file, schema, dump_dir=None, label=""):
    """Score one (ground truth, prediction) pair. Returns the metrics dict."""
    return score_one_frames(gt_file, pred_file, schema, dump_dir, label)[0]


def score_one_frames(gt_file, pred_file, schema, dump_dir=None, label=""):
    """score_one, also returning (gt_df, llm_df, result_df) for --sj-eval / --mismatch-log."""
    gt_df = load_and_preprocess_ground_truth(gt_file)
    llm_df = load_predictions(pred_file, schema)
    result_df, tie_df = compare_attributions(gt_df, llm_df)
    # Score first: calculate_performance_metrics writes the Justification_Pass /
    # Justification_Join_Pass verdict columns back onto result_df, and the dump
    # is only reconcilable with the summary if it carries them.
    metrics = calculate_performance_metrics(result_df)
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        result_df.to_csv(os.path.join(dump_dir, label + "-comparison.csv"), index=False)
        if len(tie_df):
            tie_df.to_csv(
                os.path.join(dump_dir, label + "-tie-samples.csv"), index=False
            )
    return metrics, gt_df, llm_df, result_df


def build_parser():
    parser = argparse.ArgumentParser(
        description="Score v59 model output against human ground truth."
    )
    parser.add_argument(
        "--gt-dir",
        default="benchmarking/GT data/GT-2026",
        help="folder of ground-truth CSVs named {story}-{slug}.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--pred-glob",
        required=True,
        help="glob for prediction CSVs, e.g. "
        '"../rai-scu-kite/sourcing_data/v59_gt_benchmark/story*_v59.csv"',
    )
    parser.add_argument(
        "--pred-regex",
        default=DEFAULT_PRED_REGEX,
        help="regex with named groups `story` and `model`, applied to each "
        "prediction filename (default: %(default)s)",
    )
    parser.add_argument("--out-dir", default="benchmarking/metrics/v59")
    parser.add_argument("--schema", choices=["v59", "v57"], default="v59")
    parser.add_argument(
        "--stories", nargs="+", type=int, help="only these story numbers"
    )
    parser.add_argument("--models", nargs="+", help="only these model tags")
    parser.add_argument(
        "--match-method",
        choices=sorted(eval_config.match_threshold),
        help="override configure.py's match_method",
    )
    parser.add_argument("--threshold", type=float, help="override the match threshold")
    parser.add_argument("--desc-fuzzy", type=int,
                        help="lenient descriptor fuzz.ratio threshold (default %d)" % DESC_FUZZY_THRESHOLD)
    parser.add_argument("--desc-semantic", type=float,
                        help="lenient descriptor semantic threshold (default %.2f)" % DESC_SEMANTIC_THRESHOLD)
    parser.add_argument(
        "--apply-fixes",
        action="store_true",
        help="turn ON the optional matching fixes (exclusive claims, pair on "
        "name, round-2 tuple fix, bidirectional containment). Default is OFF so "
        "numbers stay comparable to eval_multi_llm_article.py.",
    )
    parser.add_argument(
        "--justification-mode",
        choices=list(justification.CHOICES),
        default=justification.LEGACY,
        help="how Source Justification is graded. `legacy` (default) glues "
        "every justification sharing a Name into one blob before comparing, "
        "reproducing the historical numbers. `best-match` keeps them as a "
        "list and scores the model's answer against each of the human's "
        "separately, keeping the best. Blank-vs-blank counts as correct in "
        "both. See SJ_Review_Proposal.md.",
    )
    parser.add_argument(
        "--dump-comparisons",
        action="store_true",
        help="also write the per-pair comparison and tie-sample CSVs",
    )
    parser.add_argument(
        "--sj-eval",
        action="store_true",
        help="also score Source Justification three ways (sj_eval package): "
        "SJ_Source_Score, SJ_Placement_Score, SJ_Concat_Score. Additive -- no "
        "existing metric changes. Meant for v60sj output (each SJ on its own row).",
    )
    parser.add_argument(
        "--sj-margin",
        type=int,
        default=1,
        help="named sources: statements of slack before an SJ counts as "
        "misplaced (default %(default)s)",
    )
    parser.add_argument(
        "--mismatch-log",
        action="store_true",
        help="also write mismatches.xlsx: one tab per score (statements, Type, "
        "Name, Title, Descriptors, old Justification, and the SJ scores when "
        "--sj-eval is on), one row per miss",
    )
    parser.add_argument("--tag", default="", help="suffix for the output filenames")
    return parser


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()

    if args.apply_fixes:
        fixes.enable_all()
    justification.mode = args.justification_mode
    if args.match_method:
        eval_config.match_method = args.match_method
    if args.threshold is not None:
        eval_config.match_threshold[eval_config.match_method] = args.threshold
    global DESC_FUZZY_THRESHOLD, DESC_SEMANTIC_THRESHOLD
    if args.desc_fuzzy is not None:
        DESC_FUZZY_THRESHOLD = args.desc_fuzzy
    if args.desc_semantic is not None:
        DESC_SEMANTIC_THRESHOLD = args.desc_semantic

    pattern = re.compile(args.pred_regex)
    preds = []
    for path in sorted(glob.glob(args.pred_glob)):
        found = pattern.search(os.path.basename(path))
        if not found:
            print("  skip (does not match --pred-regex): " + path)
            continue
        story, model = int(found.group("story")), found.group("model")
        if args.stories and story not in args.stories:
            continue
        if args.models and model not in args.models:
            continue
        preds.append((story, model, path))

    if not preds:
        sys.exit("No prediction files matched --pred-glob / filters.")

    method = eval_config.match_method
    print("schema=%s  match=%s > %s" % (args.schema, method, eval_config.match_threshold[method]))
    print("fixes: " + fixes.describe())
    print("justification: " + justification.describe())
    print("scoring %d prediction files against %s" % (len(preds), args.gt_dir))
    print()

    dump_dir = (
        os.path.join(args.out_dir, "comparisons") if args.dump_comparisons else None
    )
    rows, sj_records, comparisons = [], [], []
    for story, model, path in preds:
        hits = glob.glob(os.path.join(args.gt_dir, "%d-*.csv" % story))
        if len(hits) != 1:
            print("  story %d: expected 1 ground-truth file, found %d -- skipped"
                  % (story, len(hits)))
            continue
        label = "story%02d-%s" % (story, model)
        metrics, gt_df, llm_df, result_df = score_one_frames(
            hits[0], path, args.schema, dump_dir, label
        )
        row = {"story": story, "model": model}
        row.update({key: metrics[key] for key, _ in HEADLINE_METRICS})
        row["Descr_GT_Present_Rows"] = metrics["Descriptors_GT_Present_Rows"]
        if args.sj_eval:
            records = sj_eval.score_story(gt_df, llm_df, result_df, story, model,
                                          semantic_match, margin=args.sj_margin)
            row.update(sj_eval.score_columns(records))
            sj_records += records
        if args.mismatch_log:
            comparisons.append(result_df.assign(Story=story, Model=model))
        rows.append(row)
        print("  scored story %2d / %s" % (story, model))

    df = pd.DataFrame(rows).sort_values(["model", "story"])
    os.makedirs(args.out_dir, exist_ok=True)
    if args.tag:
        suffix = "_" + args.tag
    elif args.apply_fixes:
        suffix = "_fixed"
    else:
        suffix = ""
    per_story = os.path.join(args.out_dir, "v59_eval_per_story%s.csv" % suffix)
    summary_path = os.path.join(args.out_dir, "v59_eval_summary%s.csv" % suffix)
    df.to_csv(per_story, index=False)

    keys = [key for key, _ in HEADLINE_METRICS]
    agg = df.groupby("model")[keys].mean()
    agg.insert(0, "stories", df.groupby("model")["story"].count())
    if args.sj_eval:
        # Pooled over every SJ of the model, not a mean of per-story scores.
        agg = agg.join(sj_eval.model_summary(sj_records))
        sj_detail_path = os.path.join(args.out_dir, "sj_eval_detail%s.csv" % suffix)
        sj_eval.records_frame(sj_records).to_csv(
            sj_detail_path, index=False, encoding="utf-8-sig"
        )
    agg.to_csv(summary_path)

    print()
    print("Per-model means")
    header = "%-16s%4s" % ("model", "n") + "".join(
        "%16s" % label for _, label in HEADLINE_METRICS
    )
    print(header)
    print("-" * len(header))
    for model, row in agg.iterrows():
        cells = "".join(
            "             n/a"
            if pd.isna(row[key])
            else "%15.1f%%" % (row[key] * 100)
            for key, _ in HEADLINE_METRICS
        )
        print("%-16s%4d%s" % (model, int(row["stories"]), cells))
    if args.sj_eval:
        print()
        print("Source Justification (pooled; n = GT SJs scored)")
        for model, row in agg.iterrows():
            cells = "  ".join(
                "%s %s (n=%d)" % (
                    column.replace("SJ_", "").replace("_Score", ""),
                    "n/a" if pd.isna(row[column]) else "%.3f" % row[column],
                    row[sj_eval.count_column(column)],
                )
                for column in sj_eval.SCORE_COLUMNS
            )
            print("  %-16s%s" % (model, cells))
    print()
    print("Wrote:")
    print("  " + per_story)
    print("  " + summary_path)
    if args.sj_eval:
        print("  " + sj_detail_path)
    if args.mismatch_log:
        workbook = os.path.join(args.out_dir, "mismatches%s.xlsx" % suffix)
        counts = mismatch_log.write_workbook(
            workbook,
            pd.concat(comparisons, ignore_index=True),
            sj_eval.records_frame(sj_records) if args.sj_eval else None,
            agg,
        )
        print("  " + workbook)
        print("    misses per tab: " + ", ".join("%s %d" % kv for kv in counts.items()))
    if dump_dir:
        print("  " + dump_dir + os.sep)


if __name__ == "__main__":
    main()
