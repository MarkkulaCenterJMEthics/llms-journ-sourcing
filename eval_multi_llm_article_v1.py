import glob
import logging
import re
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

run_mode = "debug"  # run or develop
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
    "anonymous_source": "Anonymous_Source",
    "unnamed people": "Anonymous_Source",
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


def semantic_match(sentence_one, sentence_two):
    embedding_1 = embedding_model.encode(sentence_one, convert_to_tensor=True)
    embedding_2 = embedding_model.encode(sentence_two, convert_to_tensor=True)

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
            df.at[index, target_column] = ",".join(list(name_set[name]))

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

    df["SourceType"] = df["SourceType"].str.lower().map(map_with_error)

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
    # Define the columns we want to keep and their corresponding indices
    columns_to_keep = {
        "Sourced Statements ": "SourcedStatement",  # 'Actual text in the article: attributed info (views) or quotes from a source'
        "Type of source": "SourceType",  # 'Type of source'
        "Name of Source": "Name",  # 'Name of Source'
        "Title of Source": "Title",  # 'Source\'s Title (affiliation)'
        "Source Justification": "Justification",  # 'Additional source characterizations in introduction justifying presence in the article'
    }

    # Read the CSV file, specifying the columns we want by index
    try:
        gt_df = pd.read_csv(
            file_path, skiprows=5, usecols=columns_to_keep.keys(), header=None
        )
    except Exception:
        gt_df = pd.read_csv(file_path, usecols=columns_to_keep.keys())

    # Rename the columns
    gt_df.rename(columns=columns_to_keep, inplace=True)

    # Preprocess the data
    gt_df = preprocess_dataframe(gt_df, "SourcedStatement")

    return gt_df


def load_and_preprocess_llm(file_path):
    """
    process llm csv file.
    """
    columns_to_keep = {
        "Name_of_Source": "Name of Source"
    }
    try:
        llm_df = pd.read_csv(file_path, encoding="ISO-8859-1")
    except:
        logging.info(f" read llm csv {file_path} with error.")
        llm_df = pd.read_csv(file_path)

    llm_df.rename(columns=columns_to_keep, inplace=True)
    # Preprocess the data
    llm_df = preprocess_dataframe(llm_df, "SourcedStatement")

    return llm_df


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

    # Find best matches for GT sentences
    for gt_idx, gt_sentence in gt_sentences:
        if len(llm_sentences) == 0:
            score = 0
        else:
            best_match, score = find_best_match(
                gt_sentence, [s for _, s in llm_sentences], eval_config.match_method
            )

        if (
            score > eval_config.match_threshold[eval_config.match_method]
        ):  # We use the threshold for determining a "good" match
            llm_idx = next(idx for idx, s in llm_sentences if s == best_match)
            gt_row = gt_df.loc[gt_idx]
            llm_row = llm_df.loc[llm_idx]
            new_row = pd.DataFrame(
                {
                    "GT_Sentence": [gt_sentence],
                    "LLM_Sentence": [best_match],
                    "Sentence_Match_Score": [score],
                    "Source_Type_GT": [gt_row["SourceType"]],
                    "Source_Type_LLM": [llm_row["SourceType"]],
                    "Name_GT": [gt_row["Name"]],
                    "Name_LLM": [llm_row["Name"]],
                    "Title_GT": [gt_row["Title"]],
                    "Title_LLM": [llm_row["Title"]],
                    "Justification_GT": [gt_row["Justification"]],
                    "Justification_LLM": [llm_row["Justification"]],
                    "Title_and_Justification_GT": [gt_row["Title_and_Justification"]],
                    "Title_and_Justification_LLM": [llm_row["Title_and_Justification"]],
                }
            )
        else:
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
                    "Title_and_Justification_GT": [
                        gt_df.loc[gt_idx, "Title_and_Justification"]
                    ],
                    "Title_and_Justification_LLM": [""],
                }
            )

        result_df = pd.concat([result_df, new_row], ignore_index=True)

    # Find LLM sentences not matched to any GT sentence
    matched_llm_sentences = result_df["LLM_Sentence"].tolist()
    unmatched_llm = [
        (idx, s) for idx, s in llm_sentences if s not in matched_llm_sentences
    ]

    for llm_idx, llm_sentence in unmatched_llm:
        llm_row = llm_df.loc[llm_idx]
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
                "Title_and_Justification_GT": [""],
                "Title_and_Justification_LLM": [llm_row["Title_and_Justification"]],
            }
        )
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
            semantic_compare, axis=1, gt_col=f"{attr}_GT", llm_col=f"{attr}_LLM",
        )

    return result_df


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


def process_justification_match_a(row):
    score = semantic_match(
        clean_text(row["Justification_LLM"]), clean_text(row["Justification_GT"])
    )
    return score > 0.55


def process_justification_match_b(row):
    gt_full = row["Justification_GT"] + " " + row["Title_GT"]
    score = max(
        semantic_match(clean_text(row["Justification_LLM"]), clean_text(gt_full)),
        semantic_match(
            clean_text(row["Justification_LLM"]), clean_text(row["Justification_GT"])
        ),
    )
    return score > 0.55


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
    both_found["title_match"] = both_found.apply(process_title_match, axis=1)
    both_found_with_type_match = both_found[both_found["Source_Type_Match"] == "Yes"]

    # ##
    # This is to understand the basic justification matching rule
    # both_found["title_match_a"] = both_found.apply(process_title_match_a, axis=1)
    # both_found["title_match_b"] = both_found.apply(process_title_match_b, axis=1)
    both_found["justification_match_a"] = both_found.apply(
        process_justification_match_a, axis=1
    )
    both_found["justification_match_b"] = both_found.apply(
        process_justification_match_b, axis=1
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

    # print(
    #     f"All GT Count: {all_gt_count_all}\n"
    #     f"Both Found Count: {both_found_count_all}\n"
    #     f"B Type OK Count: {b_typeok_count_all}\n"
    #     f"B Type OK & Name OK Count: {b_typeok_nameok_count_all}\n"
    #     f"B Type NOT OK & Name OK Count: {b_typenook_nameok_count_all}\n"
    #     f"B Type NOT OK & Name NOT OK Count: {b_typenook_namenook_count_all}\n"
    #     f"B Type OK, Name OK & Title OK Count: {b_typeok_nameok_titleok_count_all}\n"
    #     f"B Type OK, Name NOT OK & Title OK Count: {b_typeok_namenook_titleok_count_all}\n"
    #     f"B Type NOT OK, Name NOT OK & Title OK Count: {b_typenook_namenook_titleok_count_all}\n"
    #     f"B Type NOT OK, Name OK & Title OK Count: {b_typenook_nameok_titleok_count_all}\n"
    #     f"B Type OK, Name NOT OK & Title NOT OK Count: {b_typeok_namenook_titlenook_count_all}\n"
    #     f"B Type NOT OK, Name NOT OK & Title NOT OK Count: {b_typenook_namenook_titlenook_count_all}\n"
    #     f"B Type NOT OK, Name OK & Title NOT OK Count: {b_typenook_nameok_titlenook_count_all}"
    # )

    # source type match
    source_type_match_rate = (both_found["Source_Type_Match"] == "Yes").mean()

    if run_mode == "debug":
        # Check for non-matching rows
        non_matched_rows = both_found[both_found["Source_Type_Match"] == "No"]

        # Add corresponding types to the error set
        for _, row in non_matched_rows.iterrows():
            error_matched[f'{row["Source_Type_GT"]}_to_{row["Source_Type_LLM"]}'] += 1

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
    justification_match_rate = both_found["justification_match_a"].mean()
    title_justification_join_match_rate = both_found["justification_match_b"].mean()
    
    # Calculate total unique sentences
    total_gt_sentences = len(result_df[result_df["GT_Sentence"] != ""])
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

    # Calculate LLM statement match rate with type correct
    source_type_match_rate_with_statement = len(both_found[both_found["Source_Type_Match"] == "Yes"]) /  total_gt_sentences

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
        "Type_Match_Rate": source_type_match_rate,
        "Combined_Statement_Type_Match_Rate": source_type_match_rate_with_statement,
        "Name_Match_Rate": name_match_rate,
        "Type_Name_Match_Rate": type_name_match_rate,
        "Type_Correct_Name_Match_Rate": type_correct_name_match_rate,
        "Title_Match_Rate": title_match_rate,
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
    
    # avg_performance = performance_df.groupby("model_name")[metrics].mean().reset_index()

    # plt.figure(figsize=(16, 8))
    # sns.barplot(
    #     x="model_name",
    #     y="value",
    #     hue="variable",
    #     data=pd.melt(avg_performance, id_vars=["model_name"], value_vars=metrics),
    # )
    # plt.title("Overall Performance Comparison")
    # plt.ylabel("Average Score")
    # plt.xticks(rotation=45)
    # plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    # plt.tight_layout()
    # plt.savefig(f"{output_dir}/overall_performance_comparison.png")
    # plt.close()

    # avg_performance = performance_df.groupby("model_name")[metrics].mean().reset_index()

    # for metric_name in metrics:
    #     plt.figure(figsize=(16, 8))
    #     sns.barplot(
    #         x="model_name",
    #         y="value",
    #         hue="variable",
    #         data=pd.melt(
    #             performance_df.groupby("model_name")[metric_name].mean().reset_index(),
    #             id_vars=["model_name"],
    #             value_vars=[metric_name],
    #         ),
    #     )
    #     plt.title(f"Overall Performance {metric_name} Comparison")
    #     plt.ylabel("Average Score")
    #     plt.xticks(rotation=45)
    #     plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    #     plt.tight_layout()
    #     plt.savefig(f"{output_dir}/overall_{metric_name}_performance_comparison.png")
    #     plt.close()

    # Compute average performance
    avg_performance = performance_df.groupby("model_name")[metrics].mean().reset_index()

    # Melt the dataframe for plotting
    melted_df = pd.melt(avg_performance, id_vars=["model_name"], value_vars=metrics)

    # Wrap x-label text
    melted_df['model_names'] = melted_df['model_name'].apply(lambda x: textwrap.fill(x, 20))

    # Plot combined metrics
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
        ax.annotate(f'{100*height:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10)
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
        metric_df = performance_df.groupby("model_name")[metric_name].mean().reset_index()
        metric_df['model_names'] = metric_df['model_name'].apply(lambda x: textwrap.fill(x, 20))

        plt.figure(figsize=(16, 8))
        ax = sns.barplot(
            x="model_names",
            y=metric_name,
            data=metric_df,
        )

        # Add percentage annotations
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{100*height:.1f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10)
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
            llm_results = compare_attributions(human_df, llm_df)
            if run_mode == "debug":
                llm_results.to_csv(
                    f"{output_dir}/{article_name}-{model_name}-{llm_version_id}-comparison.csv",
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


def main():
    # default parameter
    # TODO: move to config file
    human_gt_dir = "benchmarking/GT data/"
    llm_base_dirs = [
        "llm_results/llms_20241211233801",
        "llm_results/llms_20241212222327",
        "llm_results/llms_20241215110531",
        "llm_results/llms_20241215124335",
        "llm_results/llms_20250209213013",
        #"llm_results/llms_20250417204917"
        # "llm_results/llms_20250504210825"
    ]
    output_dir = "benchmarking/metrics/05_7"
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


if __name__ == "__main__":
    main()

    # def load_and_preprocess_ground_truth(file_path):
    # Define the columns we want to keep and their corresponding indices
    # human_df = load_and_preprocess_ground_truth("/home/shawnyang/Documents/Project/llms-journ-sourcing/benchmarking/GT data/2-Best-DDR-player.csv")
    # llm_df = load_and_preprocess_llm("llm_results/llms_20250504210825/gemini-2.5-flash/10-Nebraska_lawmakers_bill/10-Nebraska_lawmakers_bill-gemini-2.5-flash-v4-May05.csv")
    # print(llm_df['Name'])
    # llm_results = compare_attributions(human_df, llm_df)