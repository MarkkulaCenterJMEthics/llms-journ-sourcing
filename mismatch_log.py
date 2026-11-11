"""
mismatch_log.py -- writes the mismatch workbook for eval_multi_llm_article_v2.py
--mismatch-log: one tab per evaluation score, one row per miss.

Field tabs (Type, Name, Title, Descriptors, Justification_Old) read the
verdicts the evaluator itself recorded on result_df, so a row is in a tab
exactly when that row counted as wrong in the published rate. SJ tabs are the
sj_eval Records with passed=False.

Every tab has the same leading columns -- Story, Model, GT_Row, LLM_Rows,
GT_Statement, Model_Statement -- then the GT value, model value, score and the
reason, so the workbook can be filtered by story or reason in Excel.
"""
import pandas as pd

LEAD = ["Story", "Model", "GT_Row", "LLM_Rows", "GT_Statement", "Model_Statement"]
TAIL = ["Source_Type_GT", "Name_GT", "GT_Value", "Model_Value", "Score", "Reason"]
NAME_BAR = 80
SJ_METRICS = ["SJ_Source", "SJ_Placement", "SJ_Concat"]


def _join_rows(rows):
    return " ".join(str(r) for r in rows) if isinstance(rows, (list, tuple)) else rows


def _rows(df, gt_col, llm_col, score_col, reason):
    """Shape a slice of the comparisons frame into the common tab layout."""
    out = pd.DataFrame({
        "Story": df["Story"], "Model": df["Model"],
        "GT_Row": df["GT_Row"], "LLM_Rows": df["LLM_Rows"].apply(_join_rows),
        "GT_Statement": df["GT_Sentence"], "Model_Statement": df["LLM_Sentence"],
        "Source_Type_GT": df["Source_Type_GT"], "Name_GT": df["Name_GT"],
        "GT_Value": df[gt_col] if gt_col else "",
        "Model_Value": df[llm_col] if llm_col else "",
        "Score": df[score_col] if score_col else "",
        "Reason": reason,
    })
    return out[LEAD + TAIL]


def _failed(df, verdict_column):
    """Rows whose recorded verdict is False (blank = not scored, never a miss)."""
    return df[df[verdict_column].eq(False)]


def field_tabs(comparisons: pd.DataFrame) -> dict:
    """{tab name: DataFrame} for statement and field misses."""
    gt_has = comparisons["GT_Sentence"] != ""
    llm_has = comparisons["LLM_Sentence"] != ""
    matched = comparisons[gt_has & llm_has]
    return {
        "Statements_Missed": _rows(comparisons[gt_has & ~llm_has], None, None, None,
                                   "GT statement not found by model"),
        "Statements_Extra": _rows(comparisons[~gt_has & llm_has], None, "Name_LLM", None,
                                  "model statement not in GT"),
        "Type": _rows(matched[matched["Source_Type_Match"] == "No"],
                      "Source_Type_GT", "Source_Type_LLM", None, "type differs"),
        "Name": _rows(matched[matched["Name_Match_Score"] <= NAME_BAR],
                      "Name_GT", "Name_LLM", "Name_Match_Score",
                      "name fuzzy score <= %d" % NAME_BAR),
        "Title": _rows(_failed(matched, "Title_Pass"), "Title_GT", "Title_LLM",
                       "Title_Match_Score_Whole", "title rule failed"),
        "Descriptors": _rows(_failed(matched, "Descriptors_Pass"), "Descriptors_GT",
                             "Descriptors_LLM", None, "descriptor sets differ"),
        "Justification_Old": _rows(_failed(matched, "Justification_Pass"),
                                   "Justification_GT", "Justification_LLM",
                                   "Justification_Match_Score_Whole",
                                   "old per-row justification rule failed"),
    }


def sj_tabs(records: pd.DataFrame) -> dict:
    """{tab name: DataFrame} for the SJ scores, from sj_eval.records_frame()."""
    tabs = {}
    for metric in SJ_METRICS:
        miss = records[(records["metric"] == metric) & ~records["passed"].astype(bool)]
        tabs[metric] = pd.DataFrame({
            "Story": miss["story"], "Model": miss["model"],
            "GT_Row": miss["gt_rows"].apply(_join_rows),
            "LLM_Rows": miss["model_rows"].apply(_join_rows),
            "GT_Statement": miss["gt_statement"], "Model_Statement": miss["model_statement"],
            "Bucket": miss["bucket"],
            "Source_Type_GT": miss["source_type"], "Name_GT": miss["source_name"],
            "GT_Value": miss["gt_value"], "Model_Value": miss["model_value"],
            "Score": miss["score"], "Reason": miss["reason"],
        })
    return tabs


def write_workbook(path, comparisons, records, summary) -> dict:
    """Write every tab (header frozen, autofilter on). Returns {tab: miss count}."""
    tabs = field_tabs(comparisons)
    if records is not None:
        tabs.update(sj_tabs(records))
    tabs["Summary"] = summary.reset_index()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tabs.items():
            df.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
    return {name: len(df) for name, df in tabs.items() if name != "Summary"}
