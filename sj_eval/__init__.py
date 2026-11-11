"""
sj_eval -- the three Source Justification scores, run from
eval_multi_llm_article_v2.py with --sj-eval.

    SJ_Source_Score     right SJ anywhere for the source   (Named Person/Org)
    SJ_Placement_Score  right SJ on the right row          (named: +/-1 of the
                        source's own statements; unnamed: same row)
    SJ_Concat_Score     share of a concatenated GT SJ's parts the model found
                        (only GT SJs with 2+ parts)

Every scorer returns Records (common.Record). A score is the mean of its
records' `score`, so the per-story value is (sum of matched similarities) /
(GT SJ count), and the per-model value pools every record of that model --
a story with 10 SJs weighs more than a story with 1.
"""
from dataclasses import asdict

import pandas as pd

from . import concat, placement, source
from .common import MARGIN, THRESHOLD, Record, StoryContext
from .linking import named_groups, unnamed_rows

# (summary column, metric, bucket or None for both)
COLUMNS = [
    ("SJ_Source_Score", source.METRIC, None),
    ("SJ_Placement_Score", placement.METRIC, None),
    ("SJ_Placement_Named", placement.METRIC, "named"),
    ("SJ_Placement_Unnamed", placement.METRIC, "unnamed"),
    ("SJ_Concat_Score", concat.METRIC, None),
]
SCORE_COLUMNS = [c for c, _, _ in COLUMNS]


def count_column(column: str) -> str:
    """SJ_Source_Score -> SJ_Source_N: how many GT SJs the score is over."""
    return column.replace("_Score", "") + "_N"


def score_story(gt_df, llm_df, result_df, story, model, score_fn,
                threshold=THRESHOLD, margin=MARGIN) -> list[Record]:
    """All SJ records for one (story, model). Frames are the evaluator's."""
    ctx = StoryContext(story, model, score_fn, gt_df, llm_df, threshold, margin)
    groups = named_groups(gt_df, llm_df, result_df)
    rows = unnamed_rows(gt_df, llm_df, result_df)
    return (
        source.score(groups, ctx)
        + placement.score_named(groups, ctx)
        + placement.score_unnamed(rows, ctx)
        + concat.score_named(groups, ctx)
        + concat.score_unnamed(rows, ctx)
    )


def records_frame(records) -> pd.DataFrame:
    columns = list(Record.__dataclass_fields__)
    return pd.DataFrame([asdict(r) for r in records], columns=columns)


def score_columns(records) -> dict:
    """{column: mean score, count_column: record count}; mean is NaN when N = 0."""
    out = {}
    for column, metric, bucket in COLUMNS:
        scores = [r.score for r in records
                  if r.metric == metric and (bucket is None or r.bucket == bucket)]
        out[column] = sum(scores) / len(scores) if scores else float("nan")
        out[count_column(column)] = len(scores)
    return out


def model_summary(records) -> pd.DataFrame:
    """One row per model, every record of that model pooled."""
    by_model = {}
    for r in records:
        by_model.setdefault(r.model, []).append(r)
    return pd.DataFrame.from_dict(
        {m: score_columns(rs) for m, rs in by_model.items()}, orient="index"
    )
