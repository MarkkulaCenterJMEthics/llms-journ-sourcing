"""
SJ_Concat_Score (point 4, lenient): for a GT SJ made of 2+ parts (joined by
';' or '...'), what share of its parts did the model find?

Only concatenated GT SJs are scored; a story with none reports a blank score,
not 0. Coverage is proportional (2 of 3 parts = 0.67). A part counts as found
when it matches a model part above the threshold, one-to-one within that SJ.

* Named sources: each distinct concatenated GT SJ once per source, compared
  with all of the source's model parts (location ignored -- placement is
  SJ_Placement's job).
* Every other source: row by row, against the model rows paired to that row.
"""
from collections import defaultdict

from .common import SJItem, StoryContext, assign

METRIC = "SJ_Concat"


def coverage_record(ctx, parts, model_texts, bucket, source_type, source_name,
                    gt_rows, model_rows):
    gt_items = [SJItem(t, frozenset({0})) for t in dict.fromkeys(parts)]
    model_items = [SJItem(t, frozenset({0})) for t in dict.fromkeys(model_texts)]
    chosen = assign(gt_items, model_items, ctx.score_fn, ctx.threshold)
    missed = [g.text for g in gt_items if g not in chosen]
    reason = "found %d of %d parts" % (len(chosen), len(gt_items))
    if missed:
        reason += "; missed: " + " | ".join(missed)
    return ctx.record(METRIC, bucket, source_type, source_name, gt_rows, model_rows,
                      gt_value=" ... ".join(g.text for g in gt_items),
                      model_value=" ; ".join(chosen[g].model.text for g in gt_items if g in chosen),
                      score=len(chosen) / len(gt_items), passed=not missed, reason=reason)


def score_named(groups, ctx: StoryContext) -> list:
    records = []
    for g in groups:
        parts_by_row = defaultdict(list)
        for p in g.gt_parts:
            parts_by_row[p.row].append(p.text)
        rows_by_sj = defaultdict(list)
        for row, parts in parts_by_row.items():
            if len(set(parts)) >= 2:
                rows_by_sj[tuple(parts)].append(row)
        model_texts = [p.text for p in g.model_parts]
        model_rows = {p.row for p in g.model_parts}
        for parts, rows in rows_by_sj.items():
            records.append(coverage_record(ctx, parts, model_texts, "named",
                                           g.source_type, g.name, rows, model_rows))
    return records


def score_unnamed(rows, ctx: StoryContext) -> list:
    return [
        coverage_record(ctx, r.gt_parts, r.model_parts, "unnamed", r.source_type,
                        "(unnamed)", [r.gt_row], r.model_rows)
        for r in rows if len(set(r.gt_parts)) >= 2
    ]
