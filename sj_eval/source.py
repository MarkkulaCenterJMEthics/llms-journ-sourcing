"""
SJ_Source_Score (point 2): did the model find the GT's SJ for a Named Person /
Named Organization source ANYWHERE among that source's rows?

Each distinct GT SJ part is one unit of the denominator. GT and model parts are
paired one-to-one, best similarity first; a pair counts only above the
threshold and earns its raw similarity score.
"""
from .common import StoryContext, assign, items_and_rows, miss_reason

METRIC = "SJ_Source"


def score(groups, ctx: StoryContext) -> list:
    records = []
    for g in groups:
        gt_items, gt_rows = items_and_rows(g.gt_parts)
        model_items, model_rows = items_and_rows(g.model_parts)
        found = assign(gt_items, model_items, ctx.score_fn, ctx.threshold)
        for item in sorted(gt_items, key=lambda it: min(it.positions)):
            pair = found.get(item)
            if pair:
                values = dict(model_value=pair.model.text, score=pair.score,
                              passed=True, reason="matched")
                rows = model_rows[pair.model.text]
            else:
                if not g.linked_positions and not g.model_parts:
                    reason = "model never found this source"
                else:
                    reason = miss_reason(item, model_items, ctx.score_fn, ctx.threshold)
                values = dict(model_value="", score=0.0, passed=False, reason=reason)
                rows = ()
            records.append(ctx.record(METRIC, "named", g.source_type, g.name,
                                      gt_rows[item.text], rows, gt_value=item.text, **values))
    return records
