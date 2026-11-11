"""
SJ_Placement_Score (point 3): did the model put the GT's SJ on the right row?

* Named Person / Named Organization: distance is counted in the source's OWN
  statements. A pair within MARGIN (default 1) earns its full similarity; a
  pair further away earns 0. The source-level match (no distance limit) is
  only consulted to explain a miss ("found but 3 statements away").
* Every other source: row by row. A GT row's SJ parts are compared only with
  the SJ parts of the model rows paired to that same GT row.

Each GT SJ part is one unit of the denominator (named: distinct per source,
however many rows the GT copied it onto; unnamed: per row).
"""
from .common import SJItem, StoryContext, assign, items_and_rows, miss_reason

METRIC = "SJ_Placement"


def score_named(groups, ctx: StoryContext) -> list:
    records = []
    for g in groups:
        gt_items, gt_rows = items_and_rows(g.gt_parts)
        model_items, model_rows = items_and_rows(g.model_parts)
        placed = assign(gt_items, model_items, ctx.score_fn, ctx.threshold, ctx.margin)
        found = assign(gt_items, model_items, ctx.score_fn, ctx.threshold)
        for item in sorted(gt_items, key=lambda it: min(it.positions)):
            pair = placed.get(item)
            if pair:
                values = dict(model_value=pair.model.text, score=pair.score, passed=True,
                              reason="placed (distance %d)" % pair.distance)
                rows = model_rows[pair.model.text]
            elif item in found:
                far = found[item]
                values = dict(model_value=far.model.text, score=0.0, passed=False,
                              reason="found but %d statements away" % far.distance)
                rows = model_rows[far.model.text]
            else:
                if not item.positions & g.linked_positions:
                    reason = "statement not found by model"
                else:
                    reason = miss_reason(item, model_items, ctx.score_fn, ctx.threshold)
                values = dict(model_value="", score=0.0, passed=False, reason=reason)
                rows = ()
            records.append(ctx.record(METRIC, "named", g.source_type, g.name,
                                      gt_rows[item.text], rows, gt_value=item.text, **values))
    return records


def row_items(texts) -> list:
    """Distinct parts of one row, all at the same position (row by row)."""
    return [SJItem(t, frozenset({0})) for t in dict.fromkeys(texts)]


def score_unnamed(rows, ctx: StoryContext) -> list:
    records = []
    for r in rows:
        gt_items, model_items = row_items(r.gt_parts), row_items(r.model_parts)
        chosen = assign(gt_items, model_items, ctx.score_fn, ctx.threshold)
        for item in gt_items:
            pair = chosen.get(item)
            if pair:
                values = dict(model_value=pair.model.text, score=pair.score,
                              passed=True, reason="matched on this row")
            else:
                if not r.model_rows:
                    reason = "statement not found by model"
                elif not model_items:
                    reason = "model gave no SJ on this row"
                else:
                    reason = miss_reason(item, model_items, ctx.score_fn, ctx.threshold)
                values = dict(model_value="", score=0.0, passed=False, reason=reason)
            records.append(ctx.record(METRIC, "unnamed", r.source_type, "(unnamed)",
                                      [r.gt_row], r.model_rows, gt_value=item.text, **values))
    return records
