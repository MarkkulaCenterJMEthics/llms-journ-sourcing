"""
Turns the evaluator's statement pairing into the two units the SJ scorers work
on:

* NamedGroup  -- one Named Person / Named Organization source, with its GT SJ
  parts and the model SJ parts linked to it, each placed at a position counted
  in the source's OWN statements (its 1st, 2nd, 3rd ... GT row).
* UnnamedRow  -- one GT row of any other source type (or a named type with no
  name), compared row by row with the model rows paired to it.

Inputs are the evaluator's frames: gt_df / llm_df (with `Justification_Row`,
the row's own SJ before pooling by name) and result_df (with `GT_Row` and
`LLM_Rows`, the statement pairing compare_attributions recorded).
"""
from collections import defaultdict
from dataclasses import dataclass

from rapidfuzz import fuzz

from .common import NAME_MATCH, NAMED_TYPES, clean_text, split_sj


@dataclass(frozen=True)
class Part:
    text: str
    position: int
    row: int


@dataclass(frozen=True)
class NamedGroup:
    source_type: str
    name: str
    gt_parts: tuple          # Parts; position = the source's n-th GT statement
    model_parts: tuple       # Parts; position inherited from the paired GT row
    linked_positions: frozenset  # GT positions the model found a statement for
    attached_rows: int       # unpaired model rows attached by the name fallback


@dataclass(frozen=True)
class UnnamedRow:
    source_type: str
    gt_row: int
    gt_parts: tuple          # strings
    model_rows: tuple
    model_parts: tuple       # strings, from every model row paired to gt_row


def llm_rows_by_gt(result_df) -> dict:
    """{gt_row: [model rows paired to it]} from the evaluator's result_df."""
    return {
        int(r["GT_Row"]): list(r["LLM_Rows"])
        for _, r in result_df.iterrows() if int(r["GT_Row"]) >= 0
    }


def is_named(gt_row) -> bool:
    return gt_row["SourceType"] in NAMED_TYPES and bool(clean_text(gt_row["Name"]))


def attach_unpaired(name, unpaired_llm, llm_position, llm_df) -> dict:
    """
    Name fallback: model rows paired to no GT statement whose Name matches this
    source, placed at the nearest FOLLOWING paired row of the source (nearest
    preceding if none follows) -- where v60sj step 9 puts a global SJ. Needed
    because v60sj models make a source's introduction sentence its own Sourced
    Statement and put the global SJ on it, while the GT treats that sentence as
    SJ only, so the row never pairs. {model_row: position}.
    """
    if not llm_position:
        return {}
    attached = {}
    for j in unpaired_llm:
        if fuzz.ratio(name, clean_text(llm_df.loc[j, "Name"])) <= NAME_MATCH:
            continue
        following = [k for k in llm_position if k > j]
        anchor = min(following) if following else max(llm_position)
        attached[j] = llm_position[anchor]
    return attached


def named_groups(gt_df, llm_df, result_df) -> list[NamedGroup]:
    pairs = llm_rows_by_gt(result_df)
    paired_llm = {j for js in pairs.values() for j in js}
    unpaired_llm = [j for j in llm_df.index if j not in paired_llm]

    rows_by_source = defaultdict(list)
    for i, row in gt_df.iterrows():
        if is_named(row):
            rows_by_source[(row["SourceType"], clean_text(row["Name"]))].append(i)

    groups = []
    for (stype, name), gt_idxs in rows_by_source.items():
        position = {i: n for n, i in enumerate(sorted(gt_idxs))}
        llm_position = {j: position[i] for i in gt_idxs for j in pairs.get(i, [])}
        attached = attach_unpaired(name, unpaired_llm, llm_position, llm_df)
        gt_parts = tuple(
            Part(t, position[i], i)
            for i in sorted(gt_idxs) for t in split_sj(gt_df.loc[i, "Justification_Row"])
        )
        model_parts = tuple(
            Part(t, pos, j)
            for j, pos in sorted({**llm_position, **attached}.items())
            for t in split_sj(llm_df.loc[j, "Justification_Row"])
        )
        groups.append(NamedGroup(stype, name, gt_parts, model_parts,
                                 frozenset(llm_position.values()), len(attached)))
    return groups


def unnamed_rows(gt_df, llm_df, result_df) -> list[UnnamedRow]:
    pairs = llm_rows_by_gt(result_df)
    rows = []
    for i, row in gt_df.iterrows():
        if is_named(row):
            continue
        model_rows = tuple(pairs.get(i, []))
        model_parts = tuple(
            t for j in model_rows for t in split_sj(llm_df.loc[j, "Justification_Row"])
        )
        rows.append(UnnamedRow(row["SourceType"], i,
                               tuple(split_sj(row["Justification_Row"])),
                               model_rows, model_parts))
    return rows
