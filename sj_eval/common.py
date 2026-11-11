"""
Shared pieces of the Source Justification (SJ) scorers: splitting an SJ cell
into parts, one-to-one matching of GT parts to model parts, and the Record
every scorer returns.

Nothing here imports the evaluator. The similarity function is passed in
(eval_multi_llm_article_v2.semantic_match in real runs, a toy in the tests).
"""
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

THRESHOLD = 0.55
MARGIN = 1
NAME_MATCH = 80
SJ_SEPARATORS = re.compile(r";|\.\.\.|…")
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
NULLISH = {"", "null", "`null`", "none", "n/a", "nan"}
NAMED_TYPES = {"Named_Person", "Named_Organization"}

ScoreFn = Callable[[str, str], float]


def clean_text(value) -> str:
    """Lowercase + collapse whitespace; the evaluator's clean_text, NaN-safe."""
    if value is None or value != value:  # NaN
        return ""
    return re.sub(r"\s+", " ", str(value).lower().strip())


def split_sj(value) -> list[str]:
    """
    Split an SJ cell into its parts on ';' and '...', dropping blanks and null
    words. v60sj joins a global and a local SJ with ';'; the GT annotators used
    '...'. Both are separators until the team standardizes one.
    """
    text = clean_text(value).translate(QUOTES)
    parts = [p.strip(" .,") for p in SJ_SEPARATORS.split(text)]
    return [p for p in parts if p not in NULLISH]


@dataclass(frozen=True)
class SJItem:
    """One distinct SJ part and every position it was written at."""
    text: str
    positions: frozenset


@dataclass(frozen=True)
class Pair:
    gt: SJItem
    model: SJItem
    score: float
    distance: int


@dataclass(frozen=True)
class Record:
    """
    One scored item. Every SJ scorer returns these; the summary averages
    `score` and the mismatch workbook writes the ones with passed=False, so the
    two can never disagree.
    """
    metric: str
    story: int
    model: str
    bucket: str              # "named" (Named Person/Org) or "unnamed" (row by row)
    source_type: str
    source_name: str
    gt_value: str
    model_value: str
    score: float
    passed: bool
    reason: str
    gt_rows: tuple = field(default_factory=tuple)
    model_rows: tuple = field(default_factory=tuple)
    gt_statement: str = ""
    model_statement: str = ""


@dataclass(frozen=True, eq=False)
class StoryContext:
    """Everything a scorer needs besides its groups/rows."""
    story: int
    model: str
    score_fn: ScoreFn
    gt_df: object
    llm_df: object
    threshold: float = THRESHOLD
    margin: int = MARGIN

    def gt_statement(self, rows) -> str:
        return " | ".join(self.gt_df.loc[r, "SourcedStatement"] for r in rows)

    def model_statement(self, rows) -> str:
        return " | ".join(self.llm_df.loc[r, "SourcedStatement"] for r in rows)

    def record(self, metric, bucket, source_type, source_name, gt_rows, model_rows,
               **values) -> Record:
        gt_rows, model_rows = tuple(sorted(gt_rows)), tuple(sorted(model_rows))
        return Record(metric, self.story, self.model, bucket, source_type, source_name,
                      gt_rows=gt_rows, model_rows=model_rows,
                      gt_statement=self.gt_statement(gt_rows),
                      model_statement=self.model_statement(model_rows), **values)


def items_and_rows(parts):
    """Parts -> ([SJItem], {text: set of rows it was written on})."""
    rows = defaultdict(set)
    for p in parts:
        rows[p.text].add(p.row)
    return collect_items((p.text, p.position) for p in parts), rows


def collect_items(parts_with_positions) -> list[SJItem]:
    """[(text, position)] -> [SJItem], one per distinct text, positions pooled."""
    positions = defaultdict(set)
    for text, pos in parts_with_positions:
        positions[text].add(pos)
    return [SJItem(t, frozenset(p)) for t, p in positions.items()]


def distance(a: SJItem, b: SJItem) -> int:
    return min(abs(p - q) for p in a.positions for q in b.positions)


def assign(gt_items, model_items, score_fn: ScoreFn, threshold: float = THRESHOLD,
           margin: Optional[int] = None) -> dict:
    """
    One-to-one pairing, best score first. A model SJ satisfies at most one GT
    SJ and vice versa. With `margin`, only pairs within that distance qualify.
    Returns {gt_item: Pair}.
    """
    candidates = []
    for g in gt_items:
        for m in model_items:
            d = distance(g, m)
            if margin is not None and d > margin:
                continue
            s = score_fn(g.text, m.text)
            if s > threshold:
                candidates.append(Pair(g, m, s, d))
    candidates.sort(key=lambda p: -p.score)
    taken_gt, taken_model, chosen = set(), set(), {}
    for p in candidates:
        if p.gt in taken_gt or p.model in taken_model:
            continue
        taken_gt.add(p.gt)
        taken_model.add(p.model)
        chosen[p.gt] = p
    return chosen


def miss_reason(item: SJItem, model_items, score_fn: ScoreFn, threshold: float) -> str:
    """Why a GT part went unmatched, for the mismatch workbook."""
    scores = [score_fn(item.text, m.text) for m in model_items]
    if not scores:
        return "model gave no SJ"
    best = max(scores)
    if best > threshold:
        return "best model SJ (%.3f) already used by another GT SJ" % best
    return "below threshold (best %.3f)" % best
