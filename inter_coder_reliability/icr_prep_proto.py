import sys
import pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location('icr', 'v13all-icrclaude.py')
icr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(icr)

REQUIRED_COLS = ["Sourced Statements", "Type of Source", "Name of Source", "Title of Source", "Source Justification"]
GAP_PENALTY = 0.3
EXACT_THRESHOLD = 0.02          # match distance below this = "matched" (near-identical)
TYPO_FUZZY_THRESHOLD = 0.20     # fuzzy distance to nearest canonical category, below this = likely typo
BORDERLINE_MARGIN = 0.10        # a gap's closest candidate within GAP_PENALTY + this = worth a second look

def align(s1, s2, dist_fn, gap_penalty=GAP_PENALTY):
    n, m = len(s1), len(s2)
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i * gap_penalty
    for j in range(1, m+1):
        dp[0][j] = j * gap_penalty
    for i in range(1, n+1):
        for j in range(1, m+1):
            match_cost = dp[i-1][j-1] + dist_fn(s1[i-1], s2[j-1])
            del1_cost = dp[i-1][j] + gap_penalty
            del2_cost = dp[i][j-1] + gap_penalty
            dp[i][j] = min(match_cost, del1_cost, del2_cost)
    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(dp[i][j] - (dp[i-1][j-1] + dist_fn(s1[i-1], s2[j-1]))) < 1e-9:
            pairs.append((i-1, j-1)); i -= 1; j -= 1
        elif i > 0 and abs(dp[i][j] - (dp[i-1][j] + gap_penalty)) < 1e-9:
            pairs.append((i-1, None)); i -= 1
        else:
            pairs.append((None, j-1)); j -= 1
    pairs.reverse()
    return pairs

def truncate(s, n=75):
    s = str(s).replace("\n", " ").replace("\r", " ")
    return s[:n] + ("..." if len(s) > n else "")

def prep_report(f1, f2):
    model = icr.load_semantic_model()
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    for col in REQUIRED_COLS:
        if col not in df1.columns:
            raise ValueError(f"Column '{col}' not found in {f1}")
        if col not in df2.columns:
            raise ValueError(f"Column '{col}' not found in {f2}")

    s1 = df1['Sourced Statements'].tolist()
    s2 = df2['Sourced Statements'].tolist()
    dist_fn = icr.build_cached_semantic_distance_fn(model, s1 + s2)
    pairs = align(s1, s2, dist_fn)

    missing = []       # (which_file_missing, orig_row_in_other_file, text)
    matched = []        # (ann1_row, ann2_row, distance)
    for i, j in pairs:
        if i is not None and j is not None:
            matched.append((i+1, j+1, dist_fn(s1[i], s2[j])))
        elif i is not None:
            missing.append(('ann2', i+1, s1[i]))
        else:
            missing.append(('ann1', j+1, s2[j]))

    # Category 3: borderline alignment calls. A gap only belongs here if its
    # closest candidate in the other file was a genuinely close call against
    # the gap penalty -- not every gap's closest candidate regardless of how
    # far away it is. A "total mismatch" (closest candidate far beyond the gap
    # penalty) is exactly what category 1 already means; re-listing it here
    # with a distant "closest candidate" attached is noise, not signal.
    borderline = []
    for which_missing, orig_row, text in missing:
        if which_missing == 'ann2':
            # s1[orig_row-1] has no match; find its closest candidate in s2
            src_text = s1[orig_row - 1]
            candidates = [(dist_fn(src_text, s2[k]), k+1) for k in range(len(s2))]
        else:
            src_text = s2[orig_row - 1]
            candidates = [(dist_fn(src_text, s1[k]), k+1) for k in range(len(s1))]
        if candidates:
            best_dist, best_row = min(candidates, key=lambda x: x[0])
            if best_dist <= GAP_PENALTY + BORDERLINE_MARGIN:
                borderline.append((which_missing, orig_row, text, best_dist, best_row))

    # Category 4: Type of Source data-entry errors (typo of a canonical value)
    typo_errors = []
    for fname, df in [(f1, df1), (f2, df2)]:
        for idx, raw in df['Type of Source'].items():
            if pd.isna(raw):
                continue
            key = str(raw).strip().lower()
            if key in icr.TYPE_OF_SOURCE_VARIANTS:
                continue  # already recognized, not an error
            best_cat, best_dist = min(
                ((cat, icr.fuzzy_distance_metric(str(raw), cat)) for cat in icr.TYPE_OF_SOURCE_CATEGORIES),
                key=lambda x: x[1]
            )
            if best_dist <= TYPO_FUZZY_THRESHOLD:
                typo_errors.append((fname, idx+1, raw, best_cat, best_dist))

    # --- Print report ---
    print(f"{'='*70}\nPREP REPORT: {f1} / {f2}\n{'='*70}\n")

    print(f"1. MISSING ROW CASES ({len(missing)})")
    print("-"*70)
    if not missing:
        print("  None.")
    for which_missing, orig_row, text in missing:
        found_in = 'ann1' if which_missing == 'ann2' else 'ann2'
        print(f"  {found_in} row {orig_row} has no counterpart in {which_missing}: {truncate(text)!r}")
    print()

    exact = [m for m in matched if m[2] <= EXACT_THRESHOLD]
    rough = [m for m in matched if m[2] > EXACT_THRESHOLD]
    print(f"2. MATCHED AND ROUGHLY MATCHED PAIRS ({len(matched)} total: {len(exact)} near-identical, {len(rough)} rough)")
    print("-"*70)
    for a1, a2, d in exact:
        print(f"  [near-identical] ann1 row {a1} <-> ann2 row {a2}, distance {d:.4f}")
        print(f"    ann1: {truncate(s1[a1-1])!r}")
        print(f"    ann2: {truncate(s2[a2-1])!r}")
    for a1, a2, d in rough:
        # Longer truncation than the near-identical case above -- the whole
        # point of a "rough match" is that the text differs somewhere, so a
        # short preview that hides the actual difference defeats the purpose.
        print(f"  [rough match]    ann1 row {a1} <-> ann2 row {a2}, distance {d:.4f}")
        print(f"    ann1: {truncate(s1[a1-1], 250)!r}")
        print(f"    ann2: {truncate(s2[a2-1], 250)!r}")
    print()

    print(f"3. MISMATCHED PAIRS -- borderline alignment calls ({len(borderline)})")
    print("-"*70)
    if not borderline:
        print("  None -- every gap's closest candidate was clearly beyond threshold, no close calls.")
    else:
        print(f"  {len(borderline)} found -- see below.")
    print(f"  (Gaps whose closest candidate in the other file came within {BORDERLINE_MARGIN:.2f} of the")
    print(f"  gap penalty ({GAP_PENALTY:.2f}) -- a genuinely close call worth a second look, not every")
    print("  gap regardless of distance -- a total mismatch is already category 1.)")
    for which_missing, orig_row, text, best_dist, best_row in borderline:
        found_in = 'ann1' if which_missing == 'ann2' else 'ann2'
        other = 'ann2' if which_missing == 'ann2' else 'ann1'
        print(f"  {found_in} row {orig_row} ({truncate(text, 50)!r}) -- closest candidate: "
              f"{other} row {best_row}, distance {best_dist:.4f}")
    print()

    print(f"4. DATA ENTRY ERRORS -- Type of Source ({len(typo_errors)})")
    print("-"*70)
    if not typo_errors:
        print("  None.")
    for fname, row, raw, best_cat, best_dist in typo_errors:
        print(f"  {fname.split('/')[-1]}, row {row}: {raw!r} -- likely typo of {best_cat!r} (fuzzy distance {best_dist:.4f})")
    print()
    print(f"{'='*70}")
    print("No files were modified. This is a report only.")

if __name__ == '__main__':
    prep_report(sys.argv[1], sys.argv[2])
