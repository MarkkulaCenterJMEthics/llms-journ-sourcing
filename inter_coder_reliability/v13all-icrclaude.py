import pandas as pd
import simpledorff
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import argparse
from fuzzywuzzy import fuzz

# Force UTF-8 output so the ✅/α/≥/≤ symbols below don't raise UnicodeEncodeError
# on Windows, where stdout defaults to the system codepage (e.g. cp1252) instead
# of UTF-8 — especially when output is redirected to a file (`> results.txt`).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# A blank cell means an annotator missed a sourced statement entirely (or found
# no title/justification for one they did catch). We substitute this literal
# string for NaN before handing data to simpledorff, because simpledorff's
# internal pivot table calls pandas .dropna() and drops any item where one
# annotator's cell is real NaN -- silently excluding it from the alpha
# calculation instead of counting it as a disagreement. A non-null sentinel
# survives .dropna() so the item stays in the calculation; _is_missing() below
# is what still makes it score as a true missing value rather than literal text.
MISSING_SENTINEL = "MISSING_VALUE"

def _is_missing(value):
    """True for real NaN/None or for our post-substitution missing sentinel."""
    return pd.isna(value) or value is None or value == MISSING_SENTINEL

def load_semantic_model():
    """Load the semantic model once to be used throughout the script."""
    print("Loading semantic model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded.")
    return model

def semantic_distance_metric(a, b, model):
    """
    Calculates semantic disagreement between two text strings.
    Returns 0.0 for perfect agreement (identical meaning) and 1.0 for total disagreement.
    
    Special case: If both values are NULL/missing, return 0.0 (perfect agreement)
    since this indicates the data didn't exist in the source material for both annotators.
    """
    # Check if both values are missing - this is perfect agreement (both correctly identified no data)
    if _is_missing(a) and _is_missing(b):
        return 0.0

    # If only one value is missing, this is disagreement
    if _is_missing(a) or _is_missing(b):
        return 1.0

    # Convert text to vector embeddings
    embeddings = model.encode([str(a), str(b)])

    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Convert similarity (0 to 1) to distance (1 to 0)
    distance = 1 - similarity
    return distance

def build_cached_semantic_distance_fn(model, texts):
    """
    Precompute embeddings once (batched) for every distinct non-missing text in
    `texts`, then return a metric_fn that looks them up instead of re-encoding
    on every pairwise comparison.

    simpledorff's Krippendorff's Alpha calls metric_fn(c, k) for every pair of
    distinct classes when computing expected disagreement -- for a column of
    mostly-unique free text (e.g. Sourced Statements), that's O(U^2) calls,
    where U is the number of distinct values. Without caching, each call reruns
    a full transformer forward pass (~30-60ms measured), which explodes into
    tens of minutes once U reaches ~150 (typical for a 100-row CSV, since
    statements are mostly unique text). Caching cuts the encode work to O(U)
    via one batched call; every pairwise comparison then reuses those vectors,
    so the returned distances are numerically identical to calling
    semantic_distance_metric(a, b, model) directly -- only the redundant
    re-encoding is eliminated, not the calculation itself.
    """
    distinct_texts = sorted({str(t) for t in texts if not _is_missing(t)})
    embeddings = model.encode(distinct_texts) if distinct_texts else []
    cache = dict(zip(distinct_texts, embeddings))

    def distance_fn(a, b):
        if _is_missing(a) and _is_missing(b):
            return 0.0
        if _is_missing(a) or _is_missing(b):
            return 1.0
        embedding_a = cache[str(a)]
        embedding_b = cache[str(b)]
        similarity = cosine_similarity([embedding_a], [embedding_b])[0][0]
        return 1 - similarity

    return distance_fn

def fuzzy_distance_metric(a, b):
    """
    Calculates fuzzy string disagreement between two text strings.
    Returns 0.0 for perfect agreement and 1.0 for total disagreement.
    Uses fuzzy string matching for approximate text comparison.
    
    Special case: If both values are NULL/missing, return 0.0 (perfect agreement)
    since this indicates the data didn't exist in the source material for both annotators.
    """
    # Check if both values are missing - this is perfect agreement (both correctly identified no data)
    if _is_missing(a) and _is_missing(b):
        return 0.0

    # If only one value is missing, this is disagreement
    if _is_missing(a) or _is_missing(b):
        return 1.0

    # Convert to strings and calculate fuzzy ratio
    str_a = str(a).strip()
    str_b = str(b).strip()
    
    # Use fuzzy ratio (0-100) and convert to similarity (0-1)
    fuzzy_ratio = fuzz.ratio(str_a, str_b)
    similarity = fuzzy_ratio / 100.0
    
    # Convert similarity (0 to 1) to distance (1 to 0)
    distance = 1 - similarity
    return distance

def load_and_prepare_data(csv_file1, csv_file2):
    """Load CSV files and prepare data for analysis."""
    print(f"Loading data from {csv_file1} and {csv_file2}...")
    
    # Load the CSV files
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    print(f"CSV 1 shape: {df1.shape}")
    print(f"CSV 2 shape: {df2.shape}")
    
    # Check if required columns exist
    required_columns = ["Sourced Statements", "Source Justification", "Type of Source", "Name of Source", "Title of Source"]
    for col in required_columns:
        if col not in df1.columns:
            raise ValueError(f"Column '{col}' not found in {csv_file1}")
        if col not in df2.columns:
            raise ValueError(f"Column '{col}' not found in {csv_file2}")
    
    # Replace blank cells with a non-null sentinel so simpledorff can't silently
    # drop these items -- see MISSING_SENTINEL comment above for why.
    for col in required_columns:
        df1[col] = df1[col].fillna(MISSING_SENTINEL)
        df2[col] = df2[col].fillna(MISSING_SENTINEL)

    # Assuming both CSVs have the same number of rows and correspond to the same items
    # Create item IDs based on row index
    df1['item_id'] = range(1, len(df1) + 1)
    df2['item_id'] = range(1, len(df2) + 1)

    return df1, df2

def compute_ss_gate_mask(df1, df2):
    """
    Boolean array (aligned by row position with df1/df2), True where BOTH
    annotators recorded a Sourced Statement for that item.

    Type/Name/Title/Justification of Source are only meaningful once both
    annotators agree a sourced statement exists at all -- there's no intended
    annotation work to compare on those columns for an item one annotator
    never flagged as sourced in the first place. Items where the gate is False
    are excluded entirely (not counted as agreement or disagreement) from
    those four columns' Krippendorff's Alpha. Sourced Statements itself is
    never gated by this mask: one annotator missing what the other found is
    exactly the disagreement that column's alpha should capture.
    """
    ss1_found = ~df1['Sourced Statements'].apply(_is_missing)
    ss2_found = ~df2['Sourced Statements'].apply(_is_missing)
    return (ss1_found & ss2_found).to_numpy()

def prepare_data_for_column(df1, df2, column_name, gate_mask=None):
    """Prepare data for a specific column in long format for simpledorff."""
    if gate_mask is not None:
        df1 = df1[gate_mask]
        df2 = df2[gate_mask]

    # Create combined dataframe with both annotators' data
    data = {
        'item_id': list(df1['item_id']) + list(df2['item_id']),
        'annotator': ['annotator1'] * len(df1) + ['annotator2'] * len(df2),
        'annotation_text': list(df1[column_name]) + list(df2[column_name])
    }

    df_long = pd.DataFrame(data)
    return df_long

def calculate_icr_for_column(df1, df2, column_name, distance_function, gate_mask=None):
    """Calculate ICR for a specific column using the specified distance function.

    gate_mask, if given, is a boolean array (aligned by row position with
    df1/df2) -- items where it's False are excluded from the calculation
    entirely (see compute_ss_gate_mask).
    """
    print(f"\n{'='*60}")
    print(f"Calculating ICR for column: {column_name}")
    print(f"{'='*60}")

    # Prepare data in long format
    df_long = prepare_data_for_column(df1, df2, column_name, gate_mask=gate_mask)

    # Show distances for ALL items
    print(f"\nDistances for all {len(df1)} items:")
    print("-" * 80)
    for i in range(len(df1)):
        if gate_mask is not None and not gate_mask[i]:
            print(f"Item {i+1}: SKIPPED (no Sourced Statement recorded by both annotators)")
            print()
            continue

        text1 = df1.iloc[i][column_name]
        text2 = df2.iloc[i][column_name]
        distance = distance_function(text1, text2)

        # Format text for display (truncate if too long)
        text1_display = str(text1)[:100] + "..." if pd.notna(text1) and len(str(text1)) > 100 else str(text1)
        text2_display = str(text2)[:100] + "..." if pd.notna(text2) and len(str(text2)) > 100 else str(text2)

        print(f"Item {i+1}: Distance = {distance:.4f}")
        print(f"  Annotator1: {text1_display}")
        print(f"  Annotator2: {text2_display}")
        print()

    if gate_mask is not None:
        skipped = int((~gate_mask).sum())
        if skipped:
            print(f"({skipped} item(s) skipped: no Sourced Statement recorded by both annotators)\n")

    # Calculate Krippendorff's Alpha
    print(f"Calculating Krippendorff's Alpha for {column_name}...")
    try:
        alpha_score = simpledorff.calculate_krippendorffs_alpha_for_df(
            df_long,
            experiment_col='item_id',
            annotator_col='annotator',
            class_col='annotation_text',
            metric_fn=distance_function
        )
    except ZeroDivisionError:
        # Alpha is mathematically indeterminate (0/0), not 0.0 or 1.0, when
        # every pairable item has the same value: there's no variability, so
        # "expected disagreement by chance" is exactly zero and the formula
        # has nothing to divide by. Report which value(s) tied and how many
        # items, so this doesn't require digging through the log above to
        # explain (see readme.md for why this happens on small samples).
        n_items = df_long['item_id'].nunique()
        value_counts = df_long['annotation_text'].value_counts()
        values_summary = ", ".join(f"'{v}' ({c} ratings)" for v, c in value_counts.items())
        reason = f"all {n_items} comparable item(s) got the same value from both annotators: {values_summary}"
        print(f"⚠️  Krippendorff's Alpha for '{column_name}' is UNDEFINED: {reason}. "
              f"With zero variability there's no chance-disagreement baseline for "
              f"the formula to compare against.")
        return float("nan"), reason

    print(f"✅ Krippendorff's Alpha for '{column_name}': {alpha_score:.4f}")
    return alpha_score, None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate Inter-Coder Reliability using semantic similarity and fuzzy matching')
    parser.add_argument('csv1', help='Path to first annotator CSV file')
    parser.add_argument('csv2', help='Path to second annotator CSV file')
    
    args = parser.parse_args()
    
    try:
        # Load semantic model
        model = load_semantic_model()
        
        # Load and prepare data
        df1, df2 = load_and_prepare_data(args.csv1, args.csv2)

        # Items where either annotator has no Sourced Statement recorded are
        # excluded from the four "dependent" columns below -- see
        # compute_ss_gate_mask for why. Sourced Statements itself is never gated.
        gate_mask = compute_ss_gate_mask(df1, df2)
        gated_columns = {"Type of Source", "Name of Source", "Title of Source", "Source Justification"}

        # Precompute embeddings once per distinct value for each semantic column,
        # instead of letting simpledorff re-encode text on every pairwise
        # comparison (see build_cached_semantic_distance_fn for why that matters).
        semantic_columns = ["Sourced Statements", "Source Justification", "Title of Source"]
        semantic_distance_fns = {
            col: build_cached_semantic_distance_fn(model, list(df1[col]) + list(df2[col]))
            for col in semantic_columns
        }

        # Define columns and their corresponding distance functions
        columns_config = {
            "Sourced Statements": semantic_distance_fns["Sourced Statements"],
            "Source Justification": semantic_distance_fns["Source Justification"],
            "Type of Source": fuzzy_distance_metric,
            "Name of Source": fuzzy_distance_metric,
            "Title of Source": semantic_distance_fns["Title of Source"]
        }

        results = {}

        # Calculate ICR for all columns
        for column, distance_func in columns_config.items():
            mask = gate_mask if column in gated_columns else None
            alpha_score, undefined_reason = calculate_icr_for_column(df1, df2, column, distance_func, gate_mask=mask)
            results[column] = (alpha_score, undefined_reason)
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY RESULTS - KRIPPENDORFF'S ALPHA SCORES")
        print(f"{'='*60}")
        for column, (score, undefined_reason) in results.items():
            if pd.isna(score):
                print(f"{column}: Undefined ({undefined_reason})")
            else:
                print(f"{column}: {score:.4f}")
        print(f"{'='*60}")
        
        # Print interpretation guide
        print("\nInterpretation Guide:")
        print("α ≥ 0.800: High reliability")
        print("0.667 ≤ α < 0.800: Moderate reliability")
        print("α < 0.667: Low reliability")
        print("α ≤ 0.000: No reliability")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()