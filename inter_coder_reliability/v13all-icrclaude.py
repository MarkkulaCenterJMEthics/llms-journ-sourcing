import pandas as pd
import simpledorff
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import argparse
from fuzzywuzzy import fuzz

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
    if (pd.isna(a) or a is None) and (pd.isna(b) or b is None):
        return 0.0
    
    # If only one value is missing, this is disagreement
    if pd.isna(a) or pd.isna(b) or a is None or b is None:
        return 1.0

    # Convert text to vector embeddings
    embeddings = model.encode([str(a), str(b)])

    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # Convert similarity (0 to 1) to distance (1 to 0)
    distance = 1 - similarity
    return distance

def fuzzy_distance_metric(a, b):
    """
    Calculates fuzzy string disagreement between two text strings.
    Returns 0.0 for perfect agreement and 1.0 for total disagreement.
    Uses fuzzy string matching for approximate text comparison.
    
    Special case: If both values are NULL/missing, return 0.0 (perfect agreement)
    since this indicates the data didn't exist in the source material for both annotators.
    """
    # Check if both values are missing - this is perfect agreement (both correctly identified no data)
    if (pd.isna(a) or a is None) and (pd.isna(b) or b is None):
        return 0.0
    
    # If only one value is missing, this is disagreement
    if pd.isna(a) or pd.isna(b) or a is None or b is None:
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
    
    # Assuming both CSVs have the same number of rows and correspond to the same items
    # Create item IDs based on row index
    df1['item_id'] = range(1, len(df1) + 1)
    df2['item_id'] = range(1, len(df2) + 1)
    
    return df1, df2

def prepare_data_for_column(df1, df2, column_name):
    """Prepare data for a specific column in long format for simpledorff."""
    # Create combined dataframe with both annotators' data
    data = {
        'item_id': list(df1['item_id']) + list(df2['item_id']),
        'annotator': ['annotator1'] * len(df1) + ['annotator2'] * len(df2),
        'annotation_text': list(df1[column_name]) + list(df2[column_name])
    }
    
    df_long = pd.DataFrame(data)
    return df_long

def calculate_icr_for_column(df1, df2, column_name, distance_function):
    """Calculate ICR for a specific column using the specified distance function."""
    print(f"\n{'='*60}")
    print(f"Calculating ICR for column: {column_name}")
    print(f"{'='*60}")
    
    # Prepare data in long format
    df_long = prepare_data_for_column(df1, df2, column_name)
    
    # Show distances for ALL items
    print(f"\nDistances for all {len(df1)} items:")
    print("-" * 80)
    for i in range(len(df1)):
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
    
    # Calculate Krippendorff's Alpha
    print(f"Calculating Krippendorff's Alpha for {column_name}...")
    alpha_score = simpledorff.calculate_krippendorffs_alpha_for_df(
        df_long,
        experiment_col='item_id',
        annotator_col='annotator',
        class_col='annotation_text',
        metric_fn=distance_function
    )
    
    print(f"✅ Krippendorff's Alpha for '{column_name}': {alpha_score:.4f}")
    return alpha_score

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
        
        # Define columns and their corresponding distance functions
        columns_config = {
            "Sourced Statements": lambda a, b: semantic_distance_metric(a, b, model),
            "Source Justification": lambda a, b: semantic_distance_metric(a, b, model),
            "Type of Source": fuzzy_distance_metric,
            "Name of Source": fuzzy_distance_metric,
            "Title of Source": lambda a, b: semantic_distance_metric(a, b, model)
        }
        
        results = {}
        
        # Calculate ICR for all columns
        for column, distance_func in columns_config.items():
            alpha_score = calculate_icr_for_column(df1, df2, column, distance_func)
            results[column] = alpha_score
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY RESULTS - KRIPPENDORFF'S ALPHA SCORES")
        print(f"{'='*60}")
        for column, score in results.items():
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