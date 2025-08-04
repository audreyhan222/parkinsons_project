import pandas as pd
from user_parser import get_user_dataframe
from keystroke_parser import get_keystroke_features_dataframe
from merge_and_clean import get_merged_and_cleaned_dataframe
from parkinsons_classification import run_classification_pipeline

def run_pipeline():
    print("--- Starting Data Processing Pipeline ---")

    # 1. Get User DataFrame
    user_df = get_user_dataframe()
    print("\n--- User DataFrame ---")
    print("Shape:", user_df.shape)
    print("Head:\n", user_df.head())
    print("Info:")
    user_df.info()
    print("Describe:\n", user_df.describe())

    # 2. Get Keystroke Features DataFrame
    features_df = get_keystroke_features_dataframe(user_df)
    print("\n--- Keystroke Features DataFrame ---")
    print("Shape:", features_df.shape)
    print("Head:\n", features_df.head())
    print("Info:")
    features_df.info()
    print("Describe:\n", features_df.describe())

    # 3. Get Merged and Cleaned DataFrame
    final_df = get_merged_and_cleaned_dataframe(user_df, features_df)
    print("\n--- Final Merged and Cleaned DataFrame ---")
    print("Shape:", final_df.shape)
    print("Head:\n", final_df.head())
    print("Info:")
    final_df.info()
    print("Describe:\n", final_df.describe())
    
    # Save the final_df to CSV for consistency with the original flow
    final_df.to_csv('final_data.csv', index=False)
    print("\nSaved final_data.csv")

    # 4. Run Classification Pipeline
    print("\n--- Running Classification Pipeline ---")
    results_df = run_classification_pipeline(final_df)
    print("\n--- Model Comparison Summary ---")
    print(results_df)

    print("\n--- Data Processing Pipeline Completed ---")

run_pipeline()