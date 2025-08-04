import pandas as pd
def get_merged_and_cleaned_dataframe(user_df, features_df):
    final_df = pd.merge(user_df, features_df, on='UserKey')

    bool_cols = ['Parkinsons', 'Tremors', 'Levodopa', 'DA', 'MAOB', 'Other']
    for col in bool_cols:
        final_df[col] = final_df[col].map({'True': True, 'False': False, 'Yes': True, 'No': False})

    final_df['BirthYear'] = pd.to_numeric(final_df['BirthYear'], errors='coerce')
    final_df['DiagnosisYear'] = pd.to_numeric(final_df['DiagnosisYear'], errors='coerce')

    final_df['UPDRS'] = final_df['UPDRS'].replace("Don't know", None)
    final_df['Impact'] = final_df['Impact'].replace("", None)
    final_df['Impact'] = final_df['Impact'].map({'Mild': 1, 'Medium': 2, 'Severe': 3})

    current_year = 2019
    final_df['Age'] = current_year - final_df['BirthYear']
    final_df['DiseaseDuration'] = current_year - final_df['DiagnosisYear']
    
    # NEW
    if final_df['Age'].isnull().any():
        median_age = final_df['Age'].median()
        final_df['Age'].fillna(median_age, inplace=True)
        print(f"Imputed missing 'Age' values with median: {median_age:.1f}")
    
    # NEW
    final_df['std_hold'].fillna(0, inplace=True)
    final_df['std_latency'].fillna(0, inplace=True)
    
    return final_df