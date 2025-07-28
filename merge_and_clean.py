import pandas as pd
from keystroke_parser import features_df
from user_parser import user_df

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

final_df.to_csv('final_data.csv', index=False)
# print(final_df.head())
print(final_df.info())
print(final_df.describe())