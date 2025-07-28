import pandas as pd
from user_parser import user_df
import os


data_dir = "dataset/Archived Data/"

def create_features_for_user(user_key, data_directory):
    hold_times = []
    latency_times = []

    for filename in os.listdir(data_directory):
        if filename.endswith('.txt') and filename.startswith(user_key):
            filepath = os.path.join(data_directory, filename)
            try:
                temp_df = pd.read_csv(filepath, sep='\t', header=None, usecols=range(8))

                temp_df.columns = [
                    'UserKey', 'Date', 'Timestamp', 'Hand', 'Hold Time', 'Direction', 'Latency Time', 'Flight Time'
                ]

                hold_times.extend(pd.to_numeric(temp_df['Hold Time'], errors='coerce').dropna().tolist())
                latency_times.extend(pd.to_numeric(temp_df['Latency Time'], errors='coerce').dropna().tolist())
            except Exception as e:
                print(f"Error {e}")
                continue
    if not hold_times or not latency_times:
        return None
    
    features = {
        'UserKey': user_key,
        'mean_hold': pd.Series(hold_times).mean(),
        'std_hold': pd.Series(hold_times).std(),
        'mean_latency': pd.Series(latency_times).mean(),
        'std_latency': pd.Series(latency_times).std(),
        'keystroke_count': len(hold_times)
    }
    
    return features
                

all_user_features = []

all_users = user_df['UserKey'].tolist()

for i, user_key in enumerate(all_users):
    user_features = create_features_for_user(user_key, data_dir)
    if user_features:
        all_user_features.append(user_features)

features_df = pd.DataFrame(all_user_features)
print('Features DataFrame:')
print(features_df.head())
