import pandas as pd
import os


def get_keystroke_features_dataframe(user_df, data_directory="dataset/Archived Data/"):
    
    
    def create_features_for_user(user_key, data_directory):
        hold_times = []
        latency_times = []

        for filename in os.listdir(data_directory):
            if filename.endswith('.txt') and filename.startswith(user_key):
                filepath = os.path.join(data_directory, filename)
                try:
                    temp_df = pd.read_csv(
                        filepath,
                        sep='\t',
                        header=None,
                        usecols=range(8),
                        dtype=str, 
                        low_memory=False
                    )

                    temp_df.columns = [
                        'UserKey', 'Date', 'Timestamp', 'Hand', 'Hold Time', 'Direction', 'Latency Time', 'Flight Time'
                    ]

                    temp_df['Hold Time'] = pd.to_numeric(temp_df['Hold Time'].str.replace(',', '.'), errors='coerce')
                    temp_df['Latency Time'] = pd.to_numeric(temp_df['Latency Time'].str.replace(',', '.'), errors='coerce')
                    temp_df['Flight Time'] = pd.to_numeric(temp_df['Flight Time'].str.replace(',', '.'), errors='coerce')

                    hold_times.extend(temp_df['Hold Time'].dropna().tolist())
                    latency_times.extend(temp_df['Latency Time'].dropna().tolist())
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
        user_features = create_features_for_user(user_key, data_directory)
        if user_features:
            all_user_features.append(user_features)

    features_df = pd.DataFrame(all_user_features)
    return features_df
