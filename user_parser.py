import pandas as pd
import os


def parse_user_info(filepath):
    user_data = {}
    with open(filepath, 'r') as f:
        lines = f.read().strip().split('\n')
        for line in lines:
            key, value = line.split(':', 1)
            user_data[key.strip()] = value.strip()
    return user_data


def get_user_dataframe(users_dir="dataset/Archived Users/"):
    user_list = []
    for filename in os.listdir(users_dir):
        if filename.endswith('.txt') and filename.startswith('User_'):
            filepath = os.path.join(users_dir, filename)
            user_key = filename.replace('User_', '').replace('.txt', '')
            user_info = parse_user_info(filepath)
            user_info['UserKey'] = user_key
            user_list.append(user_info)
    user_df = pd.DataFrame(user_list)
    return user_df