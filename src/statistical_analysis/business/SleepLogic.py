import os
import pickle

import pandas as pd

from src import config


def score_by_sleep_duration(nap, sleep_count):
    nap = nap.join(sleep_count.rename('SleepEfficiency')).fillna(0)
    return nap


def filter_to_subjects_that_nap(data):
    directory_str = r'C:\Users\user\PycharmProjects\gaze\Gaze\resources\nap\scoring\output\proc'
    with open(os.path.join(directory_str, f"sleep_efficiency_dict.pkl"), 'rb') as f:
        sleep_count_dict = pickle.load(f)
    sleep_count = pd.Series(list(sleep_count_dict.values()), index=sleep_count_dict.keys())
    data = data.join(sleep_count.rename('SleepCount'), on=[config.SUBJECT])
    data = data.loc[(data['SleepCount'] > 0) & (data['SleepCount'] != 'nan')]
    return data
