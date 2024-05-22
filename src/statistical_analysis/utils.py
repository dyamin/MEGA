import os

import pandas as pd

import src.config as config


def get_filtered_all_subject_data_df():
    df = get_all_valid_subject_data_df()
    filtered_df = df.loc[df.index.get_level_values(config.MOVIE).isin(config.valid_movies)]
    filtered_df = filtered_df[filtered_df[[config.gaze_X, config.gaze_Y]]]
    return filtered_df


def get_all_valid_subject_data_df():
    df = pd.read_pickle(os.path.join(config.data_dir, config.RAW_GAZE_FILE))
    valid_df = df[df.notnull().all(1)]
    return valid_df


def get_aggregated_roi_df():
    return pd.read_pickle(os.path.join(config.rois_dir, config.AGGRGATED_ROI_FILE))


def get_event_median_times():
    df = get_aggregated_roi_df()
    filtered_df = df.loc[df.index.get_level_values('Movie').isin(config.valid_movies)]
    return filtered_df['t_median']


def get_event_max_times():
    df = get_aggregated_roi_df()
    filtered_df = df.loc[df.index.get_level_values('Movie').isin(config.valid_movies)]
    return filtered_df['t_max']


def get_filtered_avg_mem_df():
    df = get_filtered_all_subject_data_df()
    filtered_df = df.loc[df.index.get_level_values('Movie').isin(config.valid_movies)]
    filtered_df = filtered_df.xs('Session B', level='Session')
    filtered_df = filtered_df.index.to_frame(index=False).drop('TimeStamp', 1).drop_duplicates()
    filtered_df.set_index(['Subject', 'Movie'], inplace=True)
    filtered_df = filtered_df.groupby(level='Movie').mean()
    return filtered_df


def get_filtered_mem_df():
    df = get_filtered_all_subject_data_df()
    filtered_df = df.loc[df.index.get_level_values('Movie').isin(config.valid_movies)]
    filtered_df = filtered_df.xs('Session B', level='Session')
    filtered_df = filtered_df.index.to_frame(index=False).drop('TimeStamp', 1).drop_duplicates()
    filtered_df.set_index(['Subject', 'Movie'], inplace=True)
    return filtered_df
