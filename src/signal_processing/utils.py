import os

import pandas as pd

import src.config as config


def get_all_subject_data_df():
    df = pd.read_pickle(os.path.join(config.data_dir, config.RAW_GAZE_FILE))
    filtered_df = df.loc[df.index.get_level_values(config.MOVIE).isin(config.valid_movies)]
    filtered_df = filtered_df[filtered_df[[config.gaze_X, config.gaze_Y]].notnull().all(1)]
    return filtered_df


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


def get_all_blinks_df():
    return pd.read_pickle(os.path.join(config.data_dir, "all_subject_blinks.pkl"))


def get_all_fixations_df():
    return pd.read_pickle(os.path.join(config.data_dir, "all_subject_fixations.pkl"))
