from src.signal_processing import config

from src.signal_processing.utils import get_all_subject_data_df


def get_filtered_gaze_tuple_position_df():
    df = get_all_subject_data_df()
    filtered_df = df.loc[df.index.get_level_values('Movie').isin(config.valid_movies)]
    filtered_df.index = filtered_df.index.droplevel(-1)
    filtered_df['XY_gaze'] = list(zip(filtered_df.X_gaze, filtered_df.Y_gaze))
    return filtered_df['XY_gaze']


def get_filtered_gaze_position_df():
    df = get_all_subject_data_df()
    filtered_df = df.loc[df.index.get_level_values('Movie').isin(config.valid_movies)]
    filtered_df.index = filtered_df.index.droplevel(-1)
    return filtered_df[['X_gaze', 'Y_gaze']]


def get_filtered_roi_df():
    df = get_all_subject_data_df()
    filtered_df = df.loc[df.index.get_level_values('Movie').isin(config.valid_movies)]
    filtered_df.index = filtered_df.index.droplevel(-1)
    return filtered_df['Distance_from_RoI']
