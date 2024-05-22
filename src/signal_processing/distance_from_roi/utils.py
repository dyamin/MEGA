from src.signal_processing import config

from src.signal_processing.utils import get_all_subject_data_df


def get_filtered_roi_df():
    df = get_all_subject_data_df()
    filtered_df = df.loc[df.index.get_level_values('Movie').isin(config.valid_movies)]
    filtered_df.index = filtered_df.index.droplevel(-1)
    return filtered_df['Distance']


def get_filtered_avg_mem_df():
    df = get_all_subject_data_df()
    filtered_df = df.loc[df.index.get_level_values('Movie').isin(config.valid_movies)]
    filtered_df = filtered_df.xs('Session B', level='Session')
    filtered_df = filtered_df.index.to_frame(index=False).drop('TimeStamp', 1).drop_duplicates()
    filtered_df.set_index(['Subject', 'Movie'], inplace=True)
    filtered_df = filtered_df.groupby(level='Movie').mean()
    return filtered_df


def get_filtered_mem_df():
    df = get_all_subject_data_df()
    filtered_df = df.loc[df.index.get_level_values('Movie').isin(config.valid_movies)]
    filtered_df = filtered_df.xs('Session B', level='Session')
    filtered_df = filtered_df.index.to_frame(index=False).drop('TimeStamp', 1).drop_duplicates()
    filtered_df.set_index(['Subject', 'Movie'], inplace=True)
    return filtered_df
