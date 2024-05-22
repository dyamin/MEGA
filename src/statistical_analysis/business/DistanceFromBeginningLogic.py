import pandas as pd

from src import config as global_config


def calculate_aoi_viewing_time_proportion(session_data: pd.Series, session_name: str):
    videos_dims = pd.read_pickle(global_config.VIDEO_DIMS_FILE_PATH)
    avg_dict = dict()
    movies = session_data.index.unique(level=global_config.MOVIE)
    subjects = session_data.index.unique(level=global_config.SUBJECT)
    for subj in subjects:
        subj_A = session_data.xs(subj, level=global_config.SUBJECT)
        for mov in movies:
            mov_subj = subj_A[subj_A.index.get_level_values(global_config.MOVIE) == mov]

            movie_width, movie_height = videos_dims.loc[mov, ['Width', 'Height']]
            aoi = 14  # max(movie_width, movie_height) * 0.2

            mov_subj_len = len(mov_subj)
            if mov_subj_len > 0:
                roi = mov_subj.loc[mov_subj['DVA'] < aoi]
                avg_dict[(subj, mov)] = (len(roi) / mov_subj_len) * 100

    return dictionary_to_df(avg_dict, session_name)


def calculate_average_for_trial(session_data: pd.Series, session_name: str):
    avg_dict = dict()
    movies = session_data.index.unique(level=global_config.MOVIE)
    subjects = session_data.index.unique(level=global_config.SUBJECT)
    for subj in subjects:
        subj_data = session_data.xs(subj, level=global_config.SUBJECT)
        for mov in movies:
            mov_subj_data = subj_data[subj_data.index.get_level_values(global_config.MOVIE) == mov]
            if not mov_subj_data.empty: avg_dict[(subj, mov)] = mov_subj_data.mean()

    return dictionary_to_df(avg_dict, session_name)


def dictionary_to_df(avg_dict: dict, session_name: str):
    df = pd.DataFrame(columns=['Session', 'Subject', 'Movie', global_config.DVA, global_config.SQRT_DVA, global_config.DISTANCE])
    for (subj_key, mov_key), values in avg_dict.items():
        new_row_df = pd.DataFrame(
            {'Session': [session_name], 'Subject': [subj_key], 'Movie': [mov_key],
             global_config.DVA: [values[global_config.DVA]], global_config.SQRT_DVA: [values[global_config.SQRT_DVA]], global_config.DISTANCE: [values[global_config.DISTANCE]]})
        df = pd.concat([df, new_row_df], ignore_index=True, axis=0)
    return df
