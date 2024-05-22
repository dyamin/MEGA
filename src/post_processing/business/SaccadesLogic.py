import itertools
import os
import time

import numpy as np
import pandas as pd

from src import config as global_config
from src.post_processing import config
from src.post_processing.business.AggregationLogic import _convert_raw_gaze_to_fit_movie_dimensions

"""
CALCULATES SACCADE STATISTICS FOR ALL SUBJECTS-SESSIONS-MOVIES

This uses the aggregated gaze DataFrame AFTER adding the 'is Blink' & 'is Fixation' columns to it.
    1. Consecutive indices that are not marked as blink/fixation are grouped together as a saccade-epoch.
    2. Two epochs with less than 'max_difference_within_saccade' between them are joined together.
    3. Epochs with duration of less than 'min_saccade_duration' (in miliseconds) are filtered out.
    4. Finally, the following stats are extracted for each saccade:
        start_time (ms) ; duration (ms) ; pupil radius: mean, stdev, median; horizontal & vertical distance_from_roi (px) & direction
    NOTE - distances are only a hueristic, since those are calculated by (end-position - start-position)

RUNTIME ~3 HOURS for 30 subjects
"""


def aggregate_saccades(subject_dir: str, show=False) -> pd.DataFrame:
    indexing_names = [global_config.SUBJECT, global_config.SESSION, global_config.MOVIE, global_config.TIMESTAMP]
    subjects_names = list()
    subjects_data = list()
    session_a = True  # indicated with session's data should be extracted in current iteration

    # walks over the subfolders in the directory tree, with root = subject_dir
    for subdir in os.listdir(subject_dir):
        subdir_path = os.path.join(subject_dir, subdir)
        name, session = subdir.split(os.sep)[-1].split('_')

        if show:
            print("Subject {} in session {} blink data is being extracted".format(name, session))

        # initialize lists for movie names and extracted data for later aggregation
        tables, movies = list(), list()

        # walks over all the files in the current subfolder
        saccades_path = os.path.join(subdir_path, 'saccades')
        for file in os.listdir(saccades_path):
            # file looks like the following: mov1.pkl
            movie = file.split('.')[0]
            mov_path = os.path.join(saccades_path, file)
            rearranged_data = _rearrange_saccades_data(mov_path)

            if not rearranged_data.empty:
                movies.append(movie)
                tables.append(
                    rearranged_data)  # ignore all-null inputs (maybe there was an issue with the recording?)

        # aggregates data from all the movies to a Session DataFrame,
        # and after two sessions, aggregates data from both of them to a Subject DataFrame
        if session == 'A' and session_a:
            sessionA_data = pd.concat(tables, keys=movies, names=indexing_names[2:], ignore_index=False)
            subjects_names.append(name)  # name should be added once, hence there are two session per name
            session_a = False  # next session to be integrated over should not be A

        elif session == 'B' and not session_a:
            sessionB_data = pd.concat(tables, keys=movies, names=indexing_names[2:], ignore_index=False)
            subjects_data.append(pd.concat([sessionA_data, sessionB_data], keys=['Session A', 'Session B'],
                                           names=indexing_names[1:]))
            session_a = True  # next session to be integrated over should be A

        else:
            raise Exception("One of the following:\n\
             1. Invalid value for session: was not A or B.\nCl\
             2. Two A sessions or two B sessions were ordered in a row.\n\
             Please check your data")

    # aggregates data from all subjects to an Aggregated DataFrame
    aggregated_data = (pd.concat(subjects_data, keys=subjects_names, names=indexing_names))

    # fit the gaze values to each video's dimensions
    video_dims = pd.read_pickle(global_config.VIDEO_DIMS_FILE_PATH)
    aggregated_data = _convert_raw_gaze_to_fit_movie_dimensions(aggregated_data, global_config.X_START,
                                                                global_config.Y_START, video_dims,
                                                                global_config.ORIG_HORIZONTAL_RESULOTION_IN_PXL,
                                                                global_config.ORIG_VERTICAL_RESULOTION_IN_PXL)
    aggregated_data = _convert_raw_gaze_to_fit_movie_dimensions(aggregated_data, global_config.X_END,
                                                                global_config.Y_END, video_dims,
                                                                global_config.ORIG_HORIZONTAL_RESULOTION_IN_PXL,
                                                                global_config.ORIG_VERTICAL_RESULOTION_IN_PXL)
    if show:
        print(aggregated_data.head())
        print(".\n.\n.")
        print(aggregated_data.tail())

    return aggregated_data


def _rearrange_saccades_data(path: str) -> pd.DataFrame:
    saccades = pd.read_pickle(path)

    if saccades.empty:
        # there was an error in the recording --> no data recorded --> return the empty DataFrame
        return saccades

    saccades.reset_index(inplace=True)
    return saccades.drop(columns=['name'])


def calculate_saccades(gaze_df):
    t0 = time.perf_counter()

    saccades_dict = dict()
    saccade_indices = gaze_df[
        ~(gaze_df['is Fixation'])].index  # find indices of neither fixation nor blink
    t1 = time.perf_counter()
    #     print(f'\tfinished finding saccade indices in {t1-t0:.2f} sec')

    cnt = 0
    for key, group in itertools.groupby(saccade_indices, key=lambda x: (x[0], x[1], x[2])):
        t2 = time.perf_counter()

        subject, session, movie = key
        df = gaze_df.xs((subject, session, movie), level=['Subject', 'Session', 'Movie']
                        ).drop(columns=['Measured Eye', 'is Fixation'])
        all_timestamps_list = list(df.index)

        saccade_timestamps = [tup[3] for tup in group]
        saccade_start_and_end_timestamps = [(x[0], x[-1]) for x in
                                            np.split(saccade_timestamps,
                                                     np.where(np.diff(
                                                         saccade_timestamps) > config.max_difference_within_saccade)[
                                                         0] + 1)
                                            if len(x) >= config.min_saccade_duration // 2]
        tmp_dict = dict()
        for i, tup in enumerate(saccade_start_and_end_timestamps):
            start_idx, end_idx = all_timestamps_list.index(tup[0]), all_timestamps_list.index(tup[1])
            partial_df = df[start_idx:end_idx]

            # saccade measures:
            duration = tup[1] - tup[0]  # in ms
            start_X, start_Y = partial_df.iloc[0]['X_gaze'], partial_df.iloc[0]['Y_gaze']

            total_distance = _calculate_total_distance_of_saccade(partial_df)
            final_horiz_distance = abs(partial_df.iloc[-1, 1] - partial_df.iloc[0, 1])  # over the X axis
            final_horiz_direction = 'R' if partial_df.iloc[-1, 1] - partial_df.iloc[0, 1] >= 0 else 'L'
            final_vertic_distance = abs(partial_df.iloc[-1, 2] - partial_df.iloc[0, 2])  # over the Y axis
            final_vertic_direction = 'U' if partial_df.iloc[-1, 2] - partial_df.iloc[0, 2] <= 0 else 'D'

            mean_pupil = partial_df['Pupil radius'].mean()
            stdev_pupil = partial_df['Pupil radius'].std()
            median_pupil = partial_df['Pupil radius'].median()

            tmp_dict[i] = (tup[0], duration, start_X, start_Y, total_distance, total_distance / duration,
                           final_horiz_distance, final_horiz_direction, final_vertic_distance, final_vertic_direction,
                           mean_pupil, stdev_pupil, median_pupil)

        # create DataFrame for this subject-session-movie:
        saccades_dict[key] = pd.DataFrame.from_dict(tmp_dict, orient='index')

        cnt += 1
        t3 = time.perf_counter()
        # print(f'\tfinished working on saccade #{cnt} in {t3 - t2:.2f} sec')

    t4 = time.perf_counter()

    # concatenate all DataFrames to a single DF & rename columns
    INDEX_NAMES = [global_config.SUBJECT, global_config.SESSION, global_config.MOVIE, config.SACCADENUMBER]
    COLUMN_NAMES = {0: 'Start_Time', 1: 'Duration', 2: 'Start X', 3: 'Start Y', 4: 'Total Distance', 5: 'Velocity',
                    6: 'Final Horizontal Distance', 7: 'Final Horizontal Direction',
                    8: 'Final Vertical Distance', 9: 'Final Vertical Direction',
                    10: 'Mean_Pupil', 11: 'StDev_Pupil', 12: 'Median_Pupil'}

    saccades_df = pd.concat(saccades_dict.values(),
                            keys=saccades_dict.keys(),
                            names=INDEX_NAMES
                            ).rename(columns=COLUMN_NAMES)
    saccades_df['Start_Time'] = saccades_df['Start_Time'].apply(lambda x: int(x))
    saccades_df['Duration'] = saccades_df['Duration'].apply(lambda x: int(x))

    t5 = time.perf_counter()
    #     print(f'\tfinished creating final df in {t3-t2:.2f} sec')

    return saccades_df


def _calculate_total_distance_of_saccade(partial_df):
    diff = (partial_df[['X_gaze', 'Y_gaze']] - partial_df[['X_gaze', 'Y_gaze']].shift(1))
    return diff.pow(2).sum(axis=1).apply(np.sqrt).sum()
