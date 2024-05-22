import os

import numpy as np
import pandas as pd

from src import config as g_config
from src.post_processing.config import use_eyelink_parser
from src.pre_processing.utils import mean_or_single


def aggregate_data(rootdir: str, show=False) -> pd.DataFrame:
    '''
    Params: rootdir is an absolute path the the directory which contains all experimental data.
            video_dims_path
            show is a boolean flag the indicates if function should print status along the way
    Returns: Aggregated DataFrame of all data in rootdir

    This function iterates over all the files and aggregates all the data into one MultiLevel DataFrame,
    which will look like the following:

                                                                    Data
                                                       Pupil radius |  Measured Eye | X_gaze | Y_gaze
                                                      |--------------------------------|
    Subject | Session | Movie | Point in Time  |                                |
    ------------------------------------------------|                                |
                 |         |       |                |                                |
                 |         |       |                |                                |
                 |         |       |                |                                |
                 |         |       |                |                                |
                 |         |       |                |--------------------------------|

    *** Access to the data from now on could be done as the following:
        1. df[columns] to access data by the column (part of data's attributes)
        2. df.ax([index1, ... ,indexN]) to access data by the index ***

    Data API
    ----------------------------------------------------------------------
    rootdir.dir-> name1_A.dir -> (movie1.pkl, movie2.pkl, ..., movieN.pkl)
                  name1_B.dir -> (movie1.pkl, movie2.pkl, ..., movieN.pkl)
                  name2_A.dir -> (movie1.pkl, movie2.pkl, ..., movieN.pkl)
                  name2_B.dir -> (movie1.pkl, movie2.pkl, ..., movieN.pkl)
                  .
                  .
                  .
                  nameN_A.dir -> (movie1.pkl, movie2.pkl, ..., movieN.pkl)
                  nameN_B.dir -> (movie1.pkl, movie2.pkl, ..., movieN.pkl)
    ----------------------------------------------------------------------
    '''

    indexing_names = [g_config.SUBJECT, g_config.SESSION, g_config.MOVIE, g_config.TIMESTAMP]
    session_a = True  # indicated with session's data should be extracted in current iteration
    subjects_names = list()
    subjects_data = list()

    # walks over the directory tree, with root = rootdir
    # subdir - current dir
    # dirs - list of dirs in subdir
    # files - list of files in subdir
    for subdir, dirs, files in os.walk(rootdir):

        # continue if subdir ends with _A or _B, which are session directories
        if subdir.endswith('_A') or subdir.endswith('_B'):
            name, session = subdir.split(os.sep)[-1].split('_')

            if show:
                print("Subject {} in session {} raw data is being extracted".format(name, session))

            # initialize lists for movie names and extracted data for later aggregation
            tables, movies = list(), list()

            for file in files:
                # file looks like the following: mov1_data.pkl
                movie = file.split('_')[0]

                path = os.path.join(subdir, file)
                # path should look like the following: this\is\an\example\name_session\mov1_data.pkl

                rearranged_data = _rearrange_raw_data(path)

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
                session_a = True  # next session to be interated over should be A

            else:
                raise Exception("One of the following:\n\
                1. Invalid value for session: was not A or B.\n\
                2. Two A sessions or two B sessions were ordered in a row.\n\
                Please check your data")

    # aggregates data from all subjects to an Aggregated DataFrame
    aggregated_data = (pd.concat(subjects_data, keys=subjects_names, names=indexing_names))

    # fit the gaze values to each video's dimensions
    video_dims = pd.read_pickle(g_config.VIDEO_DIMS_FILE_PATH)
    aggregated_data = _convert_raw_gaze_to_fit_movie_dimensions(aggregated_data, g_config.gaze_X,
                                                                g_config.gaze_Y, video_dims,
                                                                g_config.ORIG_HORIZONTAL_RESULOTION_IN_PXL,
                                                                g_config.ORIG_VERTICAL_RESULOTION_IN_PXL)

    if show:
        print(aggregated_data.head())
        print(".\n.\n.")
        print(aggregated_data.tail())

    return aggregated_data


def reset_timestamp(raw_data):
    # Set the onset column to be the index, and its starting time to be 0
    onset = 'onset'
    baseline_time = raw_data[onset][0]
    raw_data[onset] = raw_data[onset].apply(lambda time: time - baseline_time)
    raw_data.set_index(onset, inplace=True)
    return raw_data


def fix_preprocessed_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    for col in ['onset', 'x_l', 'y_l', 'pup_l', 'x_r', 'y_r', 'pup_r']:
        if col not in raw_data.columns:
            raw_data[col] = np.nan

    raw_data[g_config.gaze_X] = raw_data[['x_l', 'x_r']].apply(mean_or_single, axis=1)
    raw_data[g_config.gaze_Y] = raw_data[['y_l', 'y_r']].apply(mean_or_single, axis=1)

    raw_data.loc[:, "pup_l"].loc[raw_data["pup_l"] < 1] = np.nan
    raw_data.loc[:, "pup_r"].loc[raw_data["pup_r"] < 1] = np.nan
    raw_data.loc[:, 'pup'] = raw_data[['pup_l', 'pup_r']].apply(mean_or_single, axis=1)

    return raw_data


def _rearrange_raw_data(path: str) -> pd.DataFrame:
    '''
    Params: path -> path the data file
    Returns: DataFrame of the modified table of data

    This function removes the index columns and the x_r/x_l, y_r/y_l columns, sets the onsets the be
    relatively to zero, and indexes the table by onset.
    In addition, combines the pup_r and pup_l to pup, and added a colums of Measured Pupil with values R\L

    Raw data API
    ------------------------------------------
    index
    onset - the time of the indexed measurment
    x_r \ x_l
    y_r \ y_l
    pup_r \ pup_l
    x_gaze
    y_gaze
    -----------------------------------------
    '''

    # path should look like the following: this\is\an\example\name_session\mov1_data.pkl
    raw_data = pd.read_pickle(path)
    entries_num = raw_data.shape[0]

    if raw_data.empty:
        # there was an error in the recording --> no data recorded --> return the empty DataFrame
        return raw_data

    if not use_eyelink_parser:
        raw_data = reset_timestamp(raw_data)

    if g_config.POPULATION == 'no_nap':
        raw_data = fix_preprocessed_data(raw_data)
        measured_eye = 'right' if raw_data['pup_l'].isnull().all() else 'left' if raw_data[
            'pup_r'].isnull().all() else 'both'
        raw_data.rename(columns={'pup': "Pupil radius"}, inplace=True)
        raw_data.insert(1, "Measured Eye", [measured_eye.upper()] * entries_num
                        , allow_duplicates=True)  # 1 for column's location
    else:
        measured_eye = 'right' if raw_data['pup_l'].isnull().all() else 'left' if raw_data[
            'pup_r'].isnull().all() else 'both'
        raw_data.rename(columns={'pup': "Pupil radius", "x_gaze": "X_gaze", "y_gaze": "Y_gaze"}, inplace=True)
        raw_data.insert(1, "Measured Eye", [measured_eye.upper()] * entries_num
                        , allow_duplicates=True)  # 1 for column's location

    return raw_data[['Measured Eye', 'Pupil radius', 'X_gaze', 'Y_gaze']]


def _convert_raw_gaze_to_fit_movie_dimensions(aggregated_gaze, x_gaze, y_gaze, video_dims, original_width,
                                              original_height):
    movie_dataframes = dict()
    movies = video_dims.index
    for mov in movies:
        # get the gaze data for the current movie if exists, by filtering the level 'Movie' in the index
        movie_gaze = aggregated_gaze.loc[aggregated_gaze.index.get_level_values('Movie') == mov]
        if movie_gaze.empty:
            continue
        movie_gaze = movie_gaze.xs(mov, level='Movie')
        movie_width, movie_height = video_dims.loc[mov, ['Width', 'Height']]
        movie_gaze.loc[:, x_gaze] = movie_gaze.loc[:, x_gaze] * (movie_width / original_width)
        movie_gaze.loc[:, y_gaze] = movie_gaze.loc[:, y_gaze] * (movie_height / original_height)
        # changes negative X_gaze & Y_gaze values to None
        movie_gaze.loc[:, x_gaze] = movie_gaze.loc[:, x_gaze].apply(
            lambda x: np.nan if (x < 0) or (x > movie_width) else x)
        movie_gaze.loc[:, y_gaze] = movie_gaze.loc[:, y_gaze].apply(
            lambda y: np.nan if (y < 0) or (y > movie_height) else y)
        movie_dataframes[mov] = movie_gaze

    result = pd.concat(movie_dataframes.values(), keys=movie_dataframes.keys(),
                       names=[g_config.MOVIE, g_config.SUBJECT, g_config.SESSION, g_config.TIMESTAMP])

    return result.reorder_levels([g_config.SUBJECT, g_config.SESSION, g_config.MOVIE, g_config.TIMESTAMP])
