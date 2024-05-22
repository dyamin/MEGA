import os

import numpy as np
import pandas as pd

from src import config as global_config
from src.post_processing import config
from src.post_processing.business.AggregationLogic import _convert_raw_gaze_to_fit_movie_dimensions


def aggregate_fixations(subject_dir: str, show=False) -> pd.DataFrame:
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
        fixations_path = os.path.join(subdir_path, 'fixations')
        for file in os.listdir(fixations_path):
            # file looks like the following: mov1.pkl
            movie = file.split('.')[0]
            mov_path = os.path.join(fixations_path, file)
            rearranged_data = _rearrange_fixations_data(mov_path)

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
    aggregated_data = _convert_raw_gaze_to_fit_movie_dimensions(aggregated_data, global_config.gaze_X,
                                                                global_config.gaze_Y, video_dims,
                                                                global_config.ORIG_HORIZONTAL_RESULOTION_IN_PXL,
                                                                global_config.ORIG_VERTICAL_RESULOTION_IN_PXL)
    if show:
        print(aggregated_data.head())
        print(".\n.\n.")
        print(aggregated_data.tail())

    return aggregated_data


def _rearrange_fixations_data(path: str) -> pd.DataFrame:
    fixations = pd.read_pickle(path)

    if fixations.empty:
        # there was an error in the recording --> no data recorded --> return the empty DataFrame
        return fixations
    fixations.rename(
        columns={"x_pos": global_config.gaze_X, "y_pos": global_config.gaze_Y, "p_size": global_config.PUPIL},
        inplace=True)
    fixations.reset_index(inplace=True)
    return fixations.drop(columns=['name'])


"""
CALCULATES THE ENGBERT FIXATIONS OF ALL SUBJECTS-MOVIES-SESSIONS
[Based on an algorithm presented in Engbert & Kliegl, 2002, and Engbert & Megenthaler, 2006]

1. Calculate horizontal and vertical components of velocity (Vx, Vy)
    V_x = (X_n+2 + X_n+1 - X_n-1 - X_n-2) / (6* dt)   [and similarly for V_y]
2. Calculate the median for each velocity-component. Use the median to calculate median-based-variance.
    sig_x = (med(Vx^2) - med^2(Vx))^0.5
3. Compare each velocity-component to the size lambda*sig_x: if greater than threshold --> it's a saccade; else --> fixation
    in the paper, default lambda = 5
4. Indices where both X&Y velocity components are not saccadic, are Fixation-Candidates. Fixation-Candidates are grouped to a
Fixation-Epoch if the difference between 2 timestamps is lesser/equals the argument max_difference_within_epoch. Finally,
Fixation-Epochs are filtered out if they are of a duration lesser than min_fixation_duration.
    in the paper, epochs with less than 3ms between them are unified, thus default max_difference_within_epoch = 6;
    also, a minimal period for a fixation is 100ms, thus min_fixation_duration = 50.
5. Fixations from the same subject-session-movie are aranged in a single DataFrame, and all fixation DataFrames are aranged
together to a single final DataFrame.
6. If mark_fixations is True, the original decentralized_data is added with a new column 'is Fixation',
    marking True for fixation-related data-points.
    NOTE: THIS IS ADDING ANOTHER PASS ON THE WHOLE DATASET, MAKING THIS FUNCTION RUN MUCH SLOWER

$return => a DataFrame indexed by ['Subject', 'Session', 'Movie #', 'Fixation #']
            columns are the following fixation-parameters:
                Start_Time => first timestamp of the fixation
                Duration => number of timestamps in the fixation
                CoM_X, CoM_Y => Center of Mass, calculated as the mean of all gazes within the fixation
                StDev_X, StDev_Y => standard deviation of gaze within this fixation
                Mean_Pupil, StDev_Pupil => mean pupil radius & standard deviation within this fixation
"""


def calculate_engbert_fixations(gaze_data):
    from src.utils import get_subjects_sessions_movies

    INDEX_NAMES = [global_config.SUBJECT, global_config.SESSION, global_config.MOVIE, config.FIXATIONNUMBER]
    COLUMN_NAMES = {0: 'Start_Time', 1: 'Duration', 2: 'CoM_X', 3: 'StDev_X',
                    4: 'CoM_Y', 5: 'StDev_Y', 6: 'Mean_Pupil', 7: 'StDev_Pupil'}

    subject_IDs, _, _ = get_subjects_sessions_movies(gaze_data)
    fixation_by_subject = list()
    fixation_indices_by_subject = dict()
    for subject in subject_IDs:
        subject_gaze = gaze_data.xs((subject,), level=['Subject'])
        subject_fixations_df, subject_fixation_indices_dict_A, subject_fixation_indices_dict_B = _calculate_engbert_for_single_subject(
            subject_gaze, config.engberts_lambda, config.max_difference_within_fixation_epoch,
            config.min_fixation_duration)
        fixation_by_subject.append(subject_fixations_df)
        fixation_indices_by_subject[subject] = (subject_fixation_indices_dict_A, subject_fixation_indices_dict_B)

    if (config.should_mark_fixations_on_original_data):
        gaze_data.loc[:, 'is Fixation'] = False
        for subj in fixation_indices_by_subject.keys():
            for i, indices_dict in enumerate(fixation_indices_by_subject[subj]):
                session_id = 'Session A' if i == 0 else 'Session B'
                for movie in indices_dict.keys():
                    all_indices = set([idx for fix in indices_dict[movie] for idx in fix])
                    gaze_data.loc[(subj, session_id, movie), 'is Fixation'].loc[list(all_indices)] = True

    fixations_df = pd.concat(fixation_by_subject, keys=subject_IDs, names=INDEX_NAMES, ignore_index=False)
    fixations_df.rename(columns=COLUMN_NAMES, inplace=True)
    fixations_df['Start_Time'] = fixations_df['Start_Time'].apply(lambda x: int(x))
    fixations_df['Duration'] = fixations_df['Duration'].apply(lambda x: int(x))
    return fixations_df


def _calculate_engbert_for_single_subject(subject_DF, engbert_lambda, max_difference_within_epoch,
                                          min_fixation_duration):
    '''
    Returns:
        1. A DataFrame with all fixation-inforamtion of a given subject
        2. Two dict with per-movie indices of fixations (for later marking indices as fixations)
    '''
    Vx, Vy = __calculate_axial_velocities(subject_DF)
    x_candidates = __calculate_axial_fixation_candidate_indices(Vx, engbert_lambda)
    y_candidates = __calculate_axial_fixation_candidate_indices(Vy, engbert_lambda)
    fixation_candidates = np.intersect1d(x_candidates, y_candidates)
    indices_dict_A, indices_dict_B = _get_fixations_from_candidates(fixation_candidates,
                                                                    max_difference_within_epoch,
                                                                    min_fixation_duration)
    fixations_session_A = _calculate_fixations_for_session(indices_dict_A,
                                                           subject_DF.xs([global_config.SESSION_A], level=[0]))
    fixations_session_B = _calculate_fixations_for_session(indices_dict_B,
                                                           subject_DF.xs([global_config.SESSION_B], level=[0]))
    subject_fixations = pd.concat([fixations_session_A, fixations_session_B],
                                  keys=[global_config.SESSION_A, global_config.SESSION_B], ignore_index=False)
    return subject_fixations, indices_dict_A, indices_dict_B


def _get_fixations_from_candidates(candidate_indices,
                                   maximal_timestamp_difference_within_epochs,
                                   minimal_timestamp_duration_for_fixation):
    '''
    Returns two dictionaries (one foreach session) where each movie's fixation-indices are the values.
    Fixation-indices are calculated from fixation-candidates:
        1. Fixation candidates with less than maximal_timestamp_difference_within_epochs
            are grouped together to a single epoch
        2. Fixation Epochs with duration less than minimal_timestamp_duration_for_fixation
            are filtered out.
        3. The remaining indices are grouped by session-movie and put into a dict.
    '''
    import itertools
    indices_session_A = dict()
    indices_session_B = dict()
    for key, group in itertools.groupby(candidate_indices, key=lambda x: (x[0], x[1])):
        session, movie = key[0], key[1]
        session_movie_candidates = [g[2] for g in list(group)]
        indices_of_splitting = np.where(np.diff(session_movie_candidates) > maximal_timestamp_difference_within_epochs)[
                                   0] + 1
        session_movie_fixation_indices = [fix for fix in np.split(session_movie_candidates, indices_of_splitting)
                                          if len(fix) >= minimal_timestamp_duration_for_fixation]

        if (session == 'Session A'):
            indices_session_A[movie] = session_movie_fixation_indices
        else:
            indices_session_B[movie] = session_movie_fixation_indices
    return indices_session_A, indices_session_B


def _calculate_fixations_for_session(session_indices_dict, session_gaze_df):
    ''' Return a DataFrame for a single session, with fixation-params (CoM X, CoM Y, etc.) for each movie '''
    session_fixations_dict = dict()
    for movie in session_indices_dict.keys():
        session_movie_dict = dict()
        session_movie_raw_df = session_gaze_df.xs((movie,), level=[0])
        for i, fix in enumerate(session_indices_dict[movie]):
            session_movie_dict[i + 1] = [fix[0], fix[-1] - fix[0],
                                         session_movie_raw_df.loc[fix]['X_gaze'].mean(),
                                         session_movie_raw_df.loc[fix]['X_gaze'].std(),
                                         session_movie_raw_df.loc[fix]['Y_gaze'].mean(),
                                         session_movie_raw_df.loc[fix]['Y_gaze'].std(),
                                         session_movie_raw_df.loc[fix]['Pupil radius'].mean(),
                                         session_movie_raw_df.loc[fix]['Pupil radius'].std()]
        # DataFrame per session-movie:
        session_fixations_dict[movie] = pd.DataFrame.from_dict(session_movie_dict, orient='index')
    session_fixations_df = pd.concat(list(session_fixations_dict.values()),
                                     keys=list(session_fixations_dict.keys()),
                                     ignore_index=False)
    return session_fixations_df


def __calculate_axial_velocities(single_subject_DF):
    ''' Returns the derivative of the gaze along the X-axis and the Y-axis '''
    from src.utils import numerical_derivative
    x_gaze = single_subject_DF['X_gaze'].loc[~single_subject_DF['is Blink']] if (
            "is Blink" in single_subject_DF.columns) else single_subject_DF['X_gaze']
    Vx = numerical_derivative(x_gaze)
    y_gaze = single_subject_DF['Y_gaze'].loc[~single_subject_DF['is Blink']] if (
            "is Blink" in single_subject_DF.columns) else single_subject_DF['Y_gaze']
    Vy = numerical_derivative(y_gaze)
    return Vx, Vy


def __calculate_axial_fixation_candidate_indices(velocity_Series, engbert_lambda):
    ''' Returns a list of indices where the velocity is no more than *lambda* times median_std away from the median velocity '''
    median_velocity, median_std_velocity = np.nanmedian(velocity_Series), __median_based_std(velocity_Series)
    greater_than = velocity_Series[velocity_Series >= median_velocity - engbert_lambda * median_std_velocity].index
    lesser_than = velocity_Series[velocity_Series <= median_velocity + engbert_lambda * median_std_velocity].index
    return np.intersect1d(greater_than, lesser_than)


def __median_based_std(serie):
    ''' calculate the mediane based standard deviation: sig=(med(X^2) - (med(X))^2)^0.5 '''
    return np.power(max(0, np.nanmedian(np.power(serie, 2)) - np.power(np.nanmedian(serie), 2)), 0.5)
