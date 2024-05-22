import os

import numpy as np
import pandas as pd

from src import config as global_config
from src.post_processing import config
from src.utils import numerical_derivative, get_subjects_sessions_movies


def aggregate_blinks(subject_dir: str, show=False) -> pd.DataFrame:
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
        blinks_path = os.path.join(subdir_path, 'blinks')
        for file in os.listdir(blinks_path):
            # file looks like the following: mov1.pkl
            movie = file.split('.')[0]
            mov_path = os.path.join(blinks_path, file)
            rearranged_data = _rearrange_blinks_data(mov_path)

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

    if show:
        print(aggregated_data.head())
        print(".\n.\n.")
        print(aggregated_data.tail())

    return aggregated_data


def _rearrange_blinks_data(path: str) -> pd.DataFrame:
    blinks_data = pd.read_pickle(path)

    if blinks_data.empty:
        # there was an error in the recording --> no data recorded --> return the empty DataFrame
        return blinks_data

    blinks_data.reset_index(inplace=True)
    blinks_data.rename(
        columns={'onset': 'Start_Time', 'eye': "Measured Eye", "last_onset": "End_Time", "duration": "Duration"},
        inplace=True)
    return blinks_data.drop(columns=['name'])


'''
BLINK CANDIDATE: A datapoint is considered a blink candidate if the pupil-size on that data-point has a temporal-derivative
    that is more than 2 standard deviations apart from the mean temporal-derivative.
BLINK EPOCH: A group of three or more consecutive blink-candidates is a blink epoch.
        Two epochs with only 2 datapoints separating them are merged as a single epoch.
BLINK: A single epoch is considered a blink if it is longer than a pre-defined length (i.e "min_blink_time")
'''


def mark_blinks(gaze_df):
    index_names = gaze_df.index.names
    subject_IDs, session_IDs, movies_IDs = get_subjects_sessions_movies(gaze_df)
    all_subjects_data = list()

    for subj in subject_IDs:
        # for each subject, find their blink epochs:
        subj_df = gaze_df.xs((subj,), level=['Subject'])
        pupil_serie = subj_df['Pupil radius']
        pupil_deriv = numerical_derivative(pupil_serie, config.num_arguments_for_numerical_derivation)
        blink_candidates = _find_blink_indices(pupil_deriv)
        blink_indices = _ignore_short_blink_epochs(blink_candidates, config.min_blink_time)

        # now mark those epochs to a new DF:
        flat_df = subj_df.reset_index()
        flat_df.loc[:, "is Blink"] = False
        flat_df.loc[list(blink_indices), "is Blink"] = True
        new_subj_df = flat_df.set_index(keys=index_names[1:])
        all_subjects_data.append(new_subj_df)

    # finished iterating over all subjects, create a unified DF:
    final_df = (pd.concat(all_subjects_data, keys=subject_IDs, names=index_names))
    if config.ignore_blinks_in_movie_start_and_end:
        final_df = _ignore_init_and_fin_indices(final_df)
    return final_df


def extract_blinks_df(gaze_df):
    '''Returns a DataFrame with each blink start-time & duration (in datapoints, not miliseconds)'''
    INDEX_NAMES = ['Subject', 'Session', 'Movie']
    all_blinks = list()
    blinks_df = gaze_df[gaze_df['is Blink']]
    subjects = set(blinks_df.index.get_level_values(INDEX_NAMES[0]))
    for subj in subjects:
        blinks_ses_A = dict()
        blinks_ses_B = dict()
        subj_df = blinks_df.xs((subj,), level=[INDEX_NAMES[0]])
        sessions = set(subj_df.index.get_level_values(INDEX_NAMES[1]))

        for ses in sessions:
            subj_ses_df = subj_df.xs((ses,), level=[INDEX_NAMES[1]])
            movies = set(subj_ses_df.index.get_level_values(INDEX_NAMES[2]))

            for mov in movies:
                d = dict()
                specific_df = subj_ses_df.xs((mov,), level=[INDEX_NAMES[2]])
                blink_idxs = np.split(specific_df.index, np.where(np.diff(specific_df.index) > 2)[0] + 1)
                for i, blink in enumerate(blink_idxs):
                    if (len(blink) < 2):
                        continue
                    d[i + 1] = [blink[0], blink[-1] - blink[0]]

                if not (len(d)):
                    continue  # this movie has no blinks. move on
                subj_ses_mov_blinks = pd.DataFrame.from_dict(d, orient='index')
                if (ses == 'Session A'):
                    blinks_ses_A[mov] = subj_ses_mov_blinks
                else:
                    blinks_ses_B[mov] = subj_ses_mov_blinks

        sessionA_df = pd.concat(list(blinks_ses_A.values()), keys=movies, ignore_index=False) if len(
            blinks_ses_A) else pd.DataFrame()
        sessionB_df = pd.concat(list(blinks_ses_B.values()), keys=movies, ignore_index=False) if len(
            blinks_ses_B) else pd.DataFrame()
        if (not sessionA_df.empty or not sessionB_df.empty):
            subj_df = pd.concat([sessionA_df, sessionB_df], keys=sessions, ignore_index=False)
            all_blinks.append(subj_df)
    if (len(all_blinks) == 0):
        return pd.DataFrame(dtype=int)
    final_df = pd.concat(all_blinks, keys=subjects)
    if (not final_df.empty):
        multi_index = pd.MultiIndex.from_tuples(final_df.index)
        multi_index.names = [global_config.SUBJECT, global_config.SESSION, global_config.MOVIE, config.BLINKNUMBER]
        final_df.index = multi_index
        final_df.rename(columns={0: 'Start_Time', 1: 'Duration'}, inplace=True)
    return final_df.astype(int)


def _find_blink_indices(pupil_deriv) -> set:
    '''
    Finds subsequent indices where the deriv is (2) stdev away from the mean.
    If (3) or more subsequent indices fit the criteria --> this is a blink-epoch.
    Two blink-epochs with less than 3 datapoints between them are joined together to a single blink.
    Result is a set of indices that need to be removed from the data
    '''

    # find indices where deriv is far from the mean, and group consecutive indices together (ignore sequence of less than 3):
    deriv_mean, deriv_stdev = pupil_deriv.mean(), pupil_deriv.std()
    candidate_idx = np.sort(
        np.concatenate([np.where(pupil_deriv > (deriv_mean + config.blink_speed_stdev_threshold * deriv_stdev))[0],
                        np.where(pupil_deriv < (deriv_mean - config.blink_speed_stdev_threshold * deriv_stdev))[0]]))

    '''
    A more strict option for choosing candidates:
    candidate_idx = np.sort(np.concatenate([np.where(pupil_deriv > (deriv_mean+3*deriv_stdev))[0],
                                            np.where(pupil_deriv < (deriv_mean-3*deriv_stdev))[0],
                                            np.where(pd.isnull(pupil_deriv))[0]]))
    '''

    epoch_list = [x for x in np.split(candidate_idx,
                                      np.where(np.diff(candidate_idx) >= config.min_samples_per_blink_epoch)[0] + 1)]

    # join epochs with less than <2> samples between them and put everything in a single set:
    res, m = set(), 0
    for i in range(len(epoch_list)):
        if epoch_list[i][0] - m > config.samples_between_blink_epochs:
            res = res.union(set(epoch_list[i]))
        else:
            tmp = set([j for j in range(m, epoch_list[i][-1] + 1)])
            res = res.union(tmp)
        m = epoch_list[i][-1]
    return res


def _ignore_short_blink_epochs(blink_candidates, min_blink_time):
    ''' Removes blink-epochs of length lower than min_blink_time '''
    blink_idx_list = sorted(blink_candidates)
    epoch = list()
    for idx in blink_idx_list:
        if (len(epoch) == 0):
            epoch.append(idx)
        elif (idx <= epoch[-1] + 2):
            epoch.append(idx)
        else:
            if len(epoch) < min_blink_time:
                # removes too short epochs
                blink_candidates.difference_update(set(epoch))
            epoch = list()
    return blink_candidates


def _ignore_init_and_fin_indices(marked_blinks_df):
    '''
    markes the start- and end-indices of each movie as non-blink, unless they are part of a longer blink
        (i.e. the indices right after/before them are True)
    '''
    grouped = marked_blinks_df.groupby(level=[0, 1, 2])

    s1 = grouped['is Blink'].transform('nth', 2).fillna(False)
    index_head = marked_blinks_df.index[~s1].intersection(grouped.head(2).index)

    s2 = grouped['is Blink'].transform('nth', -3).fillna(False)
    index_tail = marked_blinks_df.index[~s2].intersection(grouped.tail(2).index)

    union = index_head.union(index_tail)
    marked_blinks_df.loc[union, 'is Blink'] = False
    return marked_blinks_df
