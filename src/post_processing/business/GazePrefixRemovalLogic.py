import pandas as pd

from src import config as g_config
from src.post_processing import config

'''
    ALL movies start with a large pupil size, initial grow and then very fast decrease in size,
        before growing again to a relatively constant size. Therefore we want to remove the oversize-prefix.

    Based on each subject's raw gaze data, we:
        1. calculate the median gaze over all movies & sessions
        2. calculate the temporal derivative of the median pupil-size of this subject
        3. find this subject's cutoff index, i.e. the 1st index where deriv is positive AFTER seeing a negative derivative
        4. remove all of the subject's data PRIOR to their cutoff-index
        5. return a new DataFrame containing all of the subjects' prefix-free gaze data

    Why median and not mean?
        We've seen that within-subject mean-gaze has very large standard deviation, whereas median has much lower median-based std
'''


def remove_prefix_from_raw_data(raw_data):
    from src.utils import get_subjects_sessions_movies

    index_names = raw_data.index.names
    subject_IDs, session_IDs, movies_IDs = get_subjects_sessions_movies(raw_data)
    subjects_cutoff_indices = _find_cutoff_indices_for_all_subjects(raw_data, subject_IDs, movies_IDs, "both",
                                                                    config.num_arguments_for_numerical_derivation,
                                                                    config.epsilon)

    all_subjects_data = list()
    sessionA = True  # indicates which session's data should be extracted in current iteration

    for subj_ID in subject_IDs:
        for ses_ID in session_IDs:
            # print("subj: {} ses: {} bool: {}\n".format(subj_ID, ses_ID, sessionA))
            subj_movies_data = list()

            for mov_ID in movies_IDs:
                movie = raw_data.xs((subj_ID, ses_ID, mov_ID),
                                    level=[g_config.SUBJECT, g_config.SESSION, g_config.MOVIE])
                subj_movies_data.append(movie[subjects_cutoff_indices[(subj_ID, mov_ID)]:])

            # aggregates data from all movies to a single Session-DataFrame:
            if ses_ID == 'Session A' and sessionA:
                sessionA_data = pd.concat(subj_movies_data, keys=movies_IDs, names=index_names[2:], ignore_index=False)
                sessionA = False  # next session to be iterated over is not A

            elif ses_ID == 'Session B' and not sessionA:
                sessionB_data = pd.concat(subj_movies_data, keys=movies_IDs, names=index_names[2:], ignore_index=False)
                sessionA = True  # next session to be iterated over is A

            else:
                print("Subect {} \tSession {} \tBool Marker: {}".format(subj_ID, ses_ID, sessionA))
                raise Exception("One of the following:\n\
                \t1. Invalid value for session: was not A or B.\n\
                \t2. Two A sessions or two B sessions were ordered in a row.\n\
                Please check your data")

        # finished both sessions --> aggregate data from both sessions to a single
        # Subject-DataFrame & add to all_subjects_data list:
        all_subjects_data.append(pd.concat([sessionA_data, sessionB_data],
                                           keys=['Session A', 'Session B'],
                                           names=index_names[1:]))

    # finished iterating over all subjects --> aggregates data from all subjects to a single Aggregated DataFrame
    aggregated_data = (pd.concat(all_subjects_data,
                                 keys=subject_IDs,
                                 names=index_names))
    return aggregated_data


def _find_cutoff_indices_for_all_subjects(raw_data, subject_IDs, movies_IDs, session="both", deriv_args_num=3,
                                          epsilon=0.1):
    '''
    For each subjects, finds their unique prefix-cutoff-index for their gaze DFs
        prefix-cutoff-index is the first timestamp where the median pupil-size (across all movies & sessions) has
        a (roughly) zero derivative AFTER having a negative+positive one.
       reason for this: each movie starts with abnormally large pupil size that quickly reduces and then
        rises again to "normal" values. We're finding the index of the 1st "normal" value
    '''
    cutoff_indices = dict()
    for subj in subject_IDs:
        for mov in movies_IDs:
            subj_mov_data = raw_data.xs((subj, mov), level=[g_config.SUBJECT, g_config.MOVIE])
            cutoff_indices[(subj, mov)] = _find_cutoff_index_for_subject(subj_mov_data, session, deriv_args_num,
                                                                         epsilon)
    return cutoff_indices


def _find_cutoff_index_for_subject(subject_raw_gaze, session="both", deriv_args_num=3, epsilon=0.1):
    '''
    Cutoff Index is the 1st index where pupil size is roughly constant,  AFTER displaying initial decrease+increase in size
    '''
    from src.utils import numerical_derivative
    median_pupil_series = subject_raw_gaze['Pupil radius'].groupby(level=g_config.TIMESTAMP).median()
    pupil_deriv = numerical_derivative(median_pupil_series, deriv_args_num)
    cutoff_index = _find_const_index(pupil_deriv, epsilon)
    return cutoff_index


def _find_const_index(pupil_deriv_serie, epsilon=0.1):
    '''
    Removes indices of initial decrease+increase in pupil size (neg derivative then pos derivative)
    Then finds index where the derivative is (roughly) zero, i.e the pupil size is constant, and returns said index
    '''
    first_neg_index = pupil_deriv_serie[pupil_deriv_serie < 0].index.min()
    first_rising_index = pupil_deriv_serie[first_neg_index:][pupil_deriv_serie[first_neg_index:] > 0].index.min()
    pos_zero_deriv_index = pupil_deriv_serie[first_rising_index:][
        pupil_deriv_serie[first_rising_index:] <= 0 + epsilon].index.min()
    neg_zero_deriv_index = pupil_deriv_serie[first_rising_index:][
        pupil_deriv_serie[first_rising_index:] >= 0 - epsilon].index.min()
    return min(pos_zero_deriv_index, neg_zero_deriv_index) // 2
