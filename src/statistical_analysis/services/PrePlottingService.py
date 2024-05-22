import os
import pickle

import pandas as pd

from src import config as g_config
from src.statistical_analysis import config
from src.statistical_analysis.business.DataMatchingLogic import match_subjects_and_movies
from src.statistical_analysis.business.DataSplittingLogic import split
from src.statistical_analysis.business.DistanceFocusingLogic import recenter_distances, cut_tail, cut_prefix
# noinspection SpellCheckingInspection
from src.statistical_analysis.business.DistanceFromBeginningLogic import calculate_average_for_trial
from src.statistical_analysis.business.NormalizationLogic import mega_score_normalization
from src.statistical_analysis.business.StatsLogic import calculate_cohens_s


def aoi_duration_from_beginning(rois: pd.DataFrame, distances: pd.Series, remember_label, forgot_label):
    sesA, sesB = split(distances, 'ses')
    sesA, sesB = cut_tail(sesA, rois), cut_tail(sesB, rois)
    sesA, sesB = cut_prefix(sesA, rois, config.STARTING_TIME), cut_prefix(sesB, rois, config.STARTING_TIME)
    sesB_remembered, sesB_forgot = split(sesB, 'mem')
    sesA_remembered = match_subjects_and_movies(sesB_remembered,
                                                sesA)  # extract A-distances for subject who remembered in B
    sesA_forgot = match_subjects_and_movies(sesB_forgot,
                                            sesA)  # extract A-distances for subject who did not remember in B

    ses_A_remembered = calculate_average_for_trial(sesA_remembered, g_config.SESSION_A)
    ses_B_remembered = calculate_average_for_trial(sesB_remembered, g_config.SESSION_B)
    remembered_to_plot = pd.concat([ses_A_remembered, ses_B_remembered], ignore_index=True, axis=0)
    ses_A_forgot = calculate_average_for_trial(sesA_forgot, g_config.SESSION_A)
    ses_B_forgot = calculate_average_for_trial(sesB_forgot, g_config.SESSION_B)
    forgot_to_plot = pd.concat([ses_A_forgot, ses_B_forgot], ignore_index=True, axis=0)

    remembered_to_plot.to_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                              f"{remember_label}.pkl"))
    forgot_to_plot.to_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                          f"{forgot_label}.pkl"))


def average_per_trial(rois: pd.DataFrame, distances: pd.Series, remember_label, forgot_label):
    sesA, sesB = split(distances, 'ses')
    sesA, sesB = cut_tail(sesA, rois), cut_tail(sesB, rois)
    sesA, sesB = cut_prefix(sesA, rois, config.STARTING_TIME), cut_prefix(sesB, rois, config.STARTING_TIME)

    sesB_remembered, sesB_forgot = split(sesB, 'mem')
    sesA_remembered = match_subjects_and_movies(sesB_remembered,
                                                sesA)  # extract A-distances for subject who remembered in B
    sesA_forgot = match_subjects_and_movies(sesB_forgot,
                                            sesA)  # extract A-distances for subject who did not remember in B

    ses_A_remembered = calculate_average_for_trial(sesA_remembered, g_config.SESSION_A)
    ses_B_remembered = calculate_average_for_trial(sesB_remembered, g_config.SESSION_B)
    average_remembered = pd.concat([ses_A_remembered, ses_B_remembered], ignore_index=True, axis=0)
    ses_A_forgot = calculate_average_for_trial(sesA_forgot, g_config.SESSION_A)
    ses_B_forgot = calculate_average_for_trial(sesB_forgot, g_config.SESSION_B)
    average_forgot = pd.concat([ses_A_forgot, ses_B_forgot], ignore_index=True, axis=0)

    if config.should_normalize:
        average_remembered = mega_score_normalization(average_remembered, column=g_config.DVA)
        average_forgot = mega_score_normalization(average_forgot, column=g_config.DVA)

        average_remembered.to_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                                  f"normalized_{remember_label}.pkl"))
        average_forgot.to_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                              f"normalized_{forgot_label}.pkl"))
    else:
        average_remembered.to_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                                  f"{remember_label}.pkl"))
        average_forgot.to_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                              f"{forgot_label}.pkl"))


def BRemember_BForgot(rois: pd.DataFrame, distances: pd.Series, label):
    sesA, sesB = split(distances, 'ses')
    sesA, sesB = cut_tail(sesA, rois), cut_tail(sesB, rois)
    sesA, sesB = cut_prefix(sesA, rois, config.STARTING_TIME), cut_prefix(sesB, rois, config.STARTING_TIME)
    sesB_remembered, sesB_forgot = split(sesB, 'mem')

    B_remembered = calculate_average_for_trial(sesB_remembered, config.memory_labels[0])
    B_forgot = calculate_average_for_trial(sesB_forgot, config.memory_labels[1])

    if config.should_normalize:
        sesA_remembered, sesA_forgot = match_subjects_and_movies(sesB_remembered, sesA), match_subjects_and_movies(
            sesB_forgot, sesA)
        A_remembered = calculate_average_for_trial(sesA_remembered, config.memory_labels[2])
        A_forgot = calculate_average_for_trial(sesA_forgot, config.memory_labels[3])

        B_remembered = mega_score_normalization(B_remembered, A_remembered, config.memory_labels[0],
                                                config.memory_labels[2])
        B_remembered = B_remembered[B_remembered[g_config.SESSION] == config.memory_labels[0]]
        B_forgot = mega_score_normalization(B_forgot, A_forgot, config.memory_labels[1],
                                            config.memory_labels[3])
        B_forgot = B_forgot[B_forgot[g_config.SESSION] == config.memory_labels[1]]

    average_remembered = pd.concat([B_remembered, B_forgot], ignore_index=True, axis=0)
    average_remembered.to_pickle(os.path.join(g_config.statistical_analysis_resource_dir, f"normalized_{label}.pkl"))


def find_optimal_params(roi, distance, label):
    sesA_remembered, sesA_not_remembered, sesB_remembered, sesB_not_remembered = split(distance, 'ses', 'mem')

    cohens_d_dict = {}
    max_c, max_i = 0, 0
    for i in range(0, 5000, 20):
        curr_sesA, curr_sesB = cut_prefix(sesA_remembered, roi, i), cut_prefix(sesB_remembered, roi, i)
        ses_A_remembered = calculate_average_for_trial(curr_sesA, g_config.SESSION_A)
        ses_B_remembered = calculate_average_for_trial(curr_sesB, g_config.SESSION_B)
        average_from_beginning_remembered = pd.concat([ses_A_remembered, ses_B_remembered], ignore_index=True, axis=0)

        dfg_mean = \
            average_from_beginning_remembered.groupby([g_config.SESSION, g_config.SUBJECT])[
                'Result'].mean().reset_index() \
                if config.should_aggregate_by_subject \
                else average_from_beginning_remembered.groupby([g_config.SESSION, g_config.SUBJECT, g_config.MOVIE])[
                'Result'].mean().reset_index()

        cohens_d = calculate_cohens_s(dfg_mean[dfg_mean['Session'] == g_config.SESSION_A]['Result'],
                                      dfg_mean[dfg_mean['Session'] == g_config.SESSION_B]['Result'])
        cohens_d_dict[i] = cohens_d
        print(f"for {i}: the cohen's d is {cohens_d}")
        if max_c < cohens_d:
            max_c = cohens_d
            max_i = i

    print(f"Using {max_i} gives the best cohen's d: {max_c}")
    with open(os.path.join(g_config.statistical_analysis_resource_dir, f"{label}.pkl"), 'wb') as f:
        pickle.dump(cohens_d_dict, f)


def find_best_movies(roi, distance):
    sesA_remembered, sesA_not_remembered, sesB_remembered, sesB_not_remembered = split(distance, 'ses', 'mem')

    curr_sesA, curr_sesB = cut_prefix(sesA_remembered, roi, config.STARTING_TIME), cut_prefix(sesB_remembered, roi,
                                                                                              config.STARTING_TIME)
    average_from_beginning_remembered = calculate_average_for_trial(curr_sesA, curr_sesB)

    dfg_mean = average_from_beginning_remembered.groupby([g_config.SESSION, g_config.MOVIE])['Result'].mean()

    session_a = dfg_mean.xs('A', level=g_config.SESSION)
    session_b = dfg_mean.xs('B', level=g_config.SESSION)
    # session_b.reset_index(inplace=True, drop=True)

    sub = session_a.sub(session_b, axis=0)
    print(f"The best movies: {sub.sort_values()}")


def sessions(rois: pd.DataFrame, uncentered_distances: pd.Series, label):
    distances = recenter_distances(uncentered_distances, rois)
    session_distances = split(distances, 'ses')

    with open(os.path.join(g_config.statistical_analysis_resource_dir, f"{label}.pkl"), 'wb') as f:
        pickle.dump(session_distances, f)


def memory(rois: pd.DataFrame, uncentered_distances: pd.Series, label):
    distances = recenter_distances(uncentered_distances, rois)
    sesA_remembered, sesA_not_remembered, sesB_remembered, sesB_not_remembered = split(distances, 'ses', 'mem')
    distances_to_plot = [sesB_remembered]

    if config.should_plot_not_remembered:
        distances_to_plot += [sesB_not_remembered]
    if config.should_plot_session_A:
        distances_to_plot += [sesA_remembered]
        if config.should_plot_not_remembered:
            distances_to_plot += [sesA_not_remembered]

    with open(os.path.join(g_config.statistical_analysis_resource_dir, f"{label}.pkl"), 'wb') as f:
        pickle.dump(distances_to_plot, f)


def remembered_within_subject(rois: pd.DataFrame, uncentered_distances: pd.Series, label):
    distances = recenter_distances(uncentered_distances, rois)
    sesA_remembered, sesA_not_remembered, sesB_remembered, sesB_not_remembered = split(distances, 'ses', 'mem')

    distances_to_plot = [sesB_remembered, sesA_remembered, sesB_not_remembered, sesA_not_remembered]

    with open(os.path.join(g_config.statistical_analysis_resource_dir, f"{label}.pkl"), 'wb') as f:
        pickle.dump(distances_to_plot, f)
