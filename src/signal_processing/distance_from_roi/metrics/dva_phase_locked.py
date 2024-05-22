# All necessary IMPORTS - RUN THIS FIRST!

from src.signal_processing.distance_from_roi.services.DataSplittingService import DataSplittingService

from src.signal_processing import config
from src.signal_processing.distance_from_roi.services.CenteredDistancesPlottingService import \
    CenteredDistancesPlottingService
from src.signal_processing.distance_from_roi.services.DistancePlottingService import DistancePlottingService
from src.signal_processing.utils import get_aggregated_roi_df
from src.signal_processing.utils import get_all_subject_data_df
from src.utils import cut_series

raw_gaze = get_all_subject_data_df()
raw_roi = get_aggregated_roi_df()

movies_to_drop_roi = set(raw_roi.index) - set(config.valid_movies)
movies_to_drop_gaze = movies_to_drop_roi.union(set([f'mov{idx}' for idx in range(81, 102)]))

relevant_movies_roi = raw_roi.drop(movies_to_drop_roi)
relevant_moviess_gaze = raw_gaze.drop(index=movies_to_drop_gaze, level='Movie')

cdps = CenteredDistancesPlottingService(relevant_moviess_gaze['Distance_from_RoI'], relevant_movies_roi,
                                        r'C:\Users\user\PycharmProjects\gaze\Gaze\src\signal_processing\distance_from_roi\figures\Centered_Distances-Selected_Movies')

sesA = cdps.centered_distances.xs('Session A', level='Session')
sesB = cdps.centered_distances.xs('Session B', level='Session')

# A vs. B
# calculator_a_b = DistanceDifferenceCalculatorService(sesA, sesB)
# diffs_a_b = calculator_a_b.calculate_differences(-1500, 0)
# mean_diff_per_subject0 = diffs_a_b.groupby(level=['Subject', 'TimeStamp']).mean()
# sum_per_subj0 = mean_diff_per_subject0.groupby(level='Subject').sum()

splitter = DataSplittingService(cdps.centered_distances)
sesA_remembered, sesA_forgot, sesB_remembered, sesB_forgot = splitter.split_by_session_and_memory()

# A vs. B remembered
sp_remembered = DataSplittingService(sesB_remembered)
rem__sesA = sp_remembered.extract_matching_subjects_and_movies(sesA).droplevel('Memory')
rem__sesA.index.names = sesB_remembered.index.names

# calc1 = DistanceDifferenceCalculatorService(rem__sesA, sesB_remembered)
# diffs1 = calc1.calculate_differences(-1500, 0)
# mean_diff_per_subject1 = diffs1.groupby(level=['Subject', 'TimeStamp']).mean()
# sum_per_subj1 = mean_diff_per_subject1.groupby(level='Subject').sum()

# A vs. B forgot
sp_forgot = DataSplittingService(sesB_forgot)
forgot_sesA = sp_forgot.extract_matching_subjects_and_movies(sesA).droplevel('Memory')
forgot_sesA.index.names = sesB_forgot.index.names

# calc2 = DistanceDifferenceCalculatorService(forgot_sesA, sesB_forgot)
# diffs2 = calc2.calculate_differences(-1500, 0)
# mean_diff_per_subject2 = diffs2.groupby(level=['Subject', 'TimeStamp']).mean()
# sum_per_subj2 = mean_diff_per_subject2.groupby(level='Subject').sum()
#
# # look on confidence - WIERD!
# data_mean = diffs_a_b.mean(level=['Memory', 'Subject', 'TimeStamp'])
# sum_per_subj0 = data_mean.groupby(level=['Subject', 'Memory']).sum()
# sum_per_subj0.mean(level='Memory')
#
# cut_sesB = cut_series(sesB, -1500, 0)
# cut_sesB_remembered = cut_series(sesB_remembered, -1500, 0)
# cut_sesB_not_remembered = cut_series(sesB_forgot, -1500, 0)
#
# remembered_subj_mov = set(
#     zip(cut_sesB_remembered.index.get_level_values(0), cut_sesB_remembered.index.get_level_values(1)))
# not_remembered_subj_mov = set(
#     zip(cut_sesB_not_remembered.index.get_level_values(0), cut_sesB_not_remembered.index.get_level_values(1)))
# all_subj_mov = remembered_subj_mov.union(not_remembered_subj_mov)
# amount_to_choose = min(len(remembered_subj_mov), len(not_remembered_subj_mov))


# compare session B's memory-report to themselves in session A

# ttest_results = ttest_rel(sum_per_subj2.sort_index(), sum_per_subj1.sort_index())

rem_ses_b = cut_series(sesB_remembered.groupby(level='TimeStamp').mean(), -4500, 3500)
rem_ses_b_zoom = cut_series(sesB_remembered.groupby(level='TimeStamp').mean(), -1500, 0)

rem_ses_a = cut_series(rem__sesA.groupby(level='TimeStamp').mean(), -4500, 3500)
rem_ses_a_zoom = cut_series(rem__sesA.groupby(level='TimeStamp').mean(), -1500, 0)

forgot_ses_b = cut_series(sesB_forgot.groupby(level='TimeStamp').mean(), -4500, 3500)
forgot_ses_b_zoom = cut_series(sesB_forgot.groupby(level='TimeStamp').mean(), -1500, 0)

forgot_ses_a = cut_series(forgot_sesA.groupby(level='TimeStamp').mean(), -4500, 3500)
forgot_ses_a_zoom = cut_series(forgot_sesA.groupby(level='TimeStamp').mean(), -1500, 0)

plotter = DistancePlottingService(
    r'C:\Users\user\PycharmProjects\gaze\Gaze\src\signal_processing\distance_from_roi\figures')
fig1 = plotter.plot_distances_impl(distances=[rem_ses_a, rem_ses_b, rem_ses_a_zoom, rem_ses_b_zoom],
                                   labels=['Session A', 'Session B Remembered', '', ''],
                                   line_colors=['r', 'g', 'orange', 'orange'],
                                   roi_time=0, roi_duration=1500, x_label='time (ms)', y_label='distance (px)')
fig2 = plotter.plot_distances_impl(distances=[forgot_ses_a, forgot_ses_b, forgot_ses_a_zoom, forgot_ses_b_zoom],
                                   labels=['Session A', 'Session B Forgot', '', ''],
                                   line_colors=['r', 'g', 'orange', 'orange'],
                                   roi_time=0, roi_duration=1500, x_label='time (ms)', y_label='distance (px)')

figures_dir = r'C:\Users\user\PycharmProjects\gaze\Gaze\src\signal_processing\distance_from_roi\figures'
plotter.save_figure(fig1, 'Session_B_Remembered', figures_dir)
plotter.save_figure(fig2, 'Session_B_Forgot', figures_dir)
