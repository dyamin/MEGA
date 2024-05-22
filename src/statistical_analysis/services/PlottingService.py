import os
import pickle

import matplotlib.pyplot as plt
import pandas
import pandas as pd

from src import config as g_config
from src.config import get_project_root
from src.statistical_analysis import config
# noinspection SpellCheckingInspection
from src.statistical_analysis.business.DistancePlottingLogic import plot_distances, save_figure, plot_average_per_trial, \
    plot_two_populations, plot_average_per_trial_with_nap, plot_bootstrap_dist_aggregated_by_subject, \
    plot_two_populations_with_correlation


def plot_averaged_sessions(data, suptitle, suptitle_filename, labels, x_label, feature='Result'):
    if isinstance(data, list):
        if config.should_normalize:
            data[0] = f"normalized_{data[0]}"
            data[1] = f"normalized_{data[1]}"
        df1 = pd.read_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                          f"{data[0]}.pkl"))
        df2 = pd.read_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                          f"{data[1]}.pkl"))
        df = pandas.concat([df1, df2])
    else:
        if config.should_normalize:
            data = f"normalized_{data}"
        df = pd.read_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                         f"{data}.pkl"))

    if g_config.POPULATION == 'nap':
        directory_str = r'C:\Users\user\PycharmProjects\gaze\Gaze\resources\nap\scoring\output\proc'
        with open(os.path.join(directory_str, f"sleep_efficiency_dict.pkl"), 'rb') as f:
            sleep_count_dict = pickle.load(f)
        sleep_count = pd.Series(list(sleep_count_dict.values()), index=sleep_count_dict.keys())
        return plot_average_per_trial_with_nap(nap=df,
                                               feature=feature,
                                               sleep_count=sleep_count,
                                               labels=labels,
                                               filename=data,
                                               x_label=x_label,
                                               y_label=suptitle,
                                               directory=suptitle_filename)
    else:
        return plot_average_per_trial(df=df, feature=feature, labels=labels,
                                      filename=data,
                                      x_label=x_label,
                                      y_label=suptitle,
                                      directory=suptitle_filename)


def plot_bootstrap_dist(data, suptitle, suptitle_filename, x_label, feature='Result'):
    if isinstance(data, list):
        df1 = pd.read_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                          f"{data[0]}.pkl"))
        df2 = pd.read_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                          f"{data[1]}.pkl"))
        df = pandas.concat([df1, df2])

        norm_df1 = pd.read_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                               f"normalized_{data[0]}.pkl"))
        norm_df2 = pd.read_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                               f"normalized_{data[1]}.pkl"))
        norm_df = pandas.concat([norm_df1, norm_df2])
    else:
        df = pd.read_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                         f"{data}.pkl"))
        norm_df = pd.read_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                              f"normalized_{data}.pkl"))

    return plot_bootstrap_dist_aggregated_by_subject(df=df, observed_effect=norm_df, feature=feature,
                                                     filename=data,
                                                     x_label=x_label,
                                                     y_label=suptitle,
                                                     directory=suptitle_filename)


def plot_nap_vs_no_nap(data, suptitle, suptitle_filename, labels, x_label):
    if isinstance(data, list):
        if config.should_normalize:
            data[0] = f"normalized_{data[0]}"
            data[1] = f"normalized_{data[1]}"
        no_nap_1 = pd.read_pickle(os.path.join(g_config.no_nap_statistical_analysis_resource_dir,
                                               f"{data[0]}.pkl"))
        no_nap_2 = pd.read_pickle(os.path.join(g_config.no_nap_statistical_analysis_resource_dir,
                                               f"{data[1]}.pkl"))
        no_nap_to_plot = pandas.concat([no_nap_1, no_nap_2])

        nap1 = pd.read_pickle(os.path.join(g_config.nap_statistical_analysis_resource_dir,
                                           f"{data[0]}.pkl"))
        nap2 = pd.read_pickle(os.path.join(g_config.nap_statistical_analysis_resource_dir,
                                           f"{data[1]}.pkl"))
        nap_to_plot = pandas.concat([nap1, nap2])
    else:
        if config.should_normalize:
            data = f"normalized_{data}"
        no_nap_to_plot = pd.read_pickle(os.path.join(g_config.no_nap_statistical_analysis_resource_dir,
                                                     f"{data}.pkl"))
        nap_to_plot = pd.read_pickle(os.path.join(g_config.nap_statistical_analysis_resource_dir,
                                                  f"{data}.pkl"))

    directory_str = r'C:\Users\user\PycharmProjects\gaze\Gaze\resources\nap\scoring\output\proc'
    with open(os.path.join(directory_str, f"sleep_efficiency_dict.pkl"), 'rb') as f:
        sleep_count_dict = pickle.load(f)
    sleep_count = pd.Series(list(sleep_count_dict.values()), index=sleep_count_dict.keys())

    return plot_two_populations(control_group=no_nap_to_plot,
                                control_group_label='Wake',
                                main_group=nap_to_plot,
                                main_group_label='Nap',
                                sleep_count=sleep_count,
                                labels=labels,
                                filename=suptitle_filename,
                                x_label=x_label,
                                y_label=suptitle,
                                directory=suptitle_filename)


def plot_elderly_vs_mci_ad(data, suptitle, suptitle_filename, labels, x_label):
    if isinstance(data, list):
        if config.should_normalize:
            data[0] = f"normalized_{data[0]}"
            data[1] = f"normalized_{data[1]}"
        elderly_df1 = pd.read_pickle(os.path.join(g_config.elderly_statistical_analysis_resource_dir,
                                                  f"{data[0]}.pkl"))
        elderly_df2 = pd.read_pickle(os.path.join(g_config.elderly_statistical_analysis_resource_dir,
                                                  f"{data[1]}.pkl"))
        elderly_to_plot = pandas.concat([elderly_df1, elderly_df2])

        mci_ad_df1 = pd.read_pickle(os.path.join(g_config.mci_ad_statistical_analysis_resource_dir,
                                                 f"{data[0]}.pkl"))
        mci_ad_df2 = pd.read_pickle(os.path.join(g_config.mci_ad_statistical_analysis_resource_dir,
                                                 f"{data[1]}.pkl"))
        mci_ad_to_plot = pandas.concat([mci_ad_df1, mci_ad_df2])
    else:
        if config.should_normalize:
            data = f"normalized_{data}"
        elderly_to_plot = pd.read_pickle(os.path.join(g_config.elderly_statistical_analysis_resource_dir,
                                                      f"{data}.pkl"))
        mci_ad_to_plot = pd.read_pickle(os.path.join(g_config.mci_ad_statistical_analysis_resource_dir,
                                                     f"{data}.pkl"))

    return plot_two_populations(control_group=elderly_to_plot,
                                control_group_label='Elderly',
                                main_group=mci_ad_to_plot,
                                main_group_label='MCI/AD',
                                sleep_count=None,
                                labels=labels,
                                filename=suptitle_filename,
                                x_label=x_label,
                                y_label=suptitle,
                                directory=suptitle_filename)


def plot_elderly_mci_ad_correlation(data, suptitle, suptitle_filename, x_label):
    # Read memory performance data
    memory_performance = pd.read_pickle(
        os.path.join(get_project_root(), "resources", 'elderly', "statistical_analysis", "memory_performance.pkl"))

    if isinstance(data, list):
        if config.should_normalize:
            data[0] = f"normalized_{data[0]}"
            data[1] = f"normalized_{data[1]}"
        elderly_df1 = pd.read_pickle(os.path.join(g_config.elderly_statistical_analysis_resource_dir,
                                                  f"{data[0]}.pkl"))
        elderly_df2 = pd.read_pickle(os.path.join(g_config.elderly_statistical_analysis_resource_dir,
                                                  f"{data[1]}.pkl"))
        elderly_to_plot = pandas.concat([elderly_df1, elderly_df2])

        mci_ad_df1 = pd.read_pickle(os.path.join(g_config.mci_ad_statistical_analysis_resource_dir,
                                                 f"{data[0]}.pkl"))
        mci_ad_df2 = pd.read_pickle(os.path.join(g_config.mci_ad_statistical_analysis_resource_dir,
                                                 f"{data[1]}.pkl"))
        mci_ad_to_plot = pandas.concat([mci_ad_df1, mci_ad_df2])
    else:
        if config.should_normalize:
            data = f"normalized_{data}"
        elderly_to_plot = pd.read_pickle(os.path.join(g_config.elderly_statistical_analysis_resource_dir,
                                                      f"{data}.pkl"))
        mci_ad_to_plot = pd.read_pickle(os.path.join(g_config.mci_ad_statistical_analysis_resource_dir,
                                                     f"{data}.pkl"))

    return plot_two_populations_with_correlation(control_group=elderly_to_plot,
                                                 control_group_label='Elderly',
                                                 main_group=mci_ad_to_plot,
                                                 main_group_label='MCI/AD',
                                                 memory_performance=memory_performance,
                                                 filename=suptitle_filename,
                                                 x_label=x_label,
                                                 y_label=suptitle,
                                                 directory=suptitle_filename)


def plot_optimal_starting_time(label):
    with open(os.path.join(g_config.statistical_analysis_resource_dir, f"{label}.pkl"), 'rb') as f:
        data = pickle.load(f)

    x, y = zip(*sorted(data.items()))
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(x, y)
    plt.show()

    save_figure(fig, label, "Optimal_params")


def plot_memory_within(rois: pd.DataFrame, label):
    roi_duration = rois[config.T_STDEV][0] if rois.shape[0] == 1 else config.DEFAULT_ROI_DURATION

    labels = config.memory_comparison_labels
    labels = [labels['BR'], labels['AR'], labels['BF'], labels['AF']]
    colors = [config.SESSION_B_COLOR1, config.SESSION_A_COLOR1, config.SESSION_B_COLOR2, config.SESSION_A_COLOR2]

    with open(os.path.join(g_config.statistical_analysis_resource_dir, f"{label}.pkl"), 'rb') as f:
        distances_to_plot = pickle.load(f)

    errors = _get_errors(distances_to_plot)
    return plot_distances(distances=distances_to_plot, labels=labels, line_colors=colors,
                          roi_time=config.DEFAULT_ROI_TIME, roi_duration=roi_duration,
                          errors=errors, filename=label, should_add_sem=False,
                          ax_title=config.memory_comparison_filename, directory='rememberedwithinsubject')


def plot_memory_within(rois: pd.DataFrame, label):
    roi_duration = rois[config.T_STDEV][0] if rois.shape[0] == 1 else config.DEFAULT_ROI_DURATION

    labels = config.memory_comparison_labels
    labels = [labels['BR'], labels['AR'], labels['BF'], labels['AF']]
    colors = [config.SESSION_B_COLOR1, config.SESSION_A_COLOR1, config.SESSION_B_COLOR2, config.SESSION_A_COLOR2]

    with open(os.path.join(g_config.statistical_analysis_resource_dir, f"{label}.pkl"), 'rb') as f:
        distances_to_plot = pickle.load(f)

    errors = _get_errors(distances_to_plot)
    return plot_distances(distances=distances_to_plot, labels=labels, line_colors=colors,
                          roi_time=config.DEFAULT_ROI_TIME, roi_duration=roi_duration,
                          errors=errors, filename=label, should_add_sem=False,
                          ax_title=config.memory_comparison_filename, directory='rememberedwithinsubject')


def plot_memory_with_sem_shade(rois: pd.DataFrame, label):
    roi_duration = rois[config.T_STDEV][0] if rois.shape[0] == 1 else config.DEFAULT_ROI_DURATION

    labels = config.memory_comparison_labels
    labels = [labels['BR'], labels['AR']]
    colors = [config.SESSION_B_COLOR1, config.SESSION_A_COLOR1]

    with open(os.path.join(g_config.statistical_analysis_resource_dir, f"{label}.pkl"), 'rb') as f:
        distances_to_plot = pickle.load(f)[:2]

    errors = _get_errors(distances_to_plot)
    return plot_distances(distances=distances_to_plot, labels=labels, line_colors=colors,
                          roi_time=config.DEFAULT_ROI_TIME, roi_duration=roi_duration,
                          errors=errors, filename=f'{label}_with_sem_shade', should_add_sem=True,
                          ax_title=config.memory_comparison_filename, directory='rememberedwithinsubject')


def plot_sessions(rois: pd.DataFrame, label):
    roi_duration = rois[config.T_STDEV][0] if rois.shape[0] == 1 else config.DEFAULT_ROI_DURATION

    with open(os.path.join(g_config.statistical_analysis_resource_dir, f"{label}.pkl"), 'rb') as f:
        session_distances = pickle.load(f)

    errors = _get_errors(session_distances)
    return plot_distances(distances=session_distances, labels=[g_config.SESSION_A, g_config.SESSION_B],
                          line_colors=[config.SESSION_A_COLOR1, config.SESSION_B_COLOR1],
                          roi_time=config.DEFAULT_ROI_TIME, roi_duration=roi_duration,
                          errors=errors, filename=label, directory='sessions', should_add_sem=config.ADD_SEM_SHADE,
                          ax_title=config.session_subtitle)


def plot_memory(rois: pd.DataFrame, label):
    roi_duration = rois[config.T_STDEV][0] if rois.shape[0] == 1 else config.DEFAULT_ROI_DURATION

    labels = [config.memory_labels[0]]
    colors = [config.SESSION_B_COLOR1]

    if config.should_plot_not_remembered:
        labels += [config.memory_labels[1]]
        colors += [config.SESSION_B_COLOR2]
    if config.should_plot_session_A:
        labels += [config.memory_labels[2]]
        colors += [config.SESSION_A_COLOR1]
        if config.should_plot_not_remembered:
            labels += [config.memory_labels[3]]
            colors += [config.SESSION_A_COLOR2]

    with open(os.path.join(g_config.statistical_analysis_resource_dir, f"{label}.pkl"), 'rb') as f:
        distances_to_plot = pickle.load(f)

    errors = _get_errors(distances_to_plot)
    return plot_distances(distances=distances_to_plot, labels=labels, line_colors=colors,
                          roi_time=config.DEFAULT_ROI_TIME, roi_duration=roi_duration,
                          errors=errors, filename=label, directory='memory', should_add_sem=True,
                          ax_title=config.memory_subtitle)


def _get_errors(distances_to_plot: list):
    if not config.should_add_error:
        return None
    return [__calculate_mean_standard_error(dist) for dist in distances_to_plot if len(dist) > 0]


def __calculate_mean_standard_error(data: pd.Series):
    sem_by_subject = data.groupby(level=[config.SUBJECT, config.TIMESTAMP]).sem()
    mean_sem = sem_by_subject.groupby(level=config.TIMESTAMP).mean()
    return mean_sem / 2
