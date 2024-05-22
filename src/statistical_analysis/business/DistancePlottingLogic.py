import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import binom_test

from src import config as g_config, utils
from src.statistical_analysis import config
from src.statistical_analysis.business.NormalizationLogic import mega_score_normalization
from src.statistical_analysis.business.SleepLogic import score_by_sleep_duration


def save_figure(figure, file_name: str, dir_name: str) -> None:
    # assert ((file_name is not None)
    #         and (len(file_name) > 0)
    #         and (all([c.isalpha() or c.isdigit() or c == '_' or c == '-' for c in file_name]))
    #         ), f'Must provide a legal file_name, composed of alphanumeric characters only, {file_name} given.'
    full_path = os.path.join(__make_dir(dir_name), file_name + g_config.SVG)
    figure.savefig(full_path)
    return


def plot_average_per_trial(df: pd.DataFrame, feature='Result', **kwargs):
    min_percent_samples, ignore_below_min_samples, errors, sup_title, ax_title, x_label, y_label, filename, directory = __unpack_plotting_kwargs(
        **kwargs)
    print(f"plotting {y_label} for {filename}")
    fig, ax = plt.subplots(1)
    fig.set_size_inches(config.FIG_SIZE)

    if config.should_aggregate_by_subject:
        dfg_mean = df.groupby([g_config.SESSION, g_config.SUBJECT])[feature].mean().reset_index()
        dfg_mean = dfg_mean[dfg_mean[feature].notna()]
        indexed = dfg_mean.set_index(g_config.SUBJECT)

        # remove subjects with nan or no full data
        if not config.should_normalize:
            indexed = indexed[indexed.index.duplicated(keep=False)]
    else:
        dfg_mean = df.groupby([g_config.SESSION, g_config.SUBJECT, g_config.MOVIE])[feature].mean().reset_index()
        dfg_mean = dfg_mean[dfg_mean[feature].notna()]
        indexed = dfg_mean.set_index([g_config.SUBJECT, g_config.MOVIE])
        # remove subjects with nan or no full data
        if not config.should_normalize:
            indexed = indexed[indexed.index.duplicated(keep=False)]
        filename = f"subjectAndMovie_{filename}_{y_label}"

    # Filter rows with Saccade_Duration_Median > 1000
    # if config.SHOULD_FILTER_SUBJECTS:
    #     indexed = indexed[indexed[feature[0]] < 300]

    session_values = [_ for _, d in dfg_mean.groupby([g_config.SESSION])]
    if config.splitted_violin:
        # indexed["all"] = "all"
        # sns.violinplot(x="all", y=feature, data=indexed, showmeans=True, split=True, hue=g_config.SESSION,
        #                inner='quart')
        indexed["Remembered?"] = indexed["Confidence"] > 0
        sns.violinplot(x="Remembered?", y=feature[0], data=indexed, showmeans=True, inner='quart')
    else:
        filename = f"subject_{filename}_{y_label}"
        pivoted_df = indexed.pivot_table(values=[feature], index=indexed.index, columns=[g_config.SESSION],
                                         aggfunc='first')
        # datapoints = pd.melt(pivoted_df)
        datapoints = indexed.copy()
        # datapoints['ID'] = datapoints.index.values
        datapoints = datapoints.reset_index()
        # Line plot color coded based on the slope of the line (positive or negative) and the value of the slope
        datapoints['slope'] = datapoints.groupby(g_config.SUBJECT)[feature].transform(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])

        # The color will be based on the slope value
        slope_color = datapoints.apply(lambda row: 'g' if row.slope > 0 else 'r', axis=1)
        slope_color_dict = dict(zip(datapoints[g_config.SUBJECT], slope_color))
        sns.lineplot(data=datapoints, x=g_config.SESSION, y=feature, hue=g_config.SUBJECT, legend=False, palette=slope_color_dict)

        sns.set(style="whitegrid")
        sns.swarmplot(x=g_config.SESSION, y=feature, data=dfg_mean, hue=g_config.SUBJECT)

        my_palette = {ses: g_config.teal if ses == g_config.SESSION_A else g_config.salmon for ses in session_values}
        sns.violinplot(x=g_config.SESSION, y=feature, data=dfg_mean, showmeans=True, inner='quart', palette=my_palette)
        _add_stats_significance(ax, indexed, feature)

    _add_titles(fig, sup_title, ax, ax_title, x_label, y_label)
    if filename:
        save_figure(fig, filename, directory)

    return fig


def plot_average_per_trial_with_nap(nap: pd.DataFrame, sleep_count: pd.Series, feature='Result',  **kwargs):
    min_percent_samples, ignore_below_min_samples, errors, sup_title, ax_title, x_label, y_label, filename, directory = __unpack_plotting_kwargs(
        **kwargs)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(config.FIG_SIZE)
    if config.should_aggregate_by_subject:
        dfg_nap_mean = nap.groupby([g_config.SESSION, g_config.SUBJECT])[feature].mean().reset_index()
        nap_indexed = dfg_nap_mean.set_index(g_config.SUBJECT)
        filename = f"subject_{filename}"

        pivoted_df = nap_indexed.pivot_table(values=[feature], index=nap_indexed.index, columns=[g_config.SESSION],
                                             aggfunc='first')
        # datapoints = pd.melt(pivoted_df)
        datapoints = nap_indexed.copy()
        datapoints = datapoints.reset_index()

        # Line plot color coded based on the slope of the line (positive or negative) and the value of the slope
        datapoints['slope'] = datapoints.groupby(g_config.SUBJECT)[feature].transform(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])

        if sleep_count is not None:
            # filter sleep count to subjects with sleep efficiency > 0.5
            sleep_count = sleep_count[sleep_count > 0.5]
            # filter datapoints to rows with subject column with sleep efficiency > 0.5
            datapoints = datapoints[datapoints[g_config.SUBJECT].isin(sleep_count.index)]
            # add sleep efficiency column to datapoints according to the subject column
            datapoints['SleepEfficiency'] = datapoints[g_config.SUBJECT].map(sleep_count)

        # TODO Check what happens with start time > 0
        # Dump the mega score df into pkl
        datapoints.to_pickle(r'C:\Users\user\PycharmProjects\gaze\Gaze\resources\nap\statistical_analysis'
                             r'\nap_subjects_df.pkl')

        # The color will be based on the slope value
        slope_color = datapoints.apply(lambda row: 'g' if row.slope > 0 else 'r', axis=1)
        slope_color_dict = dict(zip(datapoints[g_config.SUBJECT], slope_color))
        sns.lineplot(data=datapoints, x=g_config.SESSION, y=feature, hue=g_config.SUBJECT, legend=False,
                     palette=slope_color_dict)
    else:
        datapoints = nap.groupby([g_config.SESSION, g_config.SUBJECT, g_config.MOVIE])[feature].mean().reset_index()
        datapoints = datapoints.set_index([g_config.SUBJECT, g_config.MOVIE])
        filename = f"subjectAndMovie_{filename}"

    sns.swarmplot(x=g_config.SESSION, y=feature, data=datapoints, hue=g_config.SUBJECT)
    sns.violinplot(x=g_config.SESSION, y=feature, data=datapoints, showmeans=True, inner='quart', palette="Blues")
    _add_stats_significance(ax, datapoints, feature)

    _add_titles(fig, sup_title, ax, ax_title, x_label, y_label)
    if filename:
        save_figure(fig, filename, directory)

    return fig


# this function plot comparison between the observed data and the permutation resampling distribution
# Using seaborn jointplot function
# it does it by shuffling the data on the movie level, and then averaging the data on the subject level
# it also returns the bootstrap distribution, the stat value and the p value
def plot_bootstrap_dist_aggregated_by_subject(df: pd.DataFrame, observed_effect: pd.DataFrame, feature='Result',
                                              **kwargs):
    min_percent_samples, ignore_below_min_samples, errors, sup_title, ax_title, x_label, y_label, filename, directory = __unpack_plotting_kwargs(
        **kwargs)
    print(f"plotting {y_label} for {filename}")
    fig, ax = plt.subplots(1)
    fig.set_size_inches(config.FIG_SIZE)

    # Clean and organize data
    df = df[df[feature].notna()]
    observed_effect = observed_effect.groupby([g_config.SESSION, g_config.SUBJECT])[feature].mean().reset_index()

    # Perform permutation resampling
    np.random.seed(0)  # for reproducibility
    n_permutations = config.NUM_ITERATIONS
    bootstrap_dist = []

    for i in range(n_permutations):
        # create a copy of the data
        flipped = df.copy()

        # run over all subjects and movies, and shuffle the session labels according to the coin flip
        for subject in df[g_config.SUBJECT].unique():
            for movie in df[g_config.MOVIE].unique():
                coin_flip = np.random.randint(2)
                if coin_flip == 1:
                    # Run over the (subject, movie) data and replace the value of the Session column
                    # with the opposite session
                    flipped.loc[
                        (flipped[g_config.SUBJECT] == subject) & (flipped[g_config.MOVIE] == movie), g_config.SESSION] = \
                        flipped.loc[(flipped[g_config.SUBJECT] == subject) & (
                                flipped[g_config.MOVIE] == movie), g_config.SESSION].apply(
                            lambda x: g_config.SESSION_A if x == g_config.SESSION_B else g_config.SESSION_B)

            # Normalize the subject data
            normalization = mega_score_normalization(flipped[(flipped[g_config.SUBJECT] == subject)])
            # Taking the mean of the normalized data
            bootstrap_dist.append(normalization[feature].mean())

    # Add the observed effect to the bootstrap distribution to one df, for plotting
    #  add a flag column to distinguish between the bootstrap distribution and the observed effect
    bootstrap_dist_df = pd.DataFrame(bootstrap_dist, columns=[feature])
    bootstrap_dist_df['Subjects Type'] = 'Bootstrap'
    observed_effect['Subjects Type'] = 'Observed'
    df_toplot = pd.concat([bootstrap_dist_df, observed_effect], ignore_index=True)[[feature, 'Subjects Type']]

    sns.set(style="whitegrid")
    # dots with black circles
    sns.swarmplot(x='Subjects Type', y=feature, data=df_toplot, hue='Subjects Type', palette="Set2")

    # subjects_type_values = [_ for _, d in df_toplot.groupby(['Subjects Type'])]
    sns.violinplot(x='Subjects Type', y=feature, data=df_toplot, showmeans=True, inner='quart')

    _add_stats_significance(ax, df, feature)

    _add_titles(fig, sup_title, ax, ax_title, x_label, y_label)
    filename = f"subject_{filename}_{y_label}"
    save_figure(fig, filename, directory)

    return fig


def _add_2pop_stats_significance(ax, dfg_mean, feature='Result', group_by=g_config.SESSION):
    df_list = [d for _, d in dfg_mean.groupby([group_by])]
    a = df_list[0][feature]
    a_mean = a.mean()
    a_std = a.std()
    b = df_list[1][feature]
    b_mean = b.mean()
    b_std = b.std()
    print(f"a_mean: {a_mean}")
    print(f"a_std: {a_std}")
    print(f"b_mean: {b_mean}")
    print(f"b_std: {b_std}")

    # These are your two groups
    group1 = np.array(a)  # replace with your data
    group2 = np.array(b)  # replace with your data

    # Perform the Wilcoxon rank-sum test.
    test = stats.ranksums(group1, group2, alternative='greater')  # 'two-sided', 'less', 'greater
    pvalue = test.pvalue
    print(f"Stats analysis: {test}")

    # This is the observed difference in mean
    observed_difference = np.mean(group1) - np.mean(group2)
    print(f"observed_difference: {observed_difference}")
    # This is the percentage change of the mean
    observed_percentage_change = (observed_difference / max(np.mean(group2), np.mean(group1))) * 100
    print(f"observed_percentage_change: {observed_percentage_change}")
    # STD of the percentage change of the mean
    std_observed_percentage_change = (np.std(group1) / np.mean(group1) + np.std(group2) / np.mean(group2))

    # Effect size calculation using Cohen's d
    # pooled standard deviation
    pooled_std = np.sqrt(((len(group1) - 1) * np.std(group1) ** 2 + (len(group2) - 1) * np.std(group2) ** 2) / (
            len(group1) + len(group2) - 2))
    # Cohen's d
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    print(f"cohens_d: {cohens_d}")


    # This is the combined data
    # combined = np.concatenate([group1, group2])

    # Perform the permutation test
    # num_permutations = 10000
    # count = 0
    # for _ in range(num_permutations):
    #     permuted = np.random.permutation(combined)
    #     permuted_group1 = permuted[:len(group1)]
    #     permuted_group2 = permuted[len(group1):]
    #     if np.mean(permuted_group1) - np.mean(permuted_group2) > observed_difference:
    #         count += 1
    #
    # p_value = count / num_permutations
    # print(f'p-value = {p_value}')

    # statistical annotation
    # x1, x2 = 0, 1
    # result__max = dfg_mean[feature].max()
    # margin = result__max * 0.05
    # y, h, col = result__max + margin, margin, 'k'
    # ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    # ax.text((x1 + x2) * .5, y + h, utils.convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom',
    #         color=col, size=23)


def plot_two_populations(control_group: pd.DataFrame, control_group_label: str, main_group: pd.DataFrame,
                         main_group_label: str, sleep_count: pd.Series, **kwargs):
    min_percent_samples, ignore_below_min_samples, errors, sup_title, ax_title, x_label, y_label, filename, directory = __unpack_plotting_kwargs(
        **kwargs)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(config.FIG_SIZE)
    if config.should_aggregate_by_subject:
        dfg_no_nap_mean = control_group.groupby([g_config.SESSION, g_config.SUBJECT])['Result'].mean().reset_index()
        control_group_indexed = dfg_no_nap_mean.set_index(g_config.SUBJECT)
        dfg_nap_mean = main_group.groupby([g_config.SESSION, g_config.SUBJECT])['Result'].mean().reset_index()
        main_group_indexed = dfg_nap_mean.set_index(g_config.SUBJECT)
        filename = f"subject_{filename}"
    else:
        dfg_no_nap_mean = control_group.groupby([g_config.SESSION, g_config.SUBJECT, g_config.MOVIE])[
            'Result'].mean().reset_index()
        control_group_indexed = dfg_no_nap_mean.set_index([g_config.SUBJECT, g_config.MOVIE])
        dfg_nap_mean = main_group.groupby([g_config.SESSION, g_config.SUBJECT, g_config.MOVIE])[
            'Result'].mean().reset_index()
        main_group_indexed = dfg_nap_mean.set_index([g_config.SUBJECT, g_config.MOVIE])
        filename = f"subjectAndMovie_{filename}"

    if sleep_count is not None:
        control_group_indexed['SleepEfficiency'] = 0
        main_group_indexed = score_by_sleep_duration(main_group_indexed, sleep_count)
        df_merged = main_group_indexed.append(control_group_indexed)
        df_merged[main_group_label] = df_merged.apply(lambda row: main_group_label if row.SleepEfficiency > 0.5 else control_group_label, axis=1)
    else:
        main_group_indexed[main_group_label] = main_group_label
        control_group_indexed[main_group_label] = control_group_label
        df_merged = main_group_indexed.append(control_group_indexed)

    # Dump the mega score df into pkl
    df_to_dump = df_merged.reset_index()
    df_to_dump.to_pickle(r'C:\Users\user\PycharmProjects\gaze\Gaze\resources\nap\statistical_analysis'
                        r'\two_populations_df.pkl')

    sns.violinplot(x=g_config.SESSION, y="Result", data=df_merged, showmeans=True, hue=main_group_label, split=True,
                   inner='quart')
    _add_2pop_stats_significance(ax, df_merged, group_by=main_group_label)

    _add_titles(fig, sup_title, ax, ax_title, x_label, y_label)
    if filename:
        save_figure(fig, filename, directory)

    return fig


def plot_two_populations_with_correlation(control_group: pd.DataFrame, control_group_label: str,
                                          main_group: pd.DataFrame, main_group_label: str,
                                          memory_performance: pd.DataFrame, **kwargs):
    min_percent_samples, ignore_below_min_samples, errors, sup_title, ax_title, x_label, y_label, filename, directory = __unpack_plotting_kwargs(
        **kwargs)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(config.FIG_SIZE)
    dfg_elderly_mean = control_group.groupby([g_config.SESSION, g_config.SUBJECT])['Result'].mean().reset_index()
    control_group_indexed = dfg_elderly_mean.set_index(g_config.SUBJECT)
    dfg_mci_ad_mean = main_group.groupby([g_config.SESSION, g_config.SUBJECT])['Result'].mean().reset_index()
    main_group_indexed = dfg_mci_ad_mean.set_index(g_config.SUBJECT)
    # Concat main and control groups
    df_merged = main_group_indexed.append(control_group_indexed)['Result']
    # Set subject as index
    memory_performance.set_index('Name', inplace=True)
    # Filter from both df subjects that are in both df
    df_merged = df_merged[df_merged.index.isin(memory_performance.index)]
    memory_performance = memory_performance[memory_performance.index.isin(df_merged.index)]
    filename = f"memory_performance_{filename}"

    # Plot correlation for each subject between Mega and Memory performance
    mem_metric = memory_performance[x_label]
    sns.regplot(x=mem_metric, y=df_merged, ax=ax)
    sns.scatterplot(x=mem_metric, y=df_merged, ax=ax)

    # Calculate correlation
    corr, pvalue = stats.pearsonr(mem_metric, df_merged)
    print(f"{x_label} Correlation: {corr}")
    print(f"{x_label} P-value: {pvalue}")

    _add_titles(fig, sup_title, ax, ax_title, x_label, y_label)
    if filename:
        save_figure(fig, filename, directory)

    return fig


def _add_stats_significance(ax, dfg_mean, feature='Result'):
    df_list = [d for _, d in dfg_mean.groupby([g_config.SESSION])]
    group1 = df_list[0][feature]
    group1_mean = group1.mean()
    group1_std = group1.std()
    if config.should_normalize:
        # list of zeros
        group2 = np.zeros(len(group1))
    else:
        group2 = df_list[1][feature]
    group2_mean = group2.mean()
    group2_std = group2.std()
    print(f"group1_mean: {group1_mean}")
    print(f"group1_std: {group1_std}")
    print(f"group2_mean: {group2_mean}")
    print(f"group2_std: {group2_std}")
    if config.should_normalize:
        # Calculate differences
        differences = group2 - group1

        # Count positive and negative differences
        positive_diffs = np.sum(differences > 0)
        negative_diffs = np.sum(differences < 0)

        # Use the smaller count for the test statistic
        test_statistic = min(positive_diffs, negative_diffs)

        # Perform the Sign test using the binomial test
        pvalue = binom_test(test_statistic, n=positive_diffs + negative_diffs, p=0.5)
        print(f"Sign test using the binomial test: {pvalue}")
    else:
        test = stats.wilcoxon(group1, group2, alternative='greater')  # 'two-sided', 'less', 'greater
        pvalue = test.pvalue
        print(f"Stats analysis: {test}")

        # This is the observed difference in mean
        observed_difference = group1_mean - group2_mean
        print(f"observed_difference: {observed_difference}")
        # This is the percentage change of the mean
        observed_percentage_change = (observed_difference / max(group2_mean, group1_mean)) * 100
        print(f"observed_percentage_change: {observed_percentage_change}")

        # Effect size calculation using Cohen's d
        # pooled standard deviation
        pooled_std = np.sqrt(((len(group1) - 1) * group1_std ** 2 + (len(group2) - 1) * group2_std ** 2) / (
                len(group1) + len(group2) - 2))
        # Cohen's d
        cohens_d = (group1_mean - group2_mean) / pooled_std
        print(f"cohens_d: {cohens_d}")


        # statistical annotation
        x1, x2 = 0, 1
        result__max = dfg_mean[feature].max()
        margin = result__max * 0.05
        y, h, col = result__max + margin, margin, 'k'
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        ax.text((x1 + x2) * .5, y + h, utils.convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom',
                color=col, size=23)


def is_data_normal_distributed(session_a, session_b):
    # plt.figure(10)
    # plt.hist(session_a, edgecolor='black', bins=20)
    shapiro_test_a = stats.shapiro(session_a)
    print(f"Session A: {shapiro_test_a}")
    # plt.show()

    # plt.figure(11)
    # plt.hist(session_b, edgecolor='black', bins=20)
    shapiro_test_b = stats.shapiro(session_b)
    print(f"Session B: {shapiro_test_b}")
    # plt.show()

    pvalue = (shapiro_test_a.pvalue > 0.05) and (shapiro_test_b.pvalue > 0.05)
    statistic = (not math.isnan(shapiro_test_a.statistic)) and (not math.isnan(shapiro_test_b.statistic))

    return statistic and pvalue


def plot_distances(distances: list, labels: list, line_colors: list,
                   roi_time: float, roi_duration: float, should_add_sem=False, **kwargs):
    assert (len(distances) == len(labels)
            ), f'All plotted distances must have a matching label. {len(distances)} distances and {len(labels)} provided.'
    assert (len(distances) == len(labels)
            ), f'All plotted distances must have a matching color. {len(distances)} distances and {len(line_colors)} provided.'
    min_percent_samples, ignore_below_min_samples, errors, sup_title, ax_title, x_label, y_label, filename, directory = __unpack_plotting_kwargs(
        **kwargs)
    assert ((errors is None)
            or (len(distances) == len(errors))
            or (len(errors) == 0)
            ), f'Length mismatch: must have error-bars for all distances or none of them.'

    plt.close('all')
    fig, ax = plt.subplots(1)
    fig.set_size_inches(config.FIG_SIZE)
    for i in range(len(distances)):
        dist_series, label, color = distances[i], labels[i], line_colors[i]
        error_series = errors[i] if ((errors is not None) and (len(errors))) else pd.Series()
        _plot_time_series_impl(ax, dist_series, error_series, color, label, min_percent_samples,
                               ignore_below_min_samples, should_add_sem)

    _mark_event_time_on_axis(ax, roi_time, roi_duration, )
    _add_titles(fig, sup_title, ax, ax_title, x_label, y_label)
    if filename:
        save_figure(fig, filename, directory)

    return fig


def _plot_time_series_impl(plt_axis, distances, errors, line_color: str, line_label: str,
                           min_percent_samples: float, ignore_below_min_samples: bool, should_add_sem=False):
    '''
    Plots the given mean distance on the given @plt_axis, on timestamps with at lease
        @min_percent_samples out of max possible samples.
    @args:
        distances: pd.Series of distances, containing an index-level matching self.TIMESTAMP
        plt_axis: pyplot axis object
        min_percent_samples: float within range [0,100]; the minimal percent of samples-per-timestamp necessary
            to draw that timestamp
        ignore_below_min_samples: bool; If True, only timestamps with more that the threshold sample-size
            will be plotted. If False, all will be plotted but with varying line-width and style
        line_color, line_label: str; If not provided, will use plt's default colors and none-label
    '''
    assert ((min_percent_samples >= 0) and (min_percent_samples <= 100)
            ), f'Argument @min_percent_samples must be between 0 and 100, {min_percent_samples} given.'
    if len(distances) <= 0:
        return

    if config.should_plot_time_series_of_trial:
        one_trial = distances[(distances.index.get_level_values(g_config.SUBJECT)
                                                         == 'BO03') & (
                distances.index.get_level_values(g_config.MOVIE) == 'mov14')].groupby(level=g_config.TIMESTAMP).mean()

        plt_axis.plot(one_trial, color=line_color,
                      label=line_label, lw=config.LINE_WIDTH_BOLD)

    else:
        grouped_distances = distances.groupby(level=g_config.TIMESTAMP)
        mean_distances = grouped_distances.mean()  # TODO mean vs median
        samples_counts = grouped_distances.size()
        max_possible_samples = samples_counts.max()

        mean_distance_with_high_sample_count = mean_distances[
            samples_counts >= max_possible_samples * min_percent_samples / 100]
        mean_distance_with_medium_sample_count = mean_distances[
            samples_counts >= max_possible_samples * min_percent_samples / (100 * 2)]
        mean_distance_with_low_sample_count = mean_distances

        plt_axis.plot(mean_distance_with_high_sample_count, color=line_color,
                      label=line_label, lw=config.LINE_WIDTH_BOLD)

        if should_add_sem:
            deviation = 2 * grouped_distances.sem().dropna()
            under_line = (mean_distance_with_high_sample_count - deviation).dropna()
            over_line = (mean_distance_with_high_sample_count + deviation).dropna()
            plt.fill_between(mean_distance_with_high_sample_count.index, under_line, over_line, color=line_color,
                             alpha=.1)  # std curves.

        if not ignore_below_min_samples:
            # add lighter lines where the data doesn't have enough samples
            plt_axis.plot(mean_distance_with_medium_sample_count, color=line_color,
                          ls=config.MEDIUM_LINESTYLE, lw=config.LINE_WIDTH_MEDIUM, alpha=config.LINE_ALPHA_HIGH)
            plt_axis.plot(mean_distance_with_low_sample_count, color=line_color,
                          ls=config.LIGHT_LINESTYLE, lw=config.LINE_WIDTH_NARROW, alpha=config.LINE_ALPHA_MEDIUM)
        if len(errors):
            errors_with_high_samples_count = errors[samples_counts >= max_possible_samples * min_percent_samples / 100]
            plt_axis.plot(mean_distance_with_high_sample_count + errors_with_high_samples_count,
                          color=line_color, lw=config.LINE_WIDTH_NARROW, alpha=config.LINE_ALPHA_MEDIUM)
            plt_axis.plot(mean_distance_with_high_sample_count - errors_with_high_samples_count,
                          color=line_color, lw=config.LINE_WIDTH_NARROW, alpha=config.LINE_ALPHA_MEDIUM)
    return


def _mark_event_time_on_axis(plot_axis, roi_time: float, roi_duration: float):
    plot_axis.axvline(x=roi_time, c=config.EVENT_TIME_LINE_COLOR, label='Event Time', lw=2)
    plot_axis.axvline(x=roi_time - roi_duration, c=config.EVENT_TIME_LINE_COLOR, ls='--')
    plot_axis.axvline(x=roi_time + roi_duration, c=config.EVENT_TIME_LINE_COLOR, ls='--')
    plot_axis.axvline(x=roi_time - 2 * roi_duration, c=config.EVENT_TIME_LINE_COLOR, ls=':')
    plot_axis.axvline(x=roi_time + 2 * roi_duration, c=config.EVENT_TIME_LINE_COLOR, ls=':')
    plot_axis.axvspan(roi_time - roi_duration, roi_time + roi_duration,
                      facecolor=config.EVENT_TIME_FACE_COLOR, alpha=config.SURFACE_ALPHA_MEDIUM)
    plot_axis.axvspan(roi_time - roi_duration, roi_time - 2 * roi_duration,
                      facecolor=config.EVENT_TIME_FACE_COLOR, alpha=config.SURFACE_ALPHA_LOW)
    plot_axis.axvspan(roi_time + roi_duration, roi_time + 2 * roi_duration,
                      facecolor=config.EVENT_TIME_FACE_COLOR, alpha=config.SURFACE_ALPHA_LOW)
    return


def _add_titles(plt_fig, sup_title, plt_axis, axis_title: str, x_label: str, y_label: str):
    ''' Adds a title, legend and x&y labels to the plt_axis '''
    plt_fig.suptitle(sup_title, fontsize=config.SUP_TITLE_FONTSIZE, y=0.95)
    plt_axis.set_title(axis_title, fontsize=config.SUB_TITLE_FONTSIZE)
    plt_axis.set_xlabel(x_label, fontsize=config.AXIS_LABEL_FONTSIZE)
    plt_axis.set_ylabel(y_label, fontsize=config.AXIS_LABEL_FONTSIZE)
    plt_axis.tick_params(labelsize=config.TICKS_LABEL_FONTSIZE)
    # plt_axis.set_xticklabels(['1st viewing', '2nd viewing'])

    # if config.should_normalize:
    #     plt_axis.set_ylim([0, 2])

    if config.ADD_LEGENDS:
        plt_axis.legend(fontsize=config.LEGEND_FONTSIZE)
    else:
        if config.should_aggregate_by_subject:
            plt_axis.get_legend().remove()
    return


def __make_dir(dir_name: str):
    # assert (dir_name is not None) and (len(dir_name) > 0) and (
    #     all([c.isalpha() or c.isdigit() or c == '_' for c in dir_name])
    # ), f'Must provide a legal dir_name, composed of alphanumeric characters only, {dir_name} given.'
    if not (os.path.isdir(g_config.plots_dir)):
        os.mkdir(g_config.plots_dir)
    dir_path = os.path.join(g_config.plots_dir, dir_name)
    if not (os.path.isdir(dir_path)):
        os.mkdir(dir_path)
    return dir_path


def __unpack_plotting_kwargs(**kwargs):
    min_percent_samples = kwargs.pop('min_percent_samples', config.min_percent_samples)
    ignore_below_min_samples = kwargs.pop('ignore_below_min_samples', config.should_zoom_on_high_sample_count)
    errors = kwargs.pop('errors', None)
    sup_title = kwargs.pop('sup_title', config.DEFAULT_SUPTITLE)
    ax_title = kwargs.pop('ax_title', config.DEFAULT_SUBTITLE)
    x_label = kwargs.pop('x_label', config.DEFAULT_X_LABEL)
    y_label = kwargs.pop('y_label', config.DEFAULT_Y_LABEL)
    filename = kwargs.pop('filename', '')
    directory = kwargs.pop('directory', None)
    return min_percent_samples, ignore_below_min_samples, errors, sup_title, ax_title, x_label, y_label, filename, directory
