import numpy as np
import pandas as pd
from scipy.stats import stats

from src import config
from src.features_extraction import config as fa_config, utils
from src.features_extraction.models.Point import Point
from src.features_extraction.models.Rectangle import Rectangle
from src.features_extraction.models.Roi import Roi
from src.features_extraction.services import DataService


def get_in_out_roi_count_features(ttype: str) -> pd.DataFrame:
    """
    :return:  Data frame with index (Subject, Session, Movie) with the following:
        1. Number of data points in RoI
        2. Number of data points out RoI
        3. Ratio of in RoI data points /  out RoI data pointss
        All of the above are divided to Pre-RoI, During-RoI, Post-RoI
    """

    print(f"Calculating {ttype} in-out RoI counts and ratio - START")

    in_counts = dict()
    out_counts = dict()
    ratio_counts = dict()

    for mov in config.valid_movies:

        pre_roi_data, post_roi_data, roi = _get_movie_data(ttype, mov)
        if roi is not None:
            # Calculating in,out,ratio for data points in each segment
            in_pre, out_pre, ratio_pre = _calc_in_out_counts(pre_roi_data, roi.rect, ttype)
            in_post, out_post, ratio_post = _calc_in_out_counts(post_roi_data, roi.rect, ttype)

            # Unite to one data frame and save it in a dictionary with the movie as key
            in_counts[mov] = utils.unite_series_to_df(in_pre, in_post, "counts in", ttype)
            out_counts[mov] = utils.unite_series_to_df(out_pre, out_post, "counts out", ttype)
            ratio_counts[mov] = utils.unite_series_to_df(ratio_pre, ratio_post, "ratio", ttype)

    # Convert dictionaries to data frame with movie as part of the index
    in_counts = utils.dfs_dict_to_df(in_counts)
    out_counts = utils.dfs_dict_to_df(out_counts)
    ratio_counts = utils.dfs_dict_to_df(ratio_counts)

    # Dump to pickle

    # Unite all data frames to one
    res = pd.concat([in_counts, out_counts, ratio_counts], axis=1).fillna(0)
    print(f"Calculating {ttype} in-out RoI counts and ratio - DONE\n")

    return res


def get_in_roi_stat_features(ttype: str) -> pd.DataFrame:
    """
    :return:  Data frame with index (Subject, Session, Movie) and features in RoI data point for each movie
    """

    print(f"Calculating {ttype} first-in-roi - START")

    in_roi = dict()

    for mov in config.valid_movies:

        pre_roi_data, post_roi_data, roi = _get_movie_data(ttype, mov)
        if roi is not None:
            # Calculating in,out,ratio for data points in each segment
            in_roi_pre = _calc_in_roi_stats(pre_roi_data, roi.rect, ttype)
            in_roi_post = _calc_in_roi_stats(post_roi_data, roi.rect, ttype)

            # init timing for post roi data to be relative to the roi
            in_roi_post = utils.init_time_columns(in_roi_post, roi.start)

            # Unite to one data frame and save it in a dictionary with the movie as key
            in_roi_pre = in_roi_pre.add_suffix(f"_{ttype}_In_RoI_Pre")
            in_roi_post = in_roi_post.add_suffix(f"_{ttype}_In_RoI_Post")
            in_roi[mov] = pd.concat([in_roi_pre, in_roi_post], axis=1).fillna(0)

    # Convert dictionaries to data frame with movie as part of the index
    res = utils.dfs_dict_to_df(in_roi)

    print(f"Calculating {ttype} first-in RoI {res.columns}- DONE\n")

    return res


def get_dva_features(ttype: str) -> pd.DataFrame:
    """
    :return:  Data frame with index (Subject, Session, Movie) and features in RoI data point for each movie
    """

    print(f"Calculating {ttype} DVA features - START")

    in_roi = dict()

    for mov in config.valid_movies:
        pre_roi_data, post_roi_data, roi = _get_movie_data(ttype, mov)
        # Calculating stats for data points in each segment
        roi_pre = get_stats(pre_roi_data[get_distance_mertics()])
        roi_post = get_stats(post_roi_data[get_distance_mertics()])

        # Unite to one data frame and save it in a dictionary with the movie as key
        roi_pre = roi_pre.add_suffix(f"_{ttype}_Pre")
        roi_post = roi_post.add_suffix(f"_{ttype}_Post")
        in_roi[mov] = pd.concat([roi_pre, roi_post], axis=1)

    # Convert dictionaries to data frame with movie as part of the index
    res = utils.dfs_dict_to_df(in_roi)

    print(f"Calculating {ttype} Distance {res.columns}- DONE\n")

    return res


def get_distance_mertics():
    if config.POPULATION == 'yoavdata':
        return [config.DISTANCE, config.DVA]

    return [config.DISTANCE, config.DVA, config.SQRT_DVA]


def get_stats(data_in_roi: pd.DataFrame) -> pd.DataFrame:
    mean_in_roi = data_in_roi.groupby(level=[config.SUBJECT, config.SESSION]).mean().add_suffix("_Mean")
    median_in_roi = data_in_roi.groupby(level=[config.SUBJECT, config.SESSION]).median().add_suffix("_Median")
    std_in_roi = data_in_roi.groupby(level=[config.SUBJECT, config.SESSION]).std().add_suffix("_Std")
    max_in_roi = data_in_roi.groupby(level=[config.SUBJECT, config.SESSION]).max().add_suffix("_Max")
    min_in_roi = data_in_roi.groupby(level=[config.SUBJECT, config.SESSION]).min().add_suffix("_Min")
    sem_in_roi = data_in_roi.groupby(level=[config.SUBJECT, config.SESSION]).sem().add_suffix("_Sem")
    auc_in_roi = data_in_roi.groupby(level=[config.SUBJECT, config.SESSION]).sum().add_suffix("_AUC")

    return pd.concat([mean_in_roi, median_in_roi, std_in_roi, max_in_roi, min_in_roi, sem_in_roi, auc_in_roi], axis=1)


def get_re_entries_count_features(ttype: str) -> pd.DataFrame:
    """
        :return:  Series with index (Subject, Session, Movie) with the following:
            1. Number of re-entries of data points in movie's RoI for all movies

        @note
        A re-entry is defined as a in-after-out data-point, i.e. we're counting instances of False-True pairs.
    """

    print(f"Calculating {ttype} re-entries to RoI counts and rates - START")

    re_entries_counts = dict()
    re_entries_rates = dict()

    for mov in config.valid_movies:

        pre_roi_data, post_roi_data, roi = _get_movie_data(ttype, mov)
        if roi is None:
            continue

        # Calculating re-entries counts and rates for data points in each segment
        re_entries_counts_pre, re_entries_rates_pre = _calc_re_entries_counts(pre_roi_data, roi, ttype, mov, "pre")
        re_entries_counts_post, re_entries_rates_post = _calc_re_entries_counts(post_roi_data, roi, ttype, mov, "post")

        # Unite to one data frame and save it in a dictionary with the movie as key
        re_entries_counts[mov] = utils.unite_series_to_df(re_entries_counts_pre,
                                                          re_entries_counts_post, "re entries counts", ttype)
        re_entries_rates[mov] = utils.unite_series_to_df(re_entries_rates_pre,
                                                         re_entries_rates_post, "re entries rates", ttype)

    # Convert dictionaries to data frame with movie as part of the index
    re_entries_counts = utils.dfs_dict_to_df(re_entries_counts)
    re_entries_rates = utils.dfs_dict_to_df(re_entries_rates)

    res = pd.concat([re_entries_counts, re_entries_rates], axis=1).fillna(0)
    print(f"Calculating {ttype} re-entries to RoI counts and rates - DONE\n")

    return res


def get_first_in_roi_features(ttype: str) -> pd.DataFrame:
    """
    :return:  Data frame with index (Subject, Session, Movie) and the first feature in RoI data point for each movie
    """

    print(f"Calculating {ttype} first-in-roibased - START")

    first_in = dict()

    for mov in config.valid_movies:

        pre_roi_data, post_roi_data, roi = _get_movie_data(ttype, mov)
        if roi is not None:
            # Calculating in,out,ratio for data points in each segment
            first_in_pre = _calc_first_in_roi(pre_roi_data, roi.rect, ttype)
            first_in_post = _calc_first_in_roi(post_roi_data, roi.rect,
                                               ttype)

            # init timing for post roibased data to be relative to the roibased
            first_in_post = utils.init_time_columns(first_in_post, roi.start)

            # Unite to one data frame and save it in a dictionary with the movie as key
            first_in_pre = first_in_pre.add_suffix(f"_{ttype}_First_In_RoI_Pre")
            first_in_post = first_in_post.add_suffix(f"_{ttype}_First_In_RoI_Post")
            first_in[mov] = pd.concat([first_in_pre, first_in_post], axis=1).fillna(0)

    # Convert dictionaries to data frame with movie as part of the index
    res = utils.dfs_dict_to_df(first_in)

    print(f"Calculating {ttype} first-in RoI {res.columns}- DONE\n")

    return res


def get_re_entries_pupil_features(ttype: str) -> pd.DataFrame:
    """
        :return:  Series with index (Subject, Session, Movie) with the following:
            1. Number of re-entries of data points in movie's RoI for all movies

        @note
        A re-entry is defined as a in-after-out data-point, i.e. we're counting instances of False-True pairs.
    """

    print(f"Calculating {ttype} re-entries to RoI pupil diff features - START")

    first_entry_pupil_diff = dict()
    re_entries_pupil_mean_diff = dict()

    for mov in config.valid_movies:

        pre_roi_data, post_roi_data, roi = _get_movie_data(ttype, mov)
        if roi is None:
            continue

        # Calculating re-entries for data points in each segment
        first_entry_pupil_diff_pre, re_entries_pupil_mean_diff_pre = _calc_re_entries_pupil(pre_roi_data, roi, ttype)
        first_entry_pupil_diff_post, re_entries_pupil_mean_diff_post = _calc_re_entries_pupil(post_roi_data, roi, ttype)

        # Unite to one data frame and save it in a dictionary with the movie as key
        first_entry_pupil_diff_pre = first_entry_pupil_diff_pre.add_suffix(f"_{ttype}_First_Diff_Pre")
        first_entry_pupil_diff_post = first_entry_pupil_diff_post.add_suffix(f"_{ttype}_First_Diff_Post")
        first_entry_pupil_diff[mov] = pd.concat([first_entry_pupil_diff_pre, first_entry_pupil_diff_post],
                                                axis=1).fillna(0)
        re_entries_pupil_mean_diff_pre = re_entries_pupil_mean_diff_pre.add_suffix(f"_{ttype}_ReEntry_Mean_Diff_Pre")
        re_entries_pupil_mean_diff_post = re_entries_pupil_mean_diff_post.add_suffix(f"_{ttype}_ReEntry_Mean_Diff_Post")
        re_entries_pupil_mean_diff[mov] = pd.concat([re_entries_pupil_mean_diff_pre, re_entries_pupil_mean_diff_post],
                                                    axis=1).fillna(0)

    # Convert dictionaries to data frame with movie as part of the index
    first_entry_pupil_diff = utils.dfs_dict_to_df(first_entry_pupil_diff)
    re_entries_pupil_mean_diff = utils.dfs_dict_to_df(re_entries_pupil_mean_diff)

    res = pd.concat([first_entry_pupil_diff, re_entries_pupil_mean_diff], axis=1).fillna(0)
    print(f"Calculating {ttype} RoI pupil diff features - DONE\n")

    return res


def get_pupil_change_feature(ttype: str) -> pd.DataFrame:
    print(f"Calculating {ttype} mean pupil diff before and after the event - START")

    mean_pupil_diff = dict()

    for mov in config.valid_movies:

        pre_roi_data, post_roi_data, roi = _get_movie_data(ttype, mov)
        if roi is None:
            continue

        # Calculating mean pupil for data points in each segment
        pupil_mean_pre = pre_roi_data[config.PUPIL if ttype == 'Fixations' else 'Pupil radius'].groupby(level=[
            config.SUBJECT, config.SESSION]).mean()
        pupil_mean_post = post_roi_data[config.PUPIL if ttype == 'Fixations' else 'Pupil radius'].groupby(level=[
            config.SUBJECT, config.SESSION]).mean()
        # Take the intersection of the indices
        pupil_mean_pre = pupil_mean_pre[pupil_mean_pre.index.isin(pupil_mean_post.index)]
        pupil_mean_post = pupil_mean_post[pupil_mean_post.index.isin(pupil_mean_pre.index)]
        # Calculate the percentage change between the pre and post segments for each subject and session
        max = np.maximum(pupil_mean_pre, pupil_mean_post, np.zeros_like(pupil_mean_pre) + np.finfo(float).eps)
        mov_pupil_mean_diff = (pupil_mean_post - pupil_mean_pre)  # / max

        # Set the column name to be "_{ttype}_Mean_Pupil_Percentage_Change_On_Event
        mov_pupil_mean_diff = mov_pupil_mean_diff.rename(f"{ttype}_Mean_Pupil_Change_On_Event")

        # Save it in a dictionary with the movie as key
        mean_pupil_diff[mov] = mov_pupil_mean_diff

    # Convert dictionaries to data frame with movie as part of the index
    mean_pupil_diff_df = utils.dfs_dict_to_df(mean_pupil_diff)

    print(f"Calculating {ttype} Mean pupil diff feature - DONE\n")

    return mean_pupil_diff_df


def _get_movie_data(ttype: str, mov: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, Roi):
    data = DataService.get_data(ttype)

    mov_data = data.xs(mov, level=config.MOVIE)
    mov_duration = DataService.videos_dims.loc[mov, fa_config.duration]
    roi = _get_roi(mov)
    if roi is None:
        return None, None, None

    # Splitting data to pre\post RoI segments
    time, is_index = utils.get_time_params(ttype)
    pre_roi_data = utils.split_data(mov_data, start=0, end=roi.start, by=time, index=is_index)
    post_roi_data = utils.split_data(mov_data, start=roi.start, end=mov_duration, by=time, index=is_index)

    return pre_roi_data, post_roi_data, roi


def _calc_in_out_counts(data: pd.DataFrame, roi_rect: Rectangle, ttype: str) -> (
        pd.Series, pd.Series, pd.Series):
    if data is None or data.empty:
        # Match index to original data
        names = (config.SUBJECT, config.SESSION)
        return utils.handle_empty_movie_data(names, count=3)

    X, Y = utils.get_location_params(ttype)
    data_in_roi = _is_in_roi(data, X, Y, roi_rect)
    data_out_roi = ~data_in_roi

    in_roi_count = data_in_roi.groupby(level=[config.SUBJECT, config.SESSION]).sum().astype(int)
    out_roi_count = data_out_roi.groupby(level=[config.SUBJECT, config.SESSION]).sum().astype(int)
    out_roi_count_denominator = out_roi_count.replace(0, 1)
    # Divide to calculate ratio
    in_out_ratio = in_roi_count.divide(out_roi_count_denominator)

    return in_roi_count, out_roi_count, in_out_ratio


def _calc_re_entries_pupil(data: pd.DataFrame, roi: Roi, ttype: str) -> (pd.Series, pd.Series):
    if data is None or data.empty:
        # Match index to original data
        names = (config.SUBJECT, config.SESSION)
        return utils.handle_empty_movie_data(names, count=2)

    X, Y = utils.get_location_params(ttype)
    data_in_roi = _is_in_roi(data, X, Y, roi.rect)
    re_entries = data_in_roi[_get_re_entries(data_in_roi)]

    # Splitting data to dataframes by subject and session
    data_gb = data.groupby(level=[config.SUBJECT, config.SESSION])
    # calculate the pupil difference by subtracting from each fixation (row) the previous fixation (row), for each group
    pupil_diff = pd.DataFrame()
    for gb in data_gb:
        g = data_gb.get_group(gb[0])
        g.loc[:, config.PUPIL] = g.loc[:, config.PUPIL].pct_change()  # * 100
        pupil_diff = pd.concat([pupil_diff, g], axis=0)

    # filter the pupil differences by the re-entries
    pupil_diff = pupil_diff[pupil_diff.index.isin(re_entries.index)][[config.PUPIL]]
    pupil_diff = pupil_diff.dropna()

    return pupil_diff.groupby(level=[config.SUBJECT, config.SESSION]).first(), \
        pupil_diff.groupby(level=[config.SUBJECT, config.SESSION]).mean()


def _calc_re_entries_counts(data: pd.DataFrame, roi: Roi, ttype: str, mov: str, segment: str) -> (pd.Series, pd.Series):
    if data is None or data.empty:
        # Match index to original data
        names = (config.SUBJECT, config.SESSION)
        return utils.handle_empty_movie_data(names, count=2)

    X, Y = utils.get_location_params(ttype)
    data_in_roi = _is_in_roi(data, X, Y, roi.rect)

    re_entries_counts = _get_re_entries(data_in_roi).groupby(level=[config.SUBJECT, config.SESSION]).sum().astype(int)

    if segment == "pre":
        duration = roi.start
    elif segment == "post":
        duration = DataService.videos_dims.loc[mov, fa_config.duration] - roi.end
    else:
        raise ValueError(f"Time segment {segment} is not supported for re-entries calculation")

    duration /= 1000  # convert to seconds
    re_entries_rates = re_entries_counts.divide(duration)
    return re_entries_counts, re_entries_rates


def _calc_first_in_roi(data: pd.DataFrame, roi_rect: Rectangle, ttype: str) -> (
        pd.Series, pd.Series, pd.Series):
    if data is None or data.empty:
        # Match index to original data
        names = (config.SUBJECT, config.SESSION)
        return utils.handle_empty_movie_data(names, count=3)

    X, Y = utils.get_location_params(ttype)
    data_in_roi = data[_is_in_roi(data, X, Y, roi_rect)]

    # taking the first feature in each trial
    firsts_in_roi = data_in_roi.groupby(level=[config.SUBJECT, config.SESSION]).first()

    # get the relevant columns for each feature
    feature_columns = utils.get_feature_columns(ttype)
    firsts_in_roi = firsts_in_roi[feature_columns]

    return firsts_in_roi


def _calc_in_roi_stats(data: pd.DataFrame, roi_rect: Rectangle, ttype: str) -> (
        pd.Series, pd.Series, pd.Series):
    if data is None or data.empty:
        # Match index to original data
        names = (config.SUBJECT, config.SESSION)
        return utils.handle_empty_movie_data(names, count=3)

    X, Y = utils.get_location_params(ttype)
    data_in_roi = data[_is_in_roi(data, X, Y, roi_rect)]

    # get the relevant columns for each feature
    feature_columns = utils.get_feature_columns(ttype)
    data_in_roi = data_in_roi[feature_columns]

    return get_stats(data_in_roi)


def _calc_dwell_time(data: pd.DataFrame, roi: Roi, ttype: str, mov: str, segment: str) -> (pd.Series, pd.Series):
    if data is None or data.empty:
        # Match index to original data
        names = (config.subject_level, config.session_level)
        return utils.handle_empty_movie_data(names, count=2)

    X, Y = utils.get_location_params(ttype)
    data_in_roi = _is_in_roi(data, X, Y, roi.rect)

    dwell_time = _get_dwell_time(data, data_in_roi, ttype)

    duration = utils.get_segment_duration(segment, roi, mov)
    dwell_time_ratio = dwell_time.divide(duration)

    return dwell_time, dwell_time_ratio


def _get_re_entries(is_in_roi: pd.Series):
    """
        @:return:
        The number of False-True consecutive pairs in the boolean series
    """

    return is_in_roi & ~is_in_roi.shift(fill_value=True)


def _get_dwell_time(data: pd.DataFrame, is_in_roi: pd.Series, ttype: str) -> pd.Series:
    """
        @:returns
        The time (in ms) spent inside the RoI.
    """

    dwell = None

    if ttype == "fixations":
        dwell = data[is_in_roi][fa_config.duration].groupby(
            level=[config.SUBJECT, config.SESSION]
        ).sum()
    elif ttype == "gaze":
        gaze_per_second = 1000 / fa_config.gaze_recording_rate
        dwell = data[is_in_roi].groupby(
            level=[config.SUBJECT, config.SESSION]
        ).size() * gaze_per_second

    return dwell


def _is_in_roi(data, x_col: str, y_col: str, rect: Rectangle):
    """
        @:returns
        A boolean Series with the same indices as for the given data.
        x_col, y_col --> str; names of the column names for the X-axis and Y-axis values
        rect --> represntation of the RoI
    """

    assert (x_col in data.columns), f'Couldn\'t find column {x_col} in the given DataFrame\'s columns.'
    assert (y_col in data.columns), f'Couldn\'t find column {y_col} in the given DataFrame\'s columns.'

    points = pd.Series(zip(data[x_col], data[y_col])).apply(
        lambda coordinates: Point(coordinates[0], coordinates[1]))
    points = points.apply(lambda point: utils.is_in_rectangle(rect, point))
    points.index = data.index

    return points


def _get_roi(mov: str, std_window: int = 2) -> Roi or None:
    try:
        mov_data = DataService.rects.loc[mov]
        tl = Point(mov_data[fa_config.tlx], mov_data[fa_config.tly])
        br = Point(mov_data[fa_config.brx], mov_data[fa_config.bry])

        t_median_ = DataService.rects.loc[mov, config.T_MEDIAN]
        end = t_median_ + std_window * DataService.rects.loc[mov, config.T_STDEV]
        start = t_median_
        return Roi(tl, br, start, end)

    except KeyError:
        return None
