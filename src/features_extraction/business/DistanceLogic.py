import pandas as pd

from src import config
from src.features_extraction import config as fa_config
from src.features_extraction import utils
from src.features_extraction.services import DataService


def get_all_features(num_stdevs: float = 2) -> pd.DataFrame:
    print("Calculating Distance features_extraction - START\n")

    assert (num_stdevs >= 0), f'Argument @num_stdevs must be non-negative, {num_stdevs} given.'

    mean = get_collapsed_distance_before_during_and_after_roi('mean')
    stdev = get_collapsed_distance_before_during_and_after_roi('std')
    median = get_collapsed_distance_before_during_and_after_roi('median')

    assert ((len(mean) == len(stdev)) and (len(mean) == len(median))), f'Error extracting Distance Metrics!'

    # creates a flat list-of-Series where we have the before-series, then during-series and then after-series
    list_of_series = [s for tup in zip(mean, stdev, median) for s in tup]
    # distance_on_roi_time = get_distances_on_time_from_roi_time().rename('Distance_On_Event_Time')
    # list_of_series.append(distance_on_roi_time)

    res = pd.concat(list_of_series, axis=1)

    print("Calculating RoI features_extraction - DONE")
    print("\n*******************************\n")

    return res


def get_collapsed_distance_before_during_and_after_roi(statistic: str, num_stdevs: float = 2):
    """
    Foreach movie, splits the Distances Series into 3 sub-series: before-, during-, and after-RoI time.
        (Argument @num_stdevs is used to determine the duration of RoI - ending with t_RoI and beginning @num_stdevs before)
    Then, calculates the @statistic for each subject-session-movie, collapsing each sub-serie into a single value per tuple.
    These are then collected into a pd.Series indexed by subject-session-movie.
    $return ->
        Three (3) pd.Series with index names (Subject, Session, Movie), one serie for before/during/after the RoI time.
    """

    assert (statistic in fa_config.statistics), \
        f'Unknown statistic to compute. Allowed statistics are {fa_config.statistics}, \'{statistic}\' given.'

    distances_before, distances_during, distances_after = _cut_all_gaze_distances_by_movie_roi_time(num_stdevs)
        # sqrt_distances_before, sqrt_distances_during, sqrt_distances_after = \


    statistic_before = \
        _get_collapsed_gaze_distance_to_single_statistic_value(distances_before, statistic,
                                                               f'{statistic.capitalize()}_Distance_Before')
    statistic_during = \
        _get_collapsed_gaze_distance_to_single_statistic_value(distances_during, statistic,
                                                               f'{statistic.capitalize()}_Distance_During')
    statistic_after = \
        _get_collapsed_gaze_distance_to_single_statistic_value(distances_after, statistic,
                                                               f'{statistic.capitalize()}_Distance_After')

    # sqrt_statistic_before = \
    #     _get_collapsed_gaze_distance_to_single_statistic_value(sqrt_distances_before, statistic,
    #                                                             f'{statistic.capitalize()}_Sqrt_Distance_Before')
    # sqrt_statistic_during = \
    #     _get_collapsed_gaze_distance_to_single_statistic_value(sqrt_distances_during, statistic,
    #                                                             f'{statistic.capitalize()}_Sqrt_Distance_During')
    # sqrt_statistic_after = \
    #     _get_collapsed_gaze_distance_to_single_statistic_value(sqrt_distances_after, statistic,
    #                                                             f'{statistic.capitalize()}_Sqrt_Distance_After')

    return statistic_before, statistic_during, statistic_after, \
        # sqrt_statistic_before, sqrt_statistic_during, sqrt_statistic_after


def get_distances_on_time_from_roi_time(time_diff: float = None) -> pd.Series:
    """
    Extract the distance on a specific amount of SECONDS before/after the RoI-time.
    If argument time_diff is negative, will look before the RoI-time, and if positive - after.
    If any movie doesn't have a matching timestamp, that movie is excluded from the result.

    NOTE:
        1. Argument time_diff is in seconds, not ms or timestamps!
        2. Returned distances are for the closest timestamp preceding the requested time (i.e no more than 2ms before)
    @args ->
        time_diff: float; number of seconds before/after the RoI-time
    $return ->
        pd.Series with index names (Subject,Session,Movie)
    """

    time_diff = 0 if time_diff is None else time_diff
    timestamp_diff = time_diff * 1000

    distances_dict = dict()

    for mov in config.valid_movies:

        movie_distances, roi_time, _ = _get_movie_roi_data(mov)
        timestamp = utils.round_down_to_closest_even(roi_time + timestamp_diff, non_negative=False)

        if timestamp not in movie_distances.index.get_level_values(config.TIMESTAMP):
            distances_dict[mov] = pd.Series()

        distances_dict[mov] = movie_distances.xs(timestamp, level=config.TIMESTAMP).droplevel(config.MEMORY)

    series = pd.concat(distances_dict.values(), keys=distances_dict.keys(),
                       names=[config.MOVIE, config.SUBJECT, config.SESSION])

    return series.reorder_levels([config.SUBJECT, config.SESSION, config.MOVIE]
                                 ).rename(f'Distances {time_diff:.2f} Seconds from RoI Time')


def _cut_all_gaze_distances_by_movie_roi_time(num_stdevs: float = 2):
    all_distances_before, all_distances_during, all_distances_after = dict(), dict(), dict()
        # all_sqrt_distances_before, all_sqrt_distances_during, all_sqrt_distances_after = dict(), dict(), dict()

    for mov in config.valid_movies:

        distances_before_roi, distances_during_roi, distances_after_roi = _split_distances_series_before_during_and_after_roi(mov, num_stdevs)
            # sqrt_distances_before_roi, sqrt_distances_during_roi, sqrt_distances_after_roi


        if not distances_before_roi.empty:
            all_distances_before[mov] = distances_before_roi
            # all_sqrt_distances_before[mov] = sqrt_distances_before_roi
        if not distances_during_roi.empty:
            all_distances_during[mov] = distances_during_roi
            # all_sqrt_distances_during[mov] = sqrt_distances_during_roi
        if not distances_after_roi.empty:
            all_distances_after[mov] = distances_after_roi
            # all_sqrt_distances_after[mov] = sqrt_distances_after_roi

    distances_before = _convert_distances_dict_to_series(all_distances_before)
    distances_during = _convert_distances_dict_to_series(all_distances_during)
    distances_after = _convert_distances_dict_to_series(all_distances_after)
    # sqrt_distances_before = _convert_distances_dict_to_series(all_sqrt_distances_before)
    # sqrt_distances_during = _convert_distances_dict_to_series(all_sqrt_distances_during)
    # sqrt_distances_after = _convert_distances_dict_to_series(all_sqrt_distances_after)

    return distances_before, distances_during, distances_after
        # sqrt_distances_before, sqrt_distances_during, sqrt_distances_after


def _get_collapsed_gaze_distance_to_single_statistic_value(distances: pd.Series, statistic: str,
                                                           new_name: str = None) -> pd.Series:
    """
    For each subject-session-movie returns a single measure of distance
    """

    assert (statistic in fa_config.statistics), \
        f'Unknown statistic to compute. Allowed statistics are {config.statistics}, \'{statistic}\' given.'

    if len(distances) == 0:
        result = pd.Series()
    else:
        grouped = distances.groupby(level=[config.SUBJECT,
                                           config.SESSION,
                                           config.MOVIE])
        result = {'mean': grouped.mean(), 'median': grouped.median(), 'std': grouped.std(),
                  'sem': grouped.sem(), 'auc': grouped.sum()}[statistic]

    if new_name is None or new_name == '':
        return result.rename(statistic.capitalize())

    return result.rename(new_name)


def _get_movie_roi_data(movie: str) -> (pd.Series, float, float):
    assert (movie in config.valid_movies), f'No distances available for given movie {movie}.'

    movie_distances = DataService.gaze[config.DVA].xs(movie, level=config.MOVIE)
    # movie_sqrt_distances = DataService.gaze[config.SQRT_DVA].xs(movie, level=config.MOVIE)

    roi_median_time, roi_stdev_time = DataService.rois.loc[movie, [config.T_MEDIAN, config.T_STDEV]]

    # return movie_distances, movie_sqrt_distances, roi_median_time, roi_stdev_time
    return movie_distances, roi_median_time, roi_stdev_time


def _split_distances_series_before_during_and_after_roi(mov: str, num_stdevs: float = 2) -> (pd.Series,
                                                                                             pd.Series, pd.Series):
    movie_distances, roi_time, roi_std_time = _get_movie_roi_data(mov)
    distances_start_timestamp = movie_distances.index.get_level_values(config.TIMESTAMP).min()
    distances_end_timestamp = movie_distances.index.get_level_values(config.TIMESTAMP).max()

    # roibased start/end time are at least as low/high as the start/end of the distances series time
    roi_start_time = max(distances_start_timestamp,
                         utils.round_down_to_closest_even(roi_time))
    roi_end_time = min(utils.round_down_to_closest_even(roi_time + num_stdevs * roi_std_time), distances_end_timestamp)

    distances_before_roi = utils.split_series(movie_distances, start=distances_start_timestamp,
                                              end=roi_start_time, by=config.TIMESTAMP)
    distances_during_roi = utils.split_series(movie_distances, start=roi_start_time,
                                              end=roi_end_time, by=config.TIMESTAMP)
    distances_after_roi = utils.split_series(movie_distances, start=roi_end_time,
                                             end=distances_end_timestamp, by=config.TIMESTAMP)

    # sqrt_distances_before_roi = utils.split_series(movie_sqrt_distances, start=distances_start_timestamp,
    #                                                end=roi_start_time, by=config.TIMESTAMP)
    # sqrt_distances_during_roi = utils.split_series(movie_sqrt_distances, start=roi_start_time,
    #                                                end=roi_end_time, by=config.TIMESTAMP)
    # sqrt_distances_after_roi = utils.split_series(movie_sqrt_distances, start=roi_end_time,
    #                                               end=distances_end_timestamp, by=config.TIMESTAMP)

    # return (distances_before_roi, distances_during_roi, distances_after_roi,
    #         sqrt_distances_before_roi, sqrt_distances_during_roi, sqrt_distances_after_roi)
    return distances_before_roi, distances_during_roi, distances_after_roi


def _convert_distances_dict_to_series(distances_dict: dict) -> pd.Series:
    if not distances_dict:
        return pd.Series()

    series = pd.concat(distances_dict.values(), keys=distances_dict.keys())
    series.index.names = [config.MOVIE,
                          config.SUBJECT,
                          config.SESSION,
                          config.MEMORY,
                          config.TIMESTAMP]

    return series.reorder_levels([config.SUBJECT,
                                  config.SESSION,
                                  config.MOVIE,
                                  config.MEMORY,
                                  config.TIMESTAMP]
                                 ).sort_index()
