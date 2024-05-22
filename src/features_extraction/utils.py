from typing import Tuple

import pandas as pd

from src import config
from src.features_extraction import config as features_config
from src.features_extraction.models.Point import Point
from src.features_extraction.models.Rectangle import Rectangle
from src.features_extraction.models.Roi import Roi
from src.features_extraction.services import DataService


def split_data(data: pd.DataFrame, start: int, end: int, by: str, index: bool = False) -> pd.DataFrame:
    assert (start <= end), f'Argument {start} must be lesser/equal to {end}.'

    if start == end:
        return pd.DataFrame(index=data.index)

    if index:
        data = data.reset_index(by)
    assert (by in data.columns), f'Couldn\'t find {by} in the given data\'s columns.'

    if start > 0:
        splited = data[data[by] >= start]
    else:
        splited = data
    splited = splited[splited[by] <= end]

    if index:
        splited.set_index(by, append=True, inplace=True)

    return splited


def split_series(series: pd.Series, start: int, end: int, by: str) -> pd.Series:
    assert (start <= end), f'Argument @first_value must be lesser/equal to @last_value.'
    assert (by in series.index.names), f'Couldn\'t find {by} in the given Serie\'s index names.'

    if start == end:
        return pd.Series()

    serie_as_dataframe = series.reset_index(by)

    cut_dataframe = serie_as_dataframe[
        (serie_as_dataframe[by] >= start) & (serie_as_dataframe[by] <= end)]
    cut_dataframe.set_index(by, append=True, inplace=True)
    cut_series = cut_dataframe[
        cut_dataframe.columns[0]]  # treat this as a Series by taking the first (and only) column of the DF

    return cut_series


def is_in_rectangle(rect: Rectangle, point: Point) -> bool:
    """
    Returns True iff the point is within the rectangle defined by two opposite vertices
    """

    tl, br = rect.tl, rect.br

    in_x_interval = (point.x >= tl.x) and (point.x <= br.x)
    in_y_interval = (point.y >= tl.y) and (point.y <= br.y)

    return in_x_interval and in_y_interval


def dfs_dict_to_df(dfs: dict) -> pd.DataFrame:
    levels_names = [config.MOVIE, config.SUBJECT, config.SESSION]
    levels_order = [config.SUBJECT, config.SESSION, config.MOVIE]

    return pd.concat(dfs.values(), keys=dfs.keys(), names=levels_names).reorder_levels(levels_order)


def handle_empty_movie_data(names: Tuple[str, str], count: int = 1) -> pd.Series or tuple:
    # Match index to original data's indices names
    index = pd.MultiIndex(levels=[[], []],
                          codes=[[], []],
                          names=tuple(names))

    empty = pd.Series(index=index)

    if count == 1:
        return empty
    else:
        empties = list()
        for i in range(count):
            empties.append(empty)
        return tuple(empties)


def unite_series_to_df(pre: pd.Series, post: pd.Series, metric: str, ttype: str) -> pd.DataFrame:
    if metric == "ratio":
        first = "In_out"
        second = "_Ratio"
    elif metric == "counts in":
        first = "Counts_In"
        second = ""
    elif metric == "counts out":
        first = "Counts_Out"
        second = ""
    elif metric == "dwell time":
        first = "Dwell_Time_In"
        second = ""
    elif metric == "dwell time ratio":
        first = "Dwell_Time_In"
        second = "_Ratio"
    elif metric == "re entries counts":
        first = "Re_Entries_To"
        second = "_Count"
    elif metric == "re entries rates":
        first = "Re_Entries_To"
        second = "_Rate"
    else:
        raise ValueError(f"Metric {metric} is not supported")

    return pd.concat([pre.rename(f"{ttype}_{first}_RoI{second}_Pre"),
                      post.rename(f"{ttype}_{first}_RoI{second}_Post")],
                     axis=1).fillna(0)


def get_location_params(ttype: str) -> (str, str):
    if ttype == "Gaze":
        return config.gaze_X, config.gaze_Y
    elif ttype == "Fixations":
        return config.CoM_X, config.CoM_Y
    elif ttype == "Saccades_Start":
        return config.X_START, config.Y_START
    elif ttype == "Saccades_End":
        return config.X_END, config.Y_END
    else:
        raise ValueError(f"Location for data from type {ttype} is not supported")


def get_time_params(ttype: str) -> (str, bool):
    if ttype == "Fixations":
        index = False
        return config.ONSET, index
    elif ttype == "Gaze":
        index = True
        return config.TIMESTAMP, index
    if ttype == "Saccades_Start":
        index = False
        return config.ONSET, index
    if ttype == "Saccades_End":
        index = False
        return config.LAST_ONSET, index
    else:
        raise ValueError(f"Location for data from type {ttype} is not supported")


def get_feature_columns(ttype: str) -> (str, str):
    onset = config.ONSET if config.POPULATION != 'yoavdata' else 'Start_Time'
    if ttype == "Fixations":
        return [onset, config.DURATION]
    elif ttype == "Saccades_Start":
        return [onset, config.DURATION, config.AMPLITUDE, config.VELOCITY]
    elif ttype == "Saccades_End":
        return [config.LAST_ONSET, config.DURATION, config.AMPLITUDE, config.VELOCITY]
    elif ttype == "Gaze":
        return [config.DVA]
    else:
        raise ValueError(f"Location for data from type {ttype} is not supported")


def get_segment_duration(segment: str, roi: Roi, mov: str) -> float:
    if segment == "pre":
        duration = roi.start
    elif segment == "post":
        duration = DataService.videos_dims.loc[mov, features_config.duration] - roi.end
    else:
        raise ValueError(f"Time segment {segment} is not supported for re-entries calculation")

    return duration


def round_down_to_closest_even(num: float, non_negative=True) -> int:
    if non_negative and num <= 0:
        return 0

    num = int(num)

    if num % 2 == 1:
        num -= 1

    return num


def init_time_columns(df, baseline_time):
    # baseline correction for time columns if exists
    if config.ONSET in df.columns:
        df[config.ONSET] = df[config.ONSET] - baseline_time
    if config.LAST_ONSET in df.columns:
        df[config.LAST_ONSET] = df[config.LAST_ONSET] - baseline_time
    return df
