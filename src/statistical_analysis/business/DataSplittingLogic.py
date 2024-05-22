import pandas as pd

from src import config as g_config
from src.statistical_analysis import config
from src.statistical_analysis.business.DataMatchingLogic import match_subjects_and_movies

'''
This component is used to split a DataFrame/Series by some index/es,
    e.g. split(data, 'mem') to get 2 dataframes,
    one for subjects that said remembered the data and the other for those who said they did not

IMPORTANT: for the data_partitioning logic to work properly, please maintain the following rules:
    - Session A data always precedes Session B data
    - Remembered before Not Remembered
    - High confidence before Low confidence
'''

global dataset


def split(data, *args):
    split_by = set(args)
    assert (split_by.issubset({'ses', 'mem', 'conf', 'high_conf_mem'})
            ), f'Can only split the data by Session (\'ses\'), Memory Report (\'mem\'), or Confidence (\'conf\')'
    global dataset
    dataset = data

    if ('ses' in split_by) and ('mem' in split_by) and ('conf' in split_by):
        raise AttributeError(f"Splitting the data by {split_by} is not supported.")
        # return split_by_session_memory_and_confidence()
    elif ('ses' in split_by) and ('mem' in split_by):
        return _split_by_session_and_memory()
    elif ('ses' in split_by) and ('conf' in split_by):
        return _split_by_session_and_confidence()
    elif 'ses' in split_by:
        return _split_by_sessions()
    elif 'mem' in split_by:
        return _split_by_memory_response()
    elif 'high_conf_mem' in split_by:
        return _split_by_high_confidence_memory_response()
    elif 'conf' in split_by:
        return _split_by_confidence()
    raise AttributeError(f"Splitting the data by {split_by} is not supported.")


def _split_by_sessions():
    sesA = __get_data_by_level_value(g_config.SESSION, g_config.SESSION_A)
    sesB = __get_data_by_level_value(g_config.SESSION, g_config.SESSION_B)
    return sesA, sesB


def _split_by_memory_response():
    remembered = __try_concat(__get_data_by_level_value(g_config.MEMORY, 4),
                              __get_data_by_level_value(g_config.MEMORY, 3),
                              __get_data_by_level_value(g_config.MEMORY, 2),
                              __get_data_by_level_value(g_config.MEMORY, 1))
    not_remembered = __try_concat(__get_data_by_level_value(g_config.MEMORY, -4),
                                  __get_data_by_level_value(g_config.MEMORY, -3),
                                  __get_data_by_level_value(g_config.MEMORY, -2),
                                  __get_data_by_level_value(g_config.MEMORY, -1))
    return remembered, not_remembered


def _split_by_high_confidence_memory_response():
    remembered = __try_concat(__get_data_by_level_value(g_config.MEMORY, 4),
                              __get_data_by_level_value(g_config.MEMORY, 3))
    not_remembered = __try_concat(__get_data_by_level_value(g_config.MEMORY, -4),
                                  __get_data_by_level_value(g_config.MEMORY, -3))
    return remembered, not_remembered


def _split_by_confidence():
    high_conf = __try_concat(__get_data_by_level_value(config.MEMORY, 4),
                             __get_data_by_level_value(config.MEMORY, 3),
                             __get_data_by_level_value(config.MEMORY, -4),
                             __get_data_by_level_value(config.MEMORY, -3))
    low_conf = __try_concat(__get_data_by_level_value(config.MEMORY, 2),
                            __get_data_by_level_value(config.MEMORY, 1),
                            __get_data_by_level_value(config.MEMORY, -2),
                            __get_data_by_level_value(config.MEMORY, -1))
    return high_conf, low_conf


def _split_by_session_and_memory():
    sesA, sesB = _split_by_sessions()
    sesB_remembered, sesB_not_remembered = split(sesB, 'mem')
    sesA_remembered = match_subjects_and_movies(sesB_remembered,
                                                sesA)  # extract A-distances for subject who remembered in B
    sesA_not_remembered = match_subjects_and_movies(sesB_not_remembered,
                                                    sesA)  # extract A-distances for subject who did not remember in B
    return sesA_remembered, sesA_not_remembered, sesB_remembered, sesB_not_remembered


def _split_by_session_and_confidence():
    sesA, sesB = _split_by_sessions()
    sesA_high, sesA_low = split(sesA, 'conf')
    sesB_high, sesB_low = split(sesB, 'conf')
    return sesA_high, sesA_low, sesB_high, sesB_low


def split_by_session_memory_and_confidence():
    # TODO: need to implement this better!
    sesA_remembered, sesA_not_remembered, sesB_remembered, sesB_not_remembered = _split_by_session_and_memory()
    sesA_remembered_high, sesA_remembered_low = split(sesA_remembered, 'conf')
    sesA_not_remembered_high, sesA_not_remembered_low = split(sesA_not_remembered, 'conf')
    sesB_remembered_high, sesB_remembered_low = split(sesB_remembered, 'conf')
    sesB_not_remembered_high, sesB_not_remembered_low = split(sesB_not_remembered, 'conf')
    return (sesA_remembered_high, sesA_remembered_low, sesA_not_remembered_high, sesA_not_remembered_low,
            sesB_remembered_high, sesB_remembered_low, sesB_not_remembered_high, sesB_not_remembered_low)


def __get_data_by_level_value(level: str, level_value: str or int):
    try:
        return dataset.xs(level_value, level=level)
    except KeyError:
        return pd.Series()


def __try_concat(*args):
    non_pandas_types = list(filter(lambda typ: ((typ is not pd.Series)
                                                and (typ is not pd.DataFrame)),
                                   [type(elem) for elem in args]))
    assert (len(non_pandas_types) == 0
            ), f'All concatenated objects must be of type pd.Series or pd.DataFrame, {non_pandas_types} given'
    non_empty = list(filter(lambda elem: len(elem) > 0, args))
    try:
        return pd.concat(non_empty)
    except ValueError:
        return pd.Series()
