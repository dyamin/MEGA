import numpy as np
import pandas as pd

from src.statistical_analysis import config
from src.utils import cut_series


def calculate_differences(subtract_from, subtractor,
                          from_timestamp: int = None, to_timestamp: int = None,
                          filler: float = config.subtraction_nan_filler):
    subtract_from, subtractor = _match_index_levels(subtract_from, subtractor)
    filler = np.nan if (filler is None) else filler
    differences = subtract_from.subtract(subtractor,
                                         level=[config.SUBJECT, config.MOVIE, config.TIMESTAMP],
                                         fill_value=filler).dropna()
    if len(differences) <= 0:
        return pd.Series()

    min_timestamp = differences.index.get_level_values(config.TIMESTAMP).min()
    max_timestamp = differences.index.get_level_values(config.TIMESTAMP).max()
    from_timestamp = min_timestamp if (from_timestamp is None) else from_timestamp
    to_timestamp = max_timestamp if (to_timestamp is None) else to_timestamp
    return cut_series(differences, from_timestamp, to_timestamp, config.TIMESTAMP)


def _match_index_levels(subtract_from, subtractor):
    assert (subtract_from.index.names == subtractor.index.names
            ), f'Index mis-match: objects @subtractor and @subtract_from does not have matching index names.'
    in_subtractor_not_subtract_from = set(subtractor.index) - set(subtract_from.index)
    in_subtract_from_not_subtractor = set(subtract_from.index) - set(subtractor.index)

    subtract_from = subtract_from.append(
        pd.Series([np.nan for idx in in_subtractor_not_subtract_from],
                  index=in_subtractor_not_subtract_from)).sort_index()
    subtractor = subtractor.append(
        pd.Series([np.nan for idx in in_subtract_from_not_subtractor],
                  index=in_subtract_from_not_subtractor)).sort_index()
    return subtract_from, subtractor
