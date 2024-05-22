import pandas as pd

from src.utils import cut_series


class DistanceDifferenceCalculatorService:

    def __init__(self, distances_subtractee: pd.Series, distances_subtractor: pd.Series):
        assert (type(
            distances_subtractee) is pd.Series), f'The argument @distances_subtractee must be an instance of Series.'
        assert (type(
            distances_subtractor) is pd.Series), f'The argument @distances_subtractor must be an instance of Series.'
        self.subtractee = distances_subtractee
        self.subtractor = distances_subtractor
        return

    def calculate_mean_difference_over_time(self, from_timestamp: int = None, to_timestamp: int = None,
                                            filler: float = None):
        distance_diffs = self.calculate_differences(from_timestamp, to_timestamp, filler)
        return distance_diffs.groupby(level=[self.TIMESTAMP]).mean()

    def calculate_sem_difference_over_time(self, from_timestamp: int = None, to_timestamp: int = None,
                                           filler: float = None):
        distance_diffs = self.calculate_differences(from_timestamp, to_timestamp, filler)
        subjects_sem_over_time = distance_diffs.groupby(level=[self.SUBJECT, self.TIMESTAMP]).sem()
        return subjects_sem_over_time.groupby(level=self.TIMESTAMP).mean()

    def calculate_sum_of_mean_difference(self, from_timestamp: int = None, to_timestamp: int = None,
                                         filler: float = None) -> float:
        differences = self.calculate_differences(from_timestamp, to_timestamp, filler)
        mean_diff_per_subject = differences.groupby(level=[self.SUBJECT, self.TIMESTAMP]).mean()
        sum_per_subj = mean_diff_per_subject.groupby(level=self.SUBJECT).sum()
        # sum_per_subj.mean(), sum_per_subj.std(), sum_per_subj.sem()
        return sum_per_subj.mean()

    def calculate_differences(self, from_timestamp: int = None, to_timestamp: int = None, filler: float = None):
        import numpy as np
        subtractee, subtractor = self._match_index_levels(self.subtractee, self.subtractor)
        filler = np.nan if (filler is None) else filler
        differences = (subtractee.subtract(subtractor, level=[self.SUBJECT, self.MOVIE, self.TIMESTAMP],
                                           fill_value=filler)).dropna()
        if (len(differences) <= 0):
            return pd.Series()
        min_timestamp, max_timestamp = differences.index.get_level_values(
            self.TIMESTAMP).min(), differences.index.get_level_values(self.TIMESTAMP).max()
        from_timestamp = min_timestamp if (from_timestamp is None) else from_timestamp
        to_timestamp = max_timestamp if (to_timestamp is None) else to_timestamp
        return cut_series(differences, from_timestamp, to_timestamp, self.TIMESTAMP)

    @staticmethod
    def _match_index_levels(subtractee, subtractor):
        import numpy as np
        import pandas as pd
        assert (
                subtractee.index.names == subtractor.index.names), f'Index mis-match: objects @subtractor and @subtractee does not have matching index names.'
        in_subtractor_not_subtractee = set(subtractor.index) - set(subtractee.index)
        in_subtractee_not_subtractor = set(subtractee.index) - set(subtractor.index)

        subtractee = subtractee.append(pd.Series([np.nan for idx in in_subtractor_not_subtractee],
                                                 index=in_subtractor_not_subtractee)).sort_index()
        subtractor = subtractor.append(pd.Series([np.nan for idx in in_subtractee_not_subtractor],
                                                 index=in_subtractee_not_subtractor)).sort_index()
        return subtractee, subtractor
