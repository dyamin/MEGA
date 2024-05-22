## TODO: Use library scikits.bootstrap
import pandas as pd

from src.signal_processing.distance_from_roi.services.DistanceDifferenceCalculatorService import \
    DistanceDifferenceCalculatorService
from src.utils import cut_series


class DistanceBootstrappingService:

    def __init__(self, original_subtractee, original_subtractor):
        diff_calc = DistanceDifferenceCalculatorService(original_subtractee, original_subtractor)
        self.differences = diff_calc.calculate_differences()
        return

    def bootstrap_area_randomize_all(self, num_iterations: int = 100,
                                     from_timestamp: int = None, to_timestamp: int = None,
                                     verbose_every: int = None):
        import numpy as np
        return self._bootstrap_impl(randomize_level=None, constant_level=None,
                                    aggregator_function=np.nansum, num_iterations=num_iterations,
                                    from_timestamp=from_timestamp, to_timestamp=to_timestamp,
                                    verbose_every=verbose_every)

    def bootstrap_area_randomize_subjects(self, num_iterations: int = 100,
                                          from_timestamp: int = None, to_timestamp: int = None,
                                          verbose_every: int = None):
        '''
        Bootstrap the self.difference data and returns a pd.Series matching each iteration with the area between two random curves.
        We keep a constraint so that on every iteration half of the subjects are sesA-sesB and half are sesB-sesA.
        @args ->
            num_iterations: int (positive); Number of iterations to run.
            from_timestamp, to_timestamp: int (positive, even); If specified, the bootstrapped data runs only between these timestamps.
            verbose_every: int; If specified, will print a message every @verbose_every iterations.
        '''
        import numpy as np
        return self._bootstrap_impl(randomize_level=self.SUBJECT, constant_level=None,
                                    aggregator_function=np.nansum, num_iterations=num_iterations,
                                    from_timestamp=from_timestamp, to_timestamp=to_timestamp,
                                    verbose_every=verbose_every)

    def bootstrap_area_randomize_movies_within_subject(self, num_iterations: int = 100,
                                                       from_timestamp: int = None, to_timestamp: int = None,
                                                       verbose_every: int = None):
        '''
        Bootstrap the self.difference data and returns a pd.Series matching each iteration with the area between two random curves.
        We keep a constraint so that on every iteration and for each subject,
            half of the movies are sesA-sesB and half are sesB-sesA.
        @args ->
            num_iterations: int (positive); Number of iterations to run.
            from_timestamp, to_timestamp: int (positive, even); If specified, the bootstrapped data runs only between these timestamps.
            verbose_every: int; If specified, will print a message every @verbose_every iterations.
        '''
        import numpy as np
        return self._bootstrap_impl(randomize_level=self.MOVIE, constant_level=self.SUBJECT,
                                    aggregator_function=np.nansum, num_iterations=num_iterations,
                                    from_timestamp=from_timestamp, to_timestamp=to_timestamp,
                                    verbose_every=verbose_every)

    def _bootstrap_impl(self, randomize_level: str, constant_level: str, aggregator_function,
                        num_iterations: int = 100, from_timestamp: int = None, to_timestamp: int = None,
                        verbose_every: int = None):
        assert (
                num_iterations > 0), f'Must provide a positive integer for arument num_iterations, {num_iterations} provided.'

        # create the series we randomize
        if ((randomize_level is None) and ((constant_level is None))):
            randomization_index_values = pd.Series(zip(self.differences.index.get_level_values(self.SUBJECT),
                                                       self.differences.index.get_level_values(self.MOVIE)))
        elif (randomize_level is not None):
            assert (
                    randomize_level in self.differences.index.names), f'The level {randomize_level} is not in the MultiIndex.'
            if (constant_level is None):
                randomization_index_values = pd.Series(zip(self.differences.index.get_level_values(randomize_level)))
            else:
                assert (
                        constant_level in self.differences.index.names), f'The level {constant_level} is not in the MultiIndex.'
                randomization_index_values = pd.Series(zip(self.differences.index.get_level_values(constant_level),
                                                           self.differences.index.get_level_values(randomize_level)))
        else:
            raise AssertionError(
                "Must Specify a level to randomize over if you want to usea constraint constant level.")
        randomization_index_values.index = self.differences.index  # make sure index is the same

        # initiate necessary values:
        results = dict()
        from_timestamp = self.differences.index.get_level_values(self.TIMESTAMP).min() if (
                from_timestamp is None) else from_timestamp
        to_timestamp = self.differences.index.get_level_values(self.TIMESTAMP).max() if (
                to_timestamp is None) else to_timestamp

        for i in range(1, num_iterations + 1):
            if ((randomize_level is None) and ((constant_level is None))):
                random_mapping = self.__randomize_all_subject_and_movies()
            else:
                random_mapping = self.__randomize_level(randomize_level) if (
                        constant_level is None) else self.__randomize_level_for_constant_level(constant_level,
                                                                                               randomize_level)

            new_differences = self.differences.mul(randomization_index_values.apply(lambda tup: random_mapping[tup]))
            new_differences = cut_series(new_differences, from_timestamp, to_timestamp, self.TIMESTAMP)
            new_differences = new_differences.groupby(level=[self.TIMESTAMP]).mean()
            results[i] = aggregator_function(new_differences)
            if ((verbose_every is not None) and (verbose_every > 0) and (i % verbose_every == 0)):
                print(f'\tFinished {i} iterations.')
        return pd.Series(results)

    def __randomize_all_subject_and_movies(self) -> dict:
        import itertools
        import numpy as np
        idx = list(
            itertools.product(self.differences.index.unique(self.SUBJECT), self.differences.index.unique(self.MOVIE)))
        rand_boolean = np.random.rand(len(idx)) >= 0.5
        rand_boolean = rand_boolean + (rand_boolean - 1)
        return pd.Series(rand_boolean, index=pd.MultiIndex.from_tuples(idx, names=[self.SUBJECT, self.MOVIE])).to_dict()

    def __randomize_level(self, level: str) -> dict:
        ''' Returns +1 for half of the level values, and -1 for the other half '''
        assert (level in self.differences.index.names), f'The level {level} is not in the MultiIndex.'
        d = dict()
        elements = list(self.differences.index.unique(level))
        subset1, subset2 = self._split_random_halves(elements)
        d.update({(elem,): 1 for elem in subset1})
        d.update({(elem,): -1 for elem in subset2})
        return d

    def __randomize_level_for_constant_level(self, const_level: str, rand_level: str) -> dict:
        ''' Returns +1 for half of rand_level values and -1 for the other half,
                randomly assigning for each of the const_level values '''
        assert (rand_level in self.differences.index.names), f'The level {rand_level} is not in the MultiIndex.'
        assert (const_level in self.differences.index.names), f'The level {const_level} is not in the MultiIndex.'
        d = dict()
        constant_level_values = list(self.differences.index.unique(const_level))
        rand_level_values = list(self.differences.index.unique(rand_level))
        for const_elem in constant_level_values:
            subset1, subset2 = self.__split_random_halves(rand_level_values)
            d.update({(const_elem, elem): 1 for elem in subset1})
            d.update({(const_elem, elem): -1 for elem in subset2})
        return d

    @staticmethod
    def __split_random_halves(lst: list) -> (list, list):
        import random
        random.shuffle(lst)
        return lst[: len(lst) // 2], lst[len(lst) // 2:]
