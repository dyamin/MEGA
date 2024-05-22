# TODO: implement using scikits.bootstrap library

import itertools
import random

import numpy as np
import pandas as pd

from src.statistical_analysis import config
from src.utils import cut_series

'''
Create a bootstrapping-distribution based on the provided @differences series.
In each iteration we randomly choose half of the sample, and flip their sign (from - to + or vice versa).
    We randomize by specifying a randomization-level. We can also choose to specify a constant-level, adding a constraint
    that means that for each value in the const-level, we randomize the rand-level separately.
@Args ->
    differences - a pd.Series of differences between two series (e.g. Session A - Session B)
    aggregator - the type of aggregation to perform on the resulting Series
    levels - a list of length <=> 2, specifying (optionally) the randomization level and (optionally) the constraint-level
    from_timestamp, to_timestamp - if specified, we only perform the aggregation between these 2 timestamps

$Returns ->
    a pd.Series matching each iteration number to the returned aggregated value of this iteration's bootstrapped Series
'''


def gen_bootstrap(differences: pd.Series, aggregator: str, levels: list,
                  from_timestamp: int = None, to_timestamp: int = None) -> pd.Series:
    assert (levels is None or len(levels) <= 2
            ), f'Too many levels to bootstrap by, max is 2.'
    if levels is not None and len(levels):
        assert (all([level in differences.index.names for level in levels])
                ), f'The provided levels ({levels}) are not in the provided differences index names ({differences.index.names})'
    randomize_level = _get_bootstrapping_level(levels[0]) if len(levels) else None
    constant_level = _get_bootstrapping_level(levels[1]) if (len(levels) == 2) else None
    agg_func = get_bootstrapping_agg_func(aggregator)

    bootstrap_dist = _bootstrap_impl(differences,
                                     randomize_level=randomize_level, constant_level=constant_level,
                                     aggregator_function=agg_func,
                                     from_timestamp=from_timestamp, to_timestamp=to_timestamp)
    return bootstrap_dist


def _get_bootstrapping_level(level: str):
    if level.lower() == 'movie':
        return config.MOVIE
    elif level.lower() == 'subject':
        return config.SUBJECT
    raise AttributeError(f"Bootstrapping by level {level} is not supported.")


def get_bootstrapping_agg_func(agg: str):
    if agg == 'area':
        return np.nansum
    elif agg == 'mean':
        return np.nanmean
    elif agg == 'median':
        return np.nanmedian
    elif agg == 'std':
        return np.nanstd
    raise AttributeError(f"Bootstrapping with aggregation function {agg} is not supported.")


def _bootstrap_impl(differences, randomize_level: str, constant_level: str, aggregator_function,
                    from_timestamp: int = None, to_timestamp: int = None) -> pd.Series:
    assert (config.NUM_ITERATIONS > 0
            ), f'Must provide a positive integer for argument num_iterations, {config.NUM_ITERATIONS} provided.'
    assert (randomize_level in differences.index.names
            ), f'The level {randomize_level} is not in the MultiIndex.'

    # create the series we randomize
    if (randomize_level is None) and (constant_level is None):
        randomization_index_values = pd.Series(zip(differences.index.get_level_values(config.SUBJECT),
                                                   differences.index.get_level_values(config.MOVIE)))
    elif randomize_level is not None:
        assert (randomize_level in differences.index.names
                ), f'The level {randomize_level} is not in the MultiIndex.'
        if constant_level is None:
            randomization_index_values = pd.Series(zip(differences.index.get_level_values(randomize_level)))
        else:
            assert (constant_level in differences.index.names), f'The level {constant_level} is not in the MultiIndex.'
            randomization_index_values = pd.Series(zip(differences.index.get_level_values(constant_level),
                                                       differences.index.get_level_values(randomize_level)))
    else:
        raise AssertionError("Must Specify a level to randomize over if you want to use a constraint constant level.")
    randomization_index_values.index = differences.index  # make sure index is the same

    # initiate necessary values:
    results = dict()
    from_timestamp = differences.index.get_level_values(config.TIMESTAMP).min() if (
            from_timestamp is None) else from_timestamp
    to_timestamp = differences.index.get_level_values(config.TIMESTAMP).max() if (
            to_timestamp is None) else to_timestamp

    for i in range(1, config.NUM_ITERATIONS + 1):
        if (randomize_level is None) and (constant_level is None):
            random_mapping = __randomize_all_subject_and_movies(differences)
        else:
            random_mapping = __randomize_level(differences, randomize_level) if (constant_level is None) \
                else __randomize_level_for_constant_level(differences, constant_level, randomize_level)

        new_differences = differences.mul(randomization_index_values.apply(lambda tup: random_mapping[tup]))
        new_differences = cut_series(new_differences, from_timestamp, to_timestamp, config.TIMESTAMP)
        new_differences = new_differences.groupby(level=[config.TIMESTAMP]).mean()
        results[i] = aggregator_function(new_differences)
        if (config.verbose_every is not None) and (config.verbose_every > 0) and (i % config.verbose_every == 0):
            print(f'\tFinished {i} iterations.')
    return pd.Series(results)


def __randomize_all_subject_and_movies(differences) -> dict:
    ''' Returns +1 or -1 with p=0.5 for each (Subject, Movie) tuple '''
    idx = list(itertools.product(differences.index.unique(config.SUBJECT), differences.index.unique(config.MOVIE)))
    rand_boolean = np.random.rand(len(idx)) >= 0.5
    rand_boolean = rand_boolean + (rand_boolean - 1)
    return dict(pd.Series(rand_boolean, index=pd.MultiIndex.from_tuples(
        idx, names=[config.SUBJECT, config.MOVIE])).to_dict())


def __randomize_level(differences, level: str) -> dict:
    ''' Returns +1 for half of the level values, and -1 for the other half '''
    assert (level in differences.index.names), f'The level {level} is not in the MultiIndex.'
    d = dict()
    elements = list(differences.index.unique(level))
    subset1, subset2 = __split_random_halves(elements)
    d.update({(elem,): 1 for elem in subset1})
    d.update({(elem,): -1 for elem in subset2})
    return d


def __randomize_level_for_constant_level(differences, const_level: str, rand_level: str) -> dict:
    ''' Returns +1 for half of rand_level values and -1 for the other half,
            randomly assigning for each of the const_level values '''
    assert (rand_level in differences.index.names), f'The level {rand_level} is not in the MultiIndex.'
    assert (const_level in differences.index.names), f'The level {const_level} is not in the MultiIndex.'
    d = dict()
    constant_level_values = list(differences.index.unique(const_level))
    rand_level_values = list(differences.index.unique(rand_level))
    for const_elem in constant_level_values:
        subset1, subset2 = __split_random_halves(rand_level_values)
        d.update({(const_elem, elem): 1 for elem in subset1})
        d.update({(const_elem, elem): -1 for elem in subset2})
    return d


def __split_random_halves(lst: list) -> (list, list):
    random.shuffle(lst)
    return lst[: len(lst) // 2], lst[len(lst) // 2:]
