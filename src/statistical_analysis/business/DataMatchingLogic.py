import pandas as pd

from src import config

'''
This component is used to get a sub-Series / sub-DataFrame of @other, that contains the exact same index-tuples of 
    The DataFrame/Series @this.
'''


def match_subjects_and_movies(this, other):
    if len(this) <= 0:
        return pd.Series()
    assert (config.SUBJECT in other.index.names), f'No index level named {config.SUBJECT} in @this.'
    assert (config.MOVIE in other.index.names), f'No index level named {config.MOVIE} in @this.'

    relevant_couples = set(zip(this.index.get_level_values(config.SUBJECT),
                               this.index.get_level_values(config.MOVIE)))
    subseries_of_other = {(subj, mov): other.xs([subj, mov], level=[config.SUBJECT, config.MOVIE])
                          for subj, mov in relevant_couples}
    result = pd.concat(subseries_of_other.values(), keys=subseries_of_other.keys())
    result.index.names = [config.SUBJECT, config.MOVIE, config.MEMORY, config.TIMESTAMP]
    return result
