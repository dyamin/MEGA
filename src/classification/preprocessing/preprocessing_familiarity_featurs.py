import os

import numpy as np
import pandas as pd

from src import config
from src.classification.config import ID_COLUMN

features_df = pd.read_pickle(os.path.join(config.statistical_analysis_resource_dir, 'features.pkl'))

# get sessions A and B by level values
sesA, sesB = features_df[features_df.index.get_level_values(config.SESSION) == config.SESSION_A], \
    features_df[features_df.index.get_level_values(config.SESSION) == config.SESSION_B]

# get couples of subjects and movies that are in both sessions
relevant_couples = set(
    zip(sesA.index.get_level_values(config.SUBJECT), sesA.index.get_level_values(config.MOVIE))).intersection(
    set(zip(sesB.index.get_level_values(config.SUBJECT), sesB.index.get_level_values(config.MOVIE))))

normalized_df = pd.DataFrame(columns=sesB.columns, index=pd.MultiIndex.from_tuples([],
                                                                                   names=[config.SUBJECT, config.MOVIE]))

for couple in relevant_couples:
    # get the rows in sesA and sesB which their Multiindex correspond to the tuple of subject and movie
    sesA_couple = sesA[(sesA.index.get_level_values(config.SUBJECT) == couple[0]) & (
                sesA.index.get_level_values(config.MOVIE) == couple[1])]
    sesB_couple = sesB[(sesB.index.get_level_values(config.SUBJECT) == couple[0]) & (
                sesB.index.get_level_values(config.MOVIE) == couple[1])]
    # Remove index levels to match dataframes
    sesA_couple = sesA_couple.droplevel(config.SESSION)
    sesB_couple = sesB_couple.droplevel(config.SESSION)
    # get the max of the couples for each column and max with epsilon to avoid division by zero
    max_couple = np.maximum(sesA_couple, sesB_couple, np.zeros_like(sesA_couple) + np.finfo(float).eps)

    # Percentage change between sessions:
    # (A - B) / max(A, B)
    # Take the max of each subject and movie
    change = ((sesA_couple - sesB_couple) / max_couple)
    normalized_df = normalized_df.append(change)


# Dump the dataframe to pickle
normalized_df.to_pickle(os.path.join(config.classification_resource_dir, "norm_features_df.pkl"))
