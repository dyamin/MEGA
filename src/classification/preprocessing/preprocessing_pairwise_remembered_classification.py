import os

import numpy as np
import pandas as pd

from src import config
from src.classification.config import ID_COLUMN

df = pd.read_pickle(os.path.join(config.statistical_analysis_resource_dir, 'features.pkl'))

# get sessions A and B by level values
sesA, sesB = df[df.index.get_level_values(config.SESSION) == config.SESSION_A], \
    df[df.index.get_level_values(config.SESSION) == config.SESSION_B]

sesB_remembered = sesB[sesB['Confidence'] >= 1]  # extract subjects who remembered in B
# get matched trials from sesB_remembered
relevant_couples = set(zip(sesB_remembered.index.get_level_values(config.SUBJECT),
                           sesB_remembered.index.get_level_values(config.MOVIE)))

subseries_of_sesA_remembered = {(subj, mov): sesA[((sesA.index.get_level_values(config.MOVIE) == mov)
                                                   & (sesA.index.get_level_values(config.SUBJECT) == subj))]
                                for subj, mov in relevant_couples}
sesA_remembered = pd.concat(subseries_of_sesA_remembered.values())

# Data Labeling:
# Label the pairs as 0 if the first trial in the pair corresponds to the first condition
# and the second trial to the second condition, and as 1 otherwise.

# get couples of subjects and movies that are in both sessions
relevant_couples = set(
    zip(sesA_remembered.index.get_level_values(config.SUBJECT), sesA_remembered.index.get_level_values(config.MOVIE))).intersection(
    set(zip(sesB_remembered.index.get_level_values(config.SUBJECT), sesB_remembered.index.get_level_values(config.MOVIE))))

normalized_df = pd.DataFrame(columns=sesB_remembered.columns, index=pd.MultiIndex.from_tuples([],
                                                                                   names=[config.SUBJECT, config.MOVIE,
                                                                                          'normalized_by_session_a']))

# For each couple, get the corresponding rows in sesA and sesB_remembered, flip a coin to decide what is the order of percentage change normalization
# and label the rows on normalized_df accordingly
for couple in relevant_couples:
    # get the rows in sesA and sesB_remembered which their Multiindex correspond to the tuple of subject and movie
    sesA_couple = sesA_remembered[(sesA_remembered.index.get_level_values(config.SUBJECT) == couple[0]) & (
                sesA_remembered.index.get_level_values(config.MOVIE) == couple[1])]
    sesB_couple = sesB_remembered[(sesB_remembered.index.get_level_values(config.SUBJECT) == couple[0]) & (
                sesB_remembered.index.get_level_values(config.MOVIE) == couple[1])]
    # Remove index levels to match dataframes
    sesA_couple = sesA_couple.droplevel(config.SESSION)
    sesB_couple = sesB_couple.droplevel(config.SESSION)
    # get the max of the couples for each column and max with epsilon to avoid division by zero
    max_couple = np.maximum(sesA_couple, sesB_couple, np.zeros_like(sesA_couple) + np.finfo(float).eps)

    # flip a coin
    # flip = bool(np.random.randint(0, 2))
    # if flip:
    #     row_to_add = ((sesB_couple - sesA_couple) / max_couple) * 100 # percentage change normalization
    #     # add new index level with label
    #     row_to_add = row_to_add.assign(normalized_by_session_a=1).set_index('normalized_by_session_a', append=True)
    # else:
    #     row_to_add = ((sesA_couple - sesB_couple) / max_couple) * 100 # percentage change normalization
    #     # add label as a new index level
    #     row_to_add = row_to_add.assign(normalized_by_session_a=0).set_index('normalized_by_session_a', append=True)
    # normalized_df = normalized_df.append(row_to_add)

    row_to_add_neg_mega = ((sesB_couple - sesA_couple) / max_couple)  # percentage change normalization
    # add new index level with label
    row_to_add_neg_mega = row_to_add_neg_mega.assign(normalized_by_session_a=0).set_index('normalized_by_session_a',
                                                                                          append=True)
    normalized_df = normalized_df.append(row_to_add_neg_mega)

    row_to_add_pos_mega = ((sesA_couple - sesB_couple) / max_couple)  # percentage change normalization
    # add label as a new index level
    row_to_add_pos_mega = row_to_add_pos_mega.assign(normalized_by_session_a=1).set_index('normalized_by_session_a',
                                                                                          append=True)
    normalized_df = normalized_df.append(row_to_add_pos_mega)

relevant_couples = set(zip(normalized_df.index.get_level_values(config.SUBJECT),
                           normalized_df.index.get_level_values(config.MOVIE),
                           normalized_df.index.get_level_values('normalized_by_session_a')))

series_id_df = pd.DataFrame(relevant_couples, columns=[config.SUBJECT, config.MOVIE, 'normalized_by_session_a'])
series_id_df[ID_COLUMN] = series_id_df.index
series_id_df.to_pickle(os.path.join(config.classification_resource_dir, "pairwise_remembered_sessions_series_id_df.pkl"))

# Add Movie ID as a column/feature to the dataframe for the classification task (for each row)
valid_df = normalized_df.reset_index()
valid_df = valid_df.merge(series_id_df, on=[config.SUBJECT, config.MOVIE, 'normalized_by_session_a'])
# Make Movie ID a nominal feature (categorical) using label encoding
valid_df[config.MOVIE] = valid_df[config.MOVIE].astype('category').cat.codes
# Make Movie ID a nominal feature (categorical) using one-hot encoding
# valid_df = pd.get_dummies(valid_df, columns=[config.MOVIE])


TARGET_NAME = [ID_COLUMN, 'normalized_by_session_a']

sequences = []
for series_id, group in valid_df.groupby(ID_COLUMN):
    sequence_feature = group.drop([config.SUBJECT, 'normalized_by_session_a', 'Confidence'], axis=1)
    label_df = group[TARGET_NAME].drop_duplicates()
    sequences.append((sequence_feature, label_df))

labels_sequences = [t[1] for t in sequences]
labels_df = pd.concat(labels_sequences)
labels_df.set_index(ID_COLUMN, inplace=True)
labels_df.to_pickle(os.path.join(config.classification_resource_dir, "pairwise_remembered_sessions_labels_df.pkl"))

features_sequences = [t[0] for t in sequences]
features_df = pd.concat(features_sequences)
features_df.reset_index(drop=True, inplace=True)
features_df.drop(columns=['series_id'], inplace=True)
features_df.to_pickle(os.path.join(config.classification_resource_dir, "pairwise_remembered_sessions_features_df.pkl"))
