import os

import numpy as np
import pandas as pd

from src import config
from src.classification.config import ID_COLUMN

df = pd.read_pickle(os.path.join(config.statistical_analysis_resource_dir, 'features.pkl'))

# get sessions A and B by level values
sesA, sesB = df[df.index.get_level_values(config.SESSION) == config.SESSION_A], \
    df[df.index.get_level_values(config.SESSION) == config.SESSION_B]

sesB_remembered = sesB[sesB['Confidence'] > 0]  # extract subjects who remembered in B
sesB_not_remembered = sesB[sesB['Confidence'] < 0]  # extract subjects who NOT remembered in B

# get matched trials from sesB_remembered
relevant_couples = set(zip(sesB_remembered.index.get_level_values(config.SUBJECT),
                           sesB_remembered.index.get_level_values(config.MOVIE)))

subseries_of_sesA_remembered = {(subj, mov): sesA[((sesA.index.get_level_values(config.MOVIE) == mov)
                                                   & (sesA.index.get_level_values(config.SUBJECT) == subj))]
                                for subj, mov in relevant_couples}
sesA_remembered = pd.concat(subseries_of_sesA_remembered.values())

# get matched trials from sesB_not_remembered
relevant_couples = set(zip(sesB_not_remembered.index.get_level_values(config.SUBJECT),
                           sesB_not_remembered.index.get_level_values(config.MOVIE)))

subseries_of_sesA_not_remembered = {(subj, mov): sesA[((sesA.index.get_level_values(config.MOVIE) == mov)
                                                       & (sesA.index.get_level_values(config.SUBJECT) == subj))]
                                    for subj, mov in relevant_couples}
sesA_not_remembered = pd.concat(subseries_of_sesA_not_remembered.values())

# Remove index levels to match dataframes
sesA_remembered = sesA_remembered.droplevel(config.SESSION)
sesA_not_remembered = sesA_not_remembered.droplevel(config.SESSION)
sesB_remembered = sesB_remembered.droplevel(config.SESSION)
sesB_not_remembered = sesB_not_remembered.droplevel(config.SESSION)

# Percentage change between sessions:
# (B - A) / max(A, B) * 100
# Take the max of each subject and movie
max_remembered = np.maximum(sesA_remembered, sesB_remembered)
max_not_remembered = np.maximum(sesA_not_remembered, sesB_not_remembered)

# Calculate percentage change
remembered_percentage_change = ((sesB_remembered - sesA_remembered) / max_remembered) * 100
not_remembered_percentage_change = ((sesB_not_remembered - sesA_not_remembered) / max_not_remembered) * 100

# Concatenate the two dataframes
sesB = pd.concat([remembered_percentage_change, not_remembered_percentage_change])
mem_labels = np.concatenate(
    [np.ones(len(remembered_percentage_change)), np.zeros(len(not_remembered_percentage_change))])

# Shuffle the mem_labels
# np.random.shuffle(mem_labels)
sesB = sesB.assign(mem=mem_labels)
sesB.set_index('mem', append=True, inplace=True)

relevant_couples = set(zip(sesB.index.get_level_values(config.SUBJECT),
                           sesB.index.get_level_values(config.MOVIE),
                           sesB.index.get_level_values('mem')))

series_id_df = pd.DataFrame(relevant_couples, columns=[config.SUBJECT, config.MOVIE, 'mem'])
series_id_df[ID_COLUMN] = series_id_df.index
series_id_df.to_pickle(os.path.join(config.classification_resource_dir, "norm_mem_series_id_df.pkl"))

# Add Movie ID as a column/feature to the dataframe for the classification task (for each row)
valid_df = sesB.reset_index()
valid_df = valid_df.merge(series_id_df, on=[config.SUBJECT, config.MOVIE, 'mem'])
# Make Movie ID a nominal feature (categorical) using label encoding
valid_df[config.MOVIE] = valid_df[config.MOVIE].astype('category').cat.codes
# Make Movie ID a nominal feature (categorical) using one-hot encoding
# valid_df = pd.get_dummies(valid_df, columns=[config.MOVIE])

TARGET_NAME = [ID_COLUMN, 'mem']

sequences = []
for series_id, group in valid_df.groupby(ID_COLUMN):
    sequence_feature = group.drop([config.SUBJECT, 'mem', 'Confidence'], axis=1)
    label_df = group[TARGET_NAME].drop_duplicates()
    sequences.append((sequence_feature, label_df))

labels_sequences = [t[1] for t in sequences]
labels_df = pd.concat(labels_sequences)
labels_df.set_index(ID_COLUMN, inplace=True)
labels_df.to_pickle(os.path.join(config.classification_resource_dir, "norm_mem_labels_df.pkl"))

features_sequences = [t[0] for t in sequences]
features_df = pd.concat(features_sequences)
features_df.reset_index(drop=True, inplace=True)
features_df.drop(columns=['series_id'], inplace=True)
features_df.to_pickle(os.path.join(config.classification_resource_dir, "norm_mem_features_df.pkl"))
