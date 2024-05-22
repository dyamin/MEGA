import os

import numpy as np
import pandas as pd

from src import config
from src.classification.config import ID_COLUMN

df_no_nap = pd.read_pickle(os.path.join(config.no_nap_statistical_analysis_resource_dir, 'features.pkl'))
df_nap = pd.read_pickle(os.path.join(config.nap_statistical_analysis_resource_dir, 'features.pkl'))

# get sessions A and B by level values
sesA_no_nap, sesB_no_nap = df_no_nap[df_no_nap.index.get_level_values(config.SESSION) == config.SESSION_A], \
    df_no_nap[df_no_nap.index.get_level_values(config.SESSION) == config.SESSION_B]
sesA_nap, sesB_nap = df_nap[df_nap.index.get_level_values(config.SESSION) == config.SESSION_A], \
    df_nap[df_nap.index.get_level_values(config.SESSION) == config.SESSION_B]

# Remove index levels to match dataframes
sesA_no_nap = sesA_no_nap.droplevel(config.SESSION)
sesB_no_nap = sesB_no_nap.droplevel(config.SESSION)
sesA_nap = sesA_nap.droplevel(config.SESSION)
sesB_nap = sesB_nap.droplevel(config.SESSION)

# take the intersection of indexes in both sessions
sesA_no_nap = sesA_no_nap[sesA_no_nap.index.isin(sesB_no_nap.index)]
sesB_no_nap = sesB_no_nap[sesB_no_nap.index.isin(sesA_no_nap.index)]
sesA_nap = sesA_nap[sesA_nap.index.isin(sesB_nap.index)]
sesB_nap = sesB_nap[sesB_nap.index.isin(sesA_nap.index)]

# Percentage change between sessions:
# (B - A) / max(A, B) * 100
# Take the max of each subject and movie
max_no_nap = np.maximum(sesA_no_nap, sesB_no_nap)
max_nap = np.maximum(sesA_nap, sesB_nap)

no_nap_percentage_change = ((sesB_no_nap - sesA_no_nap) / max_no_nap) * 100
nap_percentage_change = ((sesB_nap - sesA_nap) / max_nap) * 100

perchange_df = pd.concat([no_nap_percentage_change, nap_percentage_change])
pop_labels = np.concatenate(
    [np.ones(len(no_nap_percentage_change)), np.zeros(len(nap_percentage_change))])

# Shuffle the mem_labels
# np.random.shuffle(mem_labels)
perchange_df = perchange_df.assign(pop=pop_labels)
perchange_df.set_index('pop', append=True, inplace=True)

relevant_couples = set(zip(perchange_df.index.get_level_values(config.SUBJECT),
                           perchange_df.index.get_level_values(config.MOVIE),
                           perchange_df.index.get_level_values('pop')))

series_id_df = pd.DataFrame(relevant_couples, columns=[config.SUBJECT, config.MOVIE, 'pop'])
series_id_df[ID_COLUMN] = series_id_df.index
series_id_df.to_pickle(os.path.join(config.classification_resource_dir, "norm_young_pop_series_id_df.pkl"))

# Add Movie ID as a column/feature to the dataframe for the classification task (for each row)
valid_df = perchange_df.reset_index()
valid_df = valid_df.merge(series_id_df, on=[config.SUBJECT, config.MOVIE, 'pop'])
# Make Movie ID a nominal feature (categorical) using label encoding
valid_df[config.MOVIE] = valid_df[config.MOVIE].astype('category').cat.codes
# Make Movie ID a nominal feature (categorical) using one-hot encoding
# valid_df = pd.get_dummies(valid_df, columns=[config.MOVIE])
TARGET_NAME = [ID_COLUMN, 'pop']

sequences = []
for series_id, group in valid_df.groupby(ID_COLUMN):
    sequence_feature = group.drop([config.SUBJECT, 'pop', 'Confidence'], axis=1)
    label_df = group[TARGET_NAME].drop_duplicates()
    sequences.append((sequence_feature, label_df))

labels_sequences = [t[1] for t in sequences]
labels_df = pd.concat(labels_sequences)
labels_df.set_index(ID_COLUMN, inplace=True)
labels_df.to_pickle(os.path.join(config.classification_resource_dir, "norm_young_pop_labels_df.pkl"))

features_sequences = [t[0] for t in sequences]
features_df = pd.concat(features_sequences)
features_df.reset_index(drop=True, inplace=True)
features_df.drop(columns=['series_id'], inplace=True)
features_df.to_pickle(os.path.join(config.classification_resource_dir, "norm_young_pop_features_df.pkl"))
