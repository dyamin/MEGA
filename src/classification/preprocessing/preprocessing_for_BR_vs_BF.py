import os

import pandas as pd

from src import config
from src.classification.config import ID_COLUMN

df = pd.read_pickle(os.path.join(config.statistical_analysis_resource_dir, 'features.pkl'))

# get sessions A and B by level values
sesA, sesB = df[df.index.get_level_values(config.SESSION) == config.SESSION_A], \
    df[df.index.get_level_values(config.SESSION) == config.SESSION_B]

sesB_remembered = sesB[sesB['Confidence'] >= 1]  # extract subjects who remembered in B
sesB_remembered = sesB_remembered.assign(mem=1).set_index('mem', append=True)
sesB_not_remembered = sesB[sesB['Confidence'] <= -1]  # extract subjects who NOT remembered in B
sesB_not_remembered = sesB_not_remembered.assign(mem=0).set_index('mem', append=True)

sesB = pd.concat([sesB_not_remembered, sesB_remembered])

relevant_couples = set(zip(sesB.index.get_level_values(config.SUBJECT),
                           sesB.index.get_level_values(config.MOVIE),
                           sesB.index.get_level_values('mem')))

series_id_df = pd.DataFrame(relevant_couples, columns=[config.SUBJECT, config.MOVIE, 'mem'])
series_id_df[ID_COLUMN] = series_id_df.index
series_id_df.to_pickle(os.path.join(config.classification_resource_dir, "memory_series_id_df.pkl"))

valid_df = sesB.reset_index()
valid_df = valid_df.merge(series_id_df, on=[config.SUBJECT, config.MOVIE, 'mem'])

TARGET_NAME = [ID_COLUMN, 'mem']

sequences = []
for series_id, group in valid_df.groupby(ID_COLUMN):
    sequence_feature = group.drop([config.SUBJECT, config.MOVIE, 'mem', 'Confidence'], axis=1)
    label_df = group[TARGET_NAME].drop_duplicates()
    sequences.append((sequence_feature, label_df))

labels_sequences = [t[1] for t in sequences]
labels_df = pd.concat(labels_sequences)
labels_df.set_index(ID_COLUMN, inplace=True)
labels_df.to_pickle(os.path.join(config.classification_resource_dir, "memory_labels_df.pkl"))

features_sequences = [t[0] for t in sequences]
features_df = pd.concat(features_sequences)
features_df.reset_index(drop=True, inplace=True)
features_df.drop(columns=['Session', 'series_id'], inplace=True)
features_df.to_pickle(os.path.join(config.classification_resource_dir, "memory_features_df.pkl"))
