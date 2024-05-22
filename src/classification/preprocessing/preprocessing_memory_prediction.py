import os

import numpy as np
import pandas as pd

from src import config
from src.classification.config import ID_COLUMN

df_elderly = pd.read_pickle(os.path.join(config.elderly_statistical_analysis_resource_dir, 'features.pkl'))
df_mci_ad = pd.read_pickle(os.path.join(config.mci_ad_statistical_analysis_resource_dir, 'features.pkl'))

# get sessions A and B by level values
sesA_elderly, sesB_elderly = df_elderly[df_elderly.index.get_level_values(config.SESSION) == config.SESSION_A], \
    df_elderly[df_elderly.index.get_level_values(config.SESSION) == config.SESSION_B]
sesA_mci_ad, sesB_mci_ad = df_mci_ad[df_mci_ad.index.get_level_values(config.SESSION) == config.SESSION_A], \
    df_mci_ad[df_mci_ad.index.get_level_values(config.SESSION) == config.SESSION_B]

# Remove index levels to match dataframes
sesA_elderly = sesA_elderly.droplevel(config.SESSION)
sesB_elderly = sesB_elderly.droplevel(config.SESSION)
sesA_mci_ad = sesA_mci_ad.droplevel(config.SESSION)
sesB_mci_ad = sesB_mci_ad.droplevel(config.SESSION)

# take the intersection of indexes in both sessions
sesA_elderly = sesA_elderly[sesA_elderly.index.isin(sesB_elderly.index)]
sesB_elderly = sesB_elderly[sesB_elderly.index.isin(sesA_elderly.index)]
sesA_mci_ad = sesA_mci_ad[sesA_mci_ad.index.isin(sesB_mci_ad.index)]
sesB_mci_ad = sesB_mci_ad[sesB_mci_ad.index.isin(sesA_mci_ad.index)]

# Percentage change between sessions:
# (B - A) / max(A, B) * 100
# Take the max of each subject and movie
max_elderly = np.maximum(sesA_elderly, sesB_elderly)
max_mci_ad = np.maximum(sesA_mci_ad, sesB_mci_ad)

elderly_percentage_change = ((sesB_elderly - sesA_elderly) / max_elderly) * 100
mci_ad_percentage_change = ((sesB_mci_ad - sesA_mci_ad) / max_mci_ad) * 100

perchange_df = pd.concat([elderly_percentage_change, mci_ad_percentage_change])

# Average the percentage change for each subject, ignore nan values
perchange_df = perchange_df.groupby(config.SUBJECT).mean()

# Load the memory performance file and add it as a feature to the dataframe
memory_performance = pd.read_pickle(os.path.join(config.elderly_statistical_analysis_resource_dir,
                                                 'memory_performance.pkl'))
memory_performance.set_index('Name', inplace=True)
# Filter from both df subjects that are in both df
perchange_df = perchange_df[perchange_df.index.isin(memory_performance.index)]
memory_performance = memory_performance[memory_performance.index.isin(perchange_df.index)]

perchange_df = perchange_df.assign(memory_performance=memory_performance['MMSE'])
perchange_df.set_index('memory_performance', append=True, inplace=True)

relevant_couples = set(zip(perchange_df.index.get_level_values(config.SUBJECT),
                           perchange_df.index.get_level_values('memory_performance')))

series_id_df = pd.DataFrame(relevant_couples, columns=[config.SUBJECT, 'memory_performance'])
series_id_df[ID_COLUMN] = series_id_df.index
series_id_df.to_pickle(os.path.join(config.classification_resource_dir, "memory_performance_series_id_df.pkl"))

# Add Movie ID as a column/feature to the dataframe for the classification task (for each row)
valid_df = perchange_df.reset_index()
valid_df = valid_df.merge(series_id_df, on=[config.SUBJECT, 'memory_performance'])

TARGET_NAME = [ID_COLUMN, 'memory_performance']

sequences = []
for series_id, group in valid_df.groupby(ID_COLUMN):
    sequence_feature = group.drop([config.SUBJECT, 'memory_performance', 'Confidence'], axis=1)
    label_df = group[TARGET_NAME].drop_duplicates()
    sequences.append((sequence_feature, label_df))

labels_sequences = [t[1] for t in sequences]
labels_df = pd.concat(labels_sequences)
labels_df.set_index(ID_COLUMN, inplace=True)
labels_df.to_pickle(os.path.join(config.classification_resource_dir, "memory_performance_labels_df.pkl"))

features_sequences = [t[0] for t in sequences]
features_df = pd.concat(features_sequences)
features_df.reset_index(drop=True, inplace=True)
features_df.drop(columns=['series_id'], inplace=True)
features_df.to_pickle(os.path.join(config.classification_resource_dir, "memory_performance_features_df.pkl"))
