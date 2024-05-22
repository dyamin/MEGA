import os

import numpy as np
import pandas as pd

from src import config
from src.classification.config import ID_COLUMN

features_df = pd.read_pickle(os.path.join(config.statistical_analysis_resource_dir, 'features.pkl'))
# load csv to pandas dataframe
eventmemory_df = pd.read_csv(os.path.join(config.rois_dir, 'DVA_scenerymemory_yoavversion.csv'))

# Move columns to index
eventmemory_df.set_index([config.SUBJECT, config.MOVIE], inplace=True)

# get sessions A and B by level values
sesA, labeled_df = features_df[features_df.index.get_level_values(config.SESSION) == config.SESSION_A], \
    features_df[features_df.index.get_level_values(config.SESSION) == config.SESSION_B]

# get remembered and not remembered trials using the eventmemory dataframe
no_memory_df = eventmemory_df[eventmemory_df['memorysource'] == 'no_memory']
scenery_memory_df = eventmemory_df[eventmemory_df['memorysource'] == 'scenery_memory']

# ge t matched trials for no memory from sesA and sesB
no_memory_relevant_couples = set(zip(no_memory_df.index.get_level_values(config.SUBJECT),
                                     no_memory_df.index.get_level_values(config.MOVIE)))
subseries_of_sesA_no_memory = {
    (subj, 'mov' + str(mov)): sesA[((sesA.index.get_level_values(config.MOVIE) == 'mov' + str(mov))
                                    & (sesA.index.get_level_values(config.SUBJECT) == subj))]
    for subj, mov in no_memory_relevant_couples}
sesA_no_memory = pd.concat(subseries_of_sesA_no_memory.values())

subseries_of_sesB_no_memory = {
    (subj, 'mov' + str(mov)): labeled_df[((labeled_df.index.get_level_values(config.MOVIE) == 'mov' + str(mov))
                                          & (labeled_df.index.get_level_values(config.SUBJECT) == subj))]
    for subj, mov in no_memory_relevant_couples}
sesB_no_memory = pd.concat(subseries_of_sesB_no_memory.values())

# get matched trials for scenery memory from sesA and sesB
scenery_memory_relevant_couples = set(zip(scenery_memory_df.index.get_level_values(config.SUBJECT),
                                          scenery_memory_df.index.get_level_values(config.MOVIE)))
subseries_of_sesA_scenery_memory = {
    (subj, 'mov' + str(mov)): sesA[((sesA.index.get_level_values(config.MOVIE) == 'mov' + str(mov))
                                    & (sesA.index.get_level_values(config.SUBJECT) == subj))]
    for subj, mov in scenery_memory_relevant_couples}
sesA_scenery_memory = pd.concat(subseries_of_sesA_scenery_memory.values())

subseries_of_sesB_scenery_memory = {
    (subj, 'mov' + str(mov)): labeled_df[((labeled_df.index.get_level_values(config.MOVIE) == 'mov' + str(mov))
                                          & (labeled_df.index.get_level_values(config.SUBJECT) == subj))]
    for subj, mov in scenery_memory_relevant_couples}
sesB_scenery_memory = pd.concat(subseries_of_sesB_scenery_memory.values())

# Remove index levels to match dataframes
sesA_scenery_memory = sesA_scenery_memory.droplevel(config.SESSION)
sesB_scenery_memory = sesB_scenery_memory.droplevel(config.SESSION)
sesA_no_memory = sesA_no_memory.droplevel(config.SESSION)
sesB_no_memory = sesB_no_memory.droplevel(config.SESSION)

# Percentage change between sessions:
# (B - A) / max(A, B) * 100
# Take the max of each subject and movie
max_remembered = np.maximum(sesA_scenery_memory, sesB_scenery_memory)
max_not_remembered = np.maximum(sesA_no_memory, sesB_no_memory)

remembered_percentage_change = ((sesB_scenery_memory - sesA_scenery_memory) / max_remembered) * 100
not_remembered_percentage_change = ((sesB_no_memory - sesA_no_memory) / max_not_remembered) * 100

labeled_df = pd.concat([remembered_percentage_change, not_remembered_percentage_change])
mem_labels = np.concatenate(
    [np.ones(len(remembered_percentage_change)), np.zeros(len(not_remembered_percentage_change))])

# Shuffle the mem_labels
# np.random.shuffle(mem_labels)
labeled_df = labeled_df.assign(familiarity=mem_labels)
labeled_df.set_index('familiarity', append=True, inplace=True)

# Dump the dataframe to pickle
labeled_df.to_pickle(os.path.join(config.classification_resource_dir, "norm_labeled_df.pkl"))

relevant_couples = set(zip(labeled_df.index.get_level_values(config.SUBJECT),
                           labeled_df.index.get_level_values(config.MOVIE),
                           labeled_df.index.get_level_values('familiarity')))

series_id_df = pd.DataFrame(relevant_couples, columns=[config.SUBJECT, config.MOVIE, 'familiarity'])
series_id_df[ID_COLUMN] = series_id_df.index
series_id_df.to_pickle(os.path.join(config.classification_resource_dir, "norm_familiarity_series_id_df.pkl"))

# Add Movie ID as a column/feature to the dataframe for the classification task (for each row)
valid_df = labeled_df.reset_index()
valid_df = valid_df.merge(series_id_df, on=[config.SUBJECT, config.MOVIE, 'familiarity'])
# Make Movie ID a nominal feature (categorical) using label encoding
valid_df[config.MOVIE] = valid_df[config.MOVIE].astype('category').cat.codes
# Make Movie ID a nominal feature (categorical) using one-hot encoding
# valid_df = pd.get_dummies(valid_df, columns=[config.MOVIE])

TARGET_NAME = [ID_COLUMN, 'familiarity']

sequences = []
for series_id, group in valid_df.groupby(ID_COLUMN):
    sequence_feature = group.drop([config.SUBJECT, 'familiarity'], axis=1)
    label_df = group[TARGET_NAME].drop_duplicates()
    sequences.append((sequence_feature, label_df))

labels_sequences = [t[1] for t in sequences]
labels_df = pd.concat(labels_sequences)
labels_df.set_index(ID_COLUMN, inplace=True)
labels_df.to_pickle(os.path.join(config.classification_resource_dir, "norm_familiarity_labels_df.pkl"))

features_sequences = [t[0] for t in sequences]
features_df = pd.concat(features_sequences)
features_df.reset_index(drop=True, inplace=True)
features_df.drop(columns=['series_id'], inplace=True)
features_df.to_pickle(os.path.join(config.classification_resource_dir, "norm_familiarity_features_df.pkl"))