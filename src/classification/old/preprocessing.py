import os

import pandas as pd

from src import config
from src.classification.config import ID_COLUMN
from src.statistical_analysis import utils

df = utils.get_all_valid_subject_data_df()
rois = utils.get_aggregated_roi_df()

roi_drop_movies = set(rois.index) - set(config.valid_movies)
distance_drop_movies = roi_drop_movies.union({
    f'mov{idx}' for idx in range(config.num_repeating_movies + 1, config.total_recorded_movies + 1)})
valid_rois = rois.drop(roi_drop_movies)
valid_df = df.drop(index=distance_drop_movies, level=config.MOVIE)

# df = cut_tail(valid_df, valid_rois)

relevant_couples = set(zip(valid_df.index.get_level_values(config.SUBJECT),
                           valid_df.index.get_level_values(config.MOVIE),
                           valid_df.index.get_level_values(config.SESSION)))

series_id_df = pd.DataFrame(relevant_couples, columns=[config.SUBJECT, config.MOVIE, config.SESSION])
series_id_df[ID_COLUMN] = series_id_df.index
series_id_df.to_pickle(os.path.join(config.classification_resource_dir, "series_id_df.pkl"))

valid_df = valid_df.reset_index()
valid_df = valid_df.merge(series_id_df, on=[config.SUBJECT, config.MOVIE, config.SESSION])
valid_df = valid_df[
    [ID_COLUMN, config.TIMESTAMP, config.gaze_X, config.gaze_Y, config.PUPIL, config.DVA, config.SESSION]]
valid_df[config.SESSION] = (valid_df[config.SESSION] == config.SESSION_B).astype(int)

FEATURE_NAMES = [ID_COLUMN, config.TIMESTAMP, config.DVA, config.gaze_X, config.gaze_Y, config.PUPIL]
TARGET_NAME = [ID_COLUMN, config.SESSION]

sequences = []
for series_id, group in valid_df.groupby(ID_COLUMN):
    sequence_feature = group[FEATURE_NAMES]
    label_df = group[TARGET_NAME].drop_duplicates()
    sequences.append((sequence_feature, label_df))

labels_sequences = [t[1] for t in sequences]
labels_df = pd.concat(labels_sequences)
labels_df.set_index(ID_COLUMN, inplace=True)
labels_df.to_pickle(os.path.join(config.classification_resource_dir, "labels_df.pkl"))

features_sequences = [t[0] for t in sequences]
features_df = pd.concat(features_sequences)
features_df.reset_index(drop=True, inplace=True)
features_df.to_pickle(os.path.join(config.classification_resource_dir, "features_df.pkl"))
