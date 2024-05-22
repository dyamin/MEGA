import os

import pandas as pd

from src import config


def get_all_valid_subject_data_df():
    df = pd.read_pickle(os.path.join(config.data_dir, config.RAW_GAZE_FILE))
    valid_df = df  # [df.notnull().all(1)]
    return valid_df


def get_aggregated_roi_df():
    return pd.read_pickle(os.path.join(config.rois_dir, config.AGGRGATED_ROI_FILE))
