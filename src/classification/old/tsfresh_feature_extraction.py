import os
from multiprocessing import freeze_support

import pandas as pd
from tsfresh import extract_features

from src import config
from src.classification.config import ID_COLUMN

features_df = pd.read_pickle(os.path.join(config.classification_resource_dir, "features_df.pkl"))

if __name__ == '__main__':
    freeze_support()
    extracted_features = extract_features(features_df, column_id=ID_COLUMN, column_sort=config.TIMESTAMP,
                                          column_kind=None, column_value=None)

    print(extracted_features)
    extracted_features.to_pickle(os.path.join(config.classification_resource_dir, "tsfresh_extracted_features.pkl"))
