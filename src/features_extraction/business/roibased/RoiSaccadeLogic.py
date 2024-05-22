import pandas as pd

from src import config
from src.features_extraction.business.roibased.RoiLogic import get_in_out_roi_count_features, \
    get_first_in_roi_features, get_in_roi_stat_features


def get_all_features() -> pd.DataFrame:
    print("Calculating RoI features_extraction - START\n")

    if config.POPULATION != 'yoavdata':
        roi_saccades_features = pd.concat([
            get_in_out_roi_count_features("Saccades_Start"),
            get_in_out_roi_count_features("Saccades_End"),
            get_first_in_roi_features("Saccades_End"),
            get_in_roi_stat_features("Saccades_Start"),
            get_in_roi_stat_features("Saccades_End"),
        ], axis=1)
    else:
        roi_saccades_features = pd.concat([
            get_in_out_roi_count_features("Saccades_Start"),
            get_in_roi_stat_features("Saccades_Start"),
        ], axis=1)

    print("Calculating RoI Saccade features- DONE")
    print("\n*******************************\n")

    return roi_saccades_features
