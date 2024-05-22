import pandas as pd

from src.features_extraction.business.roibased.RoiLogic import get_in_out_roi_count_features, \
    get_re_entries_count_features, get_pupil_change_feature


def get_all_features() -> pd.DataFrame:
    print("Calculating RoI features_extraction - START\n")

    #  combine all features_extraction and fill missing values with 0
    roi_gaze_features = pd.concat([
        get_in_out_roi_count_features("Gaze"),
        get_re_entries_count_features("Gaze"),
        get_pupil_change_feature("Gaze")
    ], axis=1)

    print("Calculating RoI Gaze features- DONE")
    print("\n*******************************\n")

    return roi_gaze_features
