import pandas as pd

from src.features_extraction.business.roibased.RoiLogic import get_in_out_roi_count_features, \
    get_re_entries_count_features, \
    get_first_in_roi_features, get_in_roi_stat_features, get_re_entries_pupil_features, \
    get_pupil_change_feature


def get_all_features() -> pd.DataFrame:
    print("Calculating RoI features_extraction - START\n")

    #  combine all features_extraction and fill missing values with 0
    roi_fixation_features = pd.concat([
        get_in_out_roi_count_features("Fixations"),
        get_re_entries_count_features("Fixations"),
        get_first_in_roi_features("Fixations"),
        get_in_roi_stat_features("Fixations"),
        get_re_entries_pupil_features("Fixations"),
        get_pupil_change_feature("Fixations")
    ], axis=1)

    print("Calculating RoI Fixation features- DONE")
    print("\n*******************************\n")

    return roi_fixation_features
