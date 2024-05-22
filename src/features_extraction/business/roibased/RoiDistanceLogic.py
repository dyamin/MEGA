import pandas as pd

from src.features_extraction.business.roibased.RoiLogic import get_dva_features


def get_all_features() -> pd.DataFrame:
    print("Calculating RoI Distance features - START\n")

    roi_dva_features = pd.concat([
        get_dva_features("Gaze"),
        get_dva_features("Fixations")
    ], axis=1)

    print("Calculating RoI Distance features- DONE")
    print("\n*******************************\n")

    return roi_dva_features
