from typing import List

import pandas as pd

from src import config as main_config
from src.features_extraction import config
from src.features_extraction.business import BlinksLogic, FixationsLogic, SaccadesLogic
from src.features_extraction.business import DistanceLogic
from src.features_extraction.business.roibased import RoiGazeLogic, RoiFixationLogic, RoiSaccadeLogic, \
    RoiDistanceLogic
from src.features_extraction.services import DataService


def calculate(args: List[str], write: bool = True):
    action = _extract_args(args)

    print("Calculating features_extraction - START")
    print("\n*******************************\n")

    if action is None or action == "all":
        features = _get_all_features()
    elif action == "blinks":
        features = BlinksLogic.get_all_features()
    elif action == "fixations":
        features = FixationsLogic.get_all_features()
    elif action == "saccades":
        features = SaccadesLogic.get_all_features()
    elif action == "fsr":  # Fixation/Saccades ratio
        features = _calc_fsr()
    elif action == "distance":
        features = DistanceLogic.get_all_features()
    elif action == "rois":
        features = _get_all_roi_features()
    else:
        raise ValueError(f"Feature {action} is not supported")

    print("\n*******************************\n")
    print("Calculating features_extraction - DONE")
    print("\n*******************************\n")

    if write:
        DataService.write_pickle(features, config.results_filename)
    return features


def _extract_args(args: List[str]) -> str:
    try:
        action = args[0].lower()
    except IndexError:
        action = None

    return action


def _get_all_features() -> pd.DataFrame:
    if main_config.POPULATION == 'yoavdata':
        return pd.concat([BlinksLogic.get_all_features(),
                          FixationsLogic.get_all_features(),
                          SaccadesLogic.get_all_features(),
                          _calc_fsr(),
                          _get_all_roi_features(),
                          # DistanceLogic.get_all_features(),
                          # DataService.memory_self_report
                          ], axis=1)
    else:
        return pd.concat([BlinksLogic.get_all_features(),
                          FixationsLogic.get_all_features(),
                          SaccadesLogic.get_all_features(),
                          _calc_fsr(),
                          _get_all_roi_features(),
                          DistanceLogic.get_all_features(),
                          ], axis=1)


def _get_all_roi_features() -> pd.DataFrame:
    if main_config.POPULATION == 'yoavdata':
        return pd.concat([
            RoiGazeLogic.get_all_features(),
            RoiFixationLogic.get_all_features(),
            RoiSaccadeLogic.get_all_features(),
            RoiDistanceLogic.get_all_features(),
        ], axis=1)
    else:
        return pd.concat([
            RoiGazeLogic.get_all_features(),
            RoiFixationLogic.get_all_features(),
            RoiSaccadeLogic.get_all_features(),
            RoiDistanceLogic.get_all_features(),
            DataService.memory_self_report
        ], axis=1)


def _calc_fsr() -> pd.DataFrame:
    print("Calculating fsr - START")

    fixations_counts = FixationsLogic.get_counts()
    saccades_counts = SaccadesLogic.get_counts()

    fsr = fixations_counts.divide(saccades_counts).rename("Fixations_Saccades_Ratio")

    print("Calculating fsr - DONE")
    print("\n*******************************\n")

    return pd.DataFrame(fsr)
