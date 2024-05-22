import os

import pandas as pd

from src import config as globalconfig
from src.roi import config

global videos_dims  # (Width, Height, Duration)
global aggregated_rois
global log_file


def init(with_rois: bool = False):
    print("Retrieving data - START")

    global videos_dims, aggregated_rois, log_file

    videos_dims = pd.read_pickle(globalconfig.VIDEO_DIMS_FILE_PATH)[:globalconfig.num_repeating_movies]

    if with_rois:
        aggregated_rois_path = os.path.join(globalconfig.data_dir, globalconfig.AGGRGATED_ROI_FILE)
        aggregated_rois = pd.read_pickle(aggregated_rois_path)

    print("Retrieving data - DONE")


def write_pickle(data: pd.DataFrame, file_name: str) -> None:
    print(f"Writing data to file {file_name} in data directory - START")
    path = os.path.join(globalconfig.rois_dir, file_name)
    pd.to_pickle(data, path)
    print(f"Writing data to file {file_name} in data directory - DONE")


def log(msg: str) -> None:
    global log_file
    log_file.write(msg)


def open_log(metric: str) -> None:
    global log_file

    # Creating the log file
    log_name = f"RoI by {metric}.txt"
    if not os.path.exists(config.log_dir):
        print(f"Creates new path {config.log_dir}")
        os.makedirs(config.log_dir)
    log_path = os.path.join(config.log_dir, log_name)
    log_file = open(log_path, 'w')


def close_log() -> None:
    global log_file
    log_file.close()
