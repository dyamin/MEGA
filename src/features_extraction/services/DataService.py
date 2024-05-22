import os

import pandas as pd

from src import config
from src.features_extraction import config as fa_config

global fixations, saccades, blinks, gaze, videos_dims, rects
global rois, memory_self_report


def init():
    global fixations, saccades, blinks, gaze, videos_dims, rects, memory_self_report, rois

    print("Retrieving data - START")
    videos_dims = pd.read_pickle(config.VIDEO_DIMS_FILE_PATH)

    fixations_path = os.path.join(config.data_dir, config.all_subject + config.fixations + '.pkl')
    fixations = pd.read_pickle(fixations_path)
    fixations = fixations.loc[fixations.index.get_level_values(config.MOVIE).isin(config.valid_movies)]

    blinks_path = os.path.join(config.data_dir, config.all_subject + config.blinks + '.pkl')
    blinks = pd.read_pickle(blinks_path)
    blinks = blinks.loc[blinks.index.get_level_values(config.MOVIE).isin(config.valid_movies)]

    saccades_path = os.path.join(config.data_dir, config.all_subject + config.saccades + '.pkl')
    saccades = pd.read_pickle(saccades_path)
    saccades = saccades.loc[saccades.index.get_level_values(config.MOVIE).isin(config.valid_movies)]

    gaze_path = os.path.join(config.data_dir, config.RAW_GAZE_FILE)
    gaze = pd.read_pickle(gaze_path)
    gaze = gaze.loc[gaze.index.get_level_values(config.MOVIE).isin(config.valid_movies)]

    rects_path = os.path.join(config.rois_dir, config.rois_rects_file)
    rects = pd.read_pickle(rects_path)

    if config.POPULATION != 'yoavdata':
        memory_self_report_path = os.path.join(config.data_dir, fa_config.self_report_filename)
        memory_self_report = pd.read_pickle(memory_self_report_path)
        memory_self_report = memory_self_report.loc[
            memory_self_report.index.get_level_values(config.MOVIE).isin(config.valid_movies)]

    rois_path = os.path.join(config.rois_dir, config.AGGRGATED_ROI_FILE)
    rois = pd.read_pickle(rois_path)

    print("Retrieving data - DONE")
    print("\n*******************************\n")


def filter_data_to_pre_roi(data, rois, onset=config.ONSET):
    # Filter out data after the roi timing of each movie
    pre_data_list = []
    post_data_list = []
    for mov in data.index.get_level_values(config.MOVIE).unique():
        # Filter data for the current movie
        current_movie = data[data.index.get_level_values(config.MOVIE) == mov]

        # Generate a boolean mask for data to keep (before the ROI median time)
        mask = current_movie[onset] < rois.loc[mov, config.T_MEDIAN]

        # Use the mask to filter the data for the current movie
        pre = current_movie[mask]
        post = current_movie[~mask]

        # Append the filtered data to the list
        pre_data_list.append(pre)
        post_data_list.append(post)
    # Concatenate all filtered data into a new DataFrame
    pre_data = pd.concat(pre_data_list, axis=0)
    post_data = pd.concat(post_data_list, axis=0)
    # Reassign the filtered data back to data
    return pre_data, post_data


def get_data(ttype: str) -> pd.DataFrame:
    global fixations, saccades, blinks, gaze

    if ttype == "Fixations":
        return fixations
    elif ttype.startswith("Saccades"):
        return saccades
    elif ttype == "Blinks":
        return blinks
    elif ttype == "Gaze":
        return gaze
    else:
        raise ValueError(f"Data from type {ttype} is not supported")


def write_pickle(data: pd.DataFrame, file_name) -> None:
    print(f"Writing data to file {file_name} in data directory - START")
    data_dir = config.statistical_analysis_resource_dir
    path = os.path.join(data_dir, file_name)
    pd.to_pickle(data, path)
    print(f"Writing data to file {file_name} in data directory - DONE")
    print("\n*******************************\n")
