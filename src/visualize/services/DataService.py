import os
import random

import numpy as np
import pandas as pd

from src import config as global_config
from src.config import AGGRGATED_ROI_FILE, RAW_GAZE_FILE
from src.visualize import config
from src.visualize.models.Point import Point
from src.visualize.models.Rectangle import Rectangle

global rois_path, rois_rects_path, all_fixations, all_gaze, subjects_movie_fixations, subjects_movie_gaze
global videos_dims  # (Width, Height, Duration)


def init():
    print("Retrieving data - START")
    global rois_path, all_fixations, all_gaze, videos_dims, rois_rects_path
    rois_path = os.path.join(global_config.rois_dir, AGGRGATED_ROI_FILE)
    rois_rects_path = os.path.join(global_config.rois_dir, config.rois_rects_file)
    fixations_path = os.path.join(global_config.data_dir, config.fixations_file)
    raw_gaze_path = os.path.join(global_config.data_dir, RAW_GAZE_FILE)
    all_fixations = pd.read_pickle(fixations_path)
    all_gaze = pd.read_pickle(raw_gaze_path)
    videos_dims = pd.read_pickle(global_config.VIDEO_DIMS_FILE_PATH)[:global_config.last_repeating_movie_ind]
    print("Retrieving data - DONE")
    print("\n*******************************\n")


def _extract_coordinates_point(rois_rects: pd.DataFrame, video_indexer: str) -> tuple or ValueError:
    try:
        movie_width, movie_height = videos_dims.loc[video_indexer, ['Width', 'Height']]
        x = (rois_rects.loc[video_indexer][global_config.X_MEDIAN] / 100) * movie_width
        y = (rois_rects.loc[video_indexer][global_config.Y_MEDIAN] / 100) * movie_height

    except ValueError:
        return ValueError("Coordinates are invalid (cannot convert to int)")

    if any(np.isnan([x, y])):
        return ValueError("Coordinates are invalid (Nan)")

    return x, y


def _generate_roi_point(coordinates: tuple) -> Rectangle:
    return Point(coordinates[0], coordinates[1])


def get_video_roi_point(video_indexer: str) -> Rectangle:
    rois = pd.read_pickle(rois_path)

    coordinates = _extract_coordinates_point(rois, video_indexer)
    if type(coordinates) is ValueError:
        raise ValueError("Error in '{}' for video with {}: {}"
                         .format(rois_rects_path, video_indexer, coordinates.args[0]))

    return _generate_roi_point(coordinates)


def _extract_coordinates(rois_rects: pd.DataFrame, video_indexer: str) -> tuple or ValueError:
    try:
        # Rectangle coordinates must be an int - see DrawerService.draw_rect_on_frame #
        tlx = int(rois_rects.loc[video_indexer]['Top Left X'])
        tly = int(rois_rects.loc[video_indexer]['Top Left Y'])
        brx = int(rois_rects.loc[video_indexer]['Bottom Right X'])
        bry = int(rois_rects.loc[video_indexer]['Bottom Right Y'])

    except ValueError:
        return ValueError("Coordinates are invalid (cannot convert to int)")

    if any(np.isnan([tlx, tly, brx, bry])):
        return ValueError("Coordinates are invalid (Nan)")

    return tlx, tly, brx, bry


def _generate_roi(coordinates: tuple) -> Rectangle:
    tl = Point(coordinates[0], coordinates[1])
    br = Point(coordinates[2], coordinates[3])

    return Rectangle(tl=tl, br=br)


def get_video_roi(video_indexer: str) -> Rectangle:
    rois = pd.read_pickle(rois_rects_path)

    coordinates = _extract_coordinates(rois, video_indexer)
    if type(coordinates) is ValueError:
        raise ValueError("Error in '{}' for video with {}: {}"
                         .format(rois_rects_path, video_indexer, coordinates.args[0]))

    return _generate_roi(coordinates)


def generate_subjects(num_subjects: int = None) -> set:
    if num_subjects is None:
        # set default value
        subjects = set(all_fixations.index.get_level_values(global_config.SUBJECT))
        print(f"Subjects included: all {len(subjects)}")
        return subjects

    subjects = set()
    subjects_names = all_fixations.index.get_level_values(global_config.SUBJECT)

    while len(subjects) < num_subjects:
        subjects.add(subjects_names[random.randint(0, len(subjects_names)) - 1])

    print("Subjects included: {}".format(subjects))
    return subjects


def set_data_for_movie_and_subjects(mov: str, subjects: list):
    global subjects_movie_fixations, subjects_movie_gaze

    def _execute(all_data: pd.DataFrame) -> pd.DataFrame:
        movie_data = all_data[all_data.index.get_level_values(2) == mov]
        return movie_data.loc[subjects]

    subjects_movie_fixations = _execute(all_fixations)  # TODO check if needs to drop fixations without full info
    subjects_movie_gaze = _execute(all_gaze).dropna(axis=0)  # drop gazes without full info


def reset_data_for_movie_and_subjects():
    global subjects_movie_fixations, subjects_movie_gaze

    subjects_movie_fixations = None
    subjects_movie_gaze = None


def map_data_to_frame(frame_duration: int) -> None:
    global subjects_movie_fixations, subjects_movie_gaze

    def timestamp_to_frame(duration: float):
        return lambda t: int(t / duration)

    # Fixations
    subjects_movie_fixations['First_Frame'] = subjects_movie_fixations[config.fixation_t0].apply(
        timestamp_to_frame(frame_duration)
    )
    subjects_movie_fixations['Last_Frame'] = (subjects_movie_fixations[config.fixation_t0] +
                                              subjects_movie_fixations[global_config.DURATION]).apply(
        timestamp_to_frame(frame_duration)
    )

    # Gaze
    timestamps = [int(ind[4] / frame_duration) for ind in subjects_movie_gaze.index]
    subjects_movie_gaze['First_Frame'] = timestamps
    subjects_movie_gaze['Last_Frame'] = subjects_movie_gaze['First_Frame']


def get_data_in_frame(frame_ind: int, ttype: str) -> pd.DataFrame:
    global subjects_movie_fixations, subjects_movie_gaze

    ttype = ttype.lower()

    if ttype == "fixations" or ttype == "fixation":
        relevant = subjects_movie_fixations
    elif ttype == "gazes" or ttype == "gaze":
        relevant = subjects_movie_gaze
    else:
        raise ValueError(f"Data from {ttype} type is not support")

    # Filtering fixations that starts after frame
    data_in_frame = relevant[relevant['First_Frame'] <= frame_ind]

    # Filtering fixations that ends before frame
    data_in_frame = data_in_frame[data_in_frame['Last_Frame'] >= frame_ind]

    return data_in_frame
