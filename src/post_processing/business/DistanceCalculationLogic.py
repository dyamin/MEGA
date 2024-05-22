import numpy as np
import pandas as pd

from src import config as global_config
from src.post_processing import config
from src.signal_processing.unit_transformers import get_deg_per_pxl


def calculate_distances(eyetracking_data: pd.DataFrame, roi_data: pd.DataFrame, video_dims: pd.DataFrame,
                        is_fixations=False):
    relevant_data = _drop_unnecessary_movies(eyetracking_data, roi_data)
    eyetracking_X_colname, eyetracking_Y_colname = _find_x_and_y_colnames(relevant_data)
    eyetracking_x_y = relevant_data[[eyetracking_X_colname, eyetracking_Y_colname]].rename(
        columns=lambda col: 'X' if 'X' in col else 'Y')

    # calculate the distances
    distances_series = eyetracking_x_y.copy()

    xy_to_sub = __get_roi_middle_point(roi_data, video_dims)
    distances_series = distances_series.subtract(xy_to_sub, level='Movie')

    dvas_series = distances_series.copy()
    distances_series = distances_series.pow(2).sum(axis=1).transform(np.sqrt)

    movie_dataframes = dict()
    for mov in dvas_series.index.unique(level='Movie'):
        movie_width, movie_height = video_dims.loc[mov, ['Width', 'Height']]
        movie_distance = dvas_series.xs(mov, level=global_config.MOVIE)
        movie_distance['X'] *= get_deg_per_pxl(movie_width, global_config.HORIZONTAL_SIZE_IN_CM)
        movie_distance['Y'] *= get_deg_per_pxl(movie_height, global_config.VERTICAL_SIZE_IN_CM)
        movie_distance = movie_distance.pow(2).sum(axis=1).transform(np.sqrt)
        movie_dataframes[mov] = movie_distance

    dvas_series = pd.concat(movie_dataframes.values(), keys=movie_dataframes.keys(),
                            names=[global_config.MOVIE, global_config.SUBJECT, global_config.SESSION,
                                   global_config.TIMESTAMP])
    dvas_series = dvas_series.reorder_levels(
        [global_config.SUBJECT, global_config.SESSION, global_config.MOVIE, global_config.TIMESTAMP])

    # rename the indexers
    distances_series.rename(global_config.DISTANCE, inplace=True)
    indexers = [global_config.SUBJECT, global_config.SESSION, global_config.MOVIE,
                config.FIXATIONNUMBER if is_fixations else global_config.TIMESTAMP]
    distances_series.index.names = indexers
    dvas_series.rename(global_config.DVA, inplace=True)
    dvas_series.index.names = indexers

    # Create another series with the sqrt transformation of the dva distance
    dvas_series_sqrt = dvas_series.copy()
    dvas_series_sqrt = dvas_series_sqrt.transform(np.sqrt)
    dvas_series_sqrt.rename(global_config.SQRT_DVA, inplace=True)
    dvas_series_sqrt.index.names = indexers

    return distances_series, dvas_series, dvas_series_sqrt


def _validate_same_name_for_movies_level(eyetracking_data: pd.DataFrame, roi_data: pd.DataFrame) -> str:
    eyetracking_data_movies_level_name = __find_string_with_substring(list(eyetracking_data.index.names),
                                                                      global_config.MOVIE)
    rois_movies_level_name = __find_string_with_substring(list(roi_data.index.names), global_config.MOVIE)
    assert (eyetracking_data_movies_level_name == rois_movies_level_name
            ), f'Movies label for eye_tracker ({eyetracking_data_movies_level_name}) and RoIs ({rois_movies_level_name}) do not match.'
    return eyetracking_data_movies_level_name


def _drop_unnecessary_movies(eye_tracking_data: pd.DataFrame, roi_data: pd.DataFrame):
    all_movies = set(eye_tracking_data.index.get_level_values(global_config.MOVIE))
    movies_with_roi = set(roi_data.index)
    movies_to_drop = all_movies - movies_with_roi  # finds movies with no RoI
    return eye_tracking_data.drop(labels=movies_to_drop, level=global_config.MOVIE, axis=0)


def _find_x_and_y_colnames(dataframe):
    x_colname = __find_string_with_substring(list(dataframe.columns), 'X')
    y_colname = __find_string_with_substring(list(dataframe.columns), 'Y')
    return x_colname, y_colname


def __find_string_with_substring(strings: list, substring: str) -> str:
    ''' Returns the first column among dataframe.columns that contains @substring as a sub-string '''
    matching_colnames = [col for col in strings if substring in col]

    assert (len(matching_colnames) > 0), f'Couldn\'t find a column containing {substring} in the given list {strings}.'
    return matching_colnames[0]


def __get_roi_middle_point(roi_data: pd.DataFrame, video_dims: pd.DataFrame):
    df = pd.DataFrame()
    df['X'] = (roi_data[global_config.X_MEDIAN] / 100) * video_dims['Width']
    df['Y'] = (roi_data[global_config.Y_MEDIAN] / 100) * video_dims['Height']
    return df
