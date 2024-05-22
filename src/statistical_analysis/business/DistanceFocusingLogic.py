import pandas as pd

from src import config as g_config
from src.post_processing.business.GazePrefixRemovalLogic import remove_prefix_from_raw_data
from src.utils import round_down_to_closest_even, convert_dict_to_series


def recenter_distances(distances, rois, verbose=False):
    distances_movies = set(distances.index.unique(level=g_config.MOVIE))
    rois_movies = set(rois.index.unique())
    assert (distances_movies == rois_movies
            ), f'Movies in Distances Series do not match the Movies in RoIs data.\n\tDistances Movies: {distances_movies}\n\tRoIs Movies: {rois_movies}'

    distances_dict = dict()
    movies = distances.index.unique(level=g_config.MOVIE)
    for movID in movies:
        distances_dict[movID] = recenter_distances_single_movie(distances, rois, movID)
        if (verbose):
            print(f'Finished recentering distance data for Movie {movID}.')
    return convert_dict_to_series(distances_dict)


def recenter_distances_single_movie(distances: pd.Series, rois: pd.DataFrame, movID: str):
    assert (movID in distances.index.unique(level=g_config.MOVIE)), f'Couldn\'t find distances for movie {movID}'
    assert (movID in rois.index.unique()), f'Couldn\'t find RoI data for movie {movID}'
    movie_distances = distances.xs(movID, level=g_config.MOVIE)
    roi_median_time, roi_stdev_time = rois.loc[movID, [g_config.T_MEDIAN, g_config.T_STDEV]]
    roi_time = round_down_to_closest_even(roi_median_time)

    # set the TimeStamp index-level to a column to enable manipulations on it:
    temp_df = movie_distances.reset_index(level=g_config.TIMESTAMP)
    temp_df[g_config.TIMESTAMP] = temp_df[g_config.TIMESTAMP] - roi_time

    # reset the TimeStamp column back to the index and convert the DataFrame to a Series
    temp_series = temp_df.set_index(g_config.TIMESTAMP, append=True)
    return temp_series[temp_series.columns[0]]


def cut_tail(distances, rois, verbose=False):
    distances_movies = set(distances.index.unique(level=g_config.MOVIE))
    rois_movies = set(rois.index.unique())
    assert (distances_movies == rois_movies), \
        f'Movies in Distances Series do not match the Movies in RoIs data.\n\tDistances Movies: {distances_movies}\n\tRoIs Movies: {rois_movies}'

    event_times = rois['t_median']
    movies = distances.index.unique(level=g_config.MOVIE)
    for movID in movies:
        assert (movID in distances.index.unique(level=g_config.MOVIE)), f'Couldn\'t find distances for movie {movID}'
        assert (movID in rois.index.unique()), f'Couldn\'t find RoI data for movie {movID}'
        event_time = round_down_to_closest_even(event_times[movID])
        time_cond = distances.index.get_level_values(g_config.TIMESTAMP) < event_time
        movie_cond = distances.index.get_level_values(g_config.MOVIE) != movID
        distances = distances.loc[movie_cond | time_cond]
        if (verbose):
            print(f'Finished cutting distance tail for Movie {movID}.')
    return distances


def cut_prefix(distances, rois, starting_point, verbose=False):
    distances_movies = set(distances.index.unique(level=g_config.MOVIE))
    rois_movies = set(rois.index.unique())
    # assert (distances_movies == rois_movies), \
    #     f'Movies in Distances Series do not match the Movies in RoIs data.\n\tDistances Movies: {distances_movies}\n\tRoIs Movies: {rois_movies}'

    event_times = rois['t_median']
    movies = distances.index.unique(level=g_config.MOVIE)
    for movID in movies:
        assert (movID in distances.index.unique(level=g_config.MOVIE)), f'Couldn\'t find distances for movie {movID}'
        assert (movID in rois.index.unique()), f'Couldn\'t find RoI data for movie {movID}'
        event_time = round_down_to_closest_even(event_times[movID])
        time_cond = distances.index.get_level_values(g_config.TIMESTAMP) > starting_point
        movie_cond = distances.index.get_level_values(g_config.MOVIE) != movID
        distances = distances.loc[movie_cond | time_cond]
        if verbose:
            print(f'Finished cutting distance prefix for Movie {movID}.')
    return distances


def adjust_pupil_data(pupils):
    pupils = remove_prefix_from_raw_data(pupils)

    # Baseline correction by session and subject
    pupils_subjects = set(pupils.index.unique(level=g_config.SUBJECT))
    pupils_sessions = set(pupils.index.unique(level=g_config.SESSION))

    for subj in pupils_subjects:
        for session in pupils_sessions:
            subj_ses = (subj, session, slice(None), slice(None), slice(None))
            pupils.loc[subj_ses, g_config.PUPIL] = \
                ((pupils.loc[subj_ses, g_config.PUPIL] - pupils.loc[subj_ses, g_config.PUPIL].median()) / pupils.loc[
                    subj_ses, g_config.PUPIL]) * 100
    return pupils
