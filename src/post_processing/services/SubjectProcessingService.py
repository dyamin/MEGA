import os
import time

import pandas as pd

from src import config
from src.post_processing.business.AggregationLogic import aggregate_data
from src.post_processing.business.AppendingMemoryLogic import append_memory_to_subject_dataframe
from src.post_processing.business.BlinksLogic import mark_blinks, extract_blinks_df, aggregate_blinks
from src.post_processing.business.DistanceCalculationLogic import calculate_distances
from src.post_processing.business.EngbertFixationsLogic import calculate_engbert_fixations, aggregate_fixations
from src.post_processing.business.GazePrefixRemovalLogic import remove_prefix_from_raw_data as remove_prefix
from src.post_processing.business.SaccadesLogic import calculate_saccades, aggregate_saccades
from src.post_processing.config import use_eyelink_parser
from src.post_processing.models.Subject import Subject


def features_already_extracted(subdir_path: str):
    return os.path.exists(os.path.join(subdir_path, 'blinks')) and os.path.exists(
        os.path.join(subdir_path, 'fixations')) \
        and os.path.exists(os.path.join(subdir_path, 'saccades'))


def process_subject(subject_directory: str, subject_memory: pd.Series,
                    should_remove_prefix=True, should_save=True, verbose=False):
    t_start = time.perf_counter()
    subject = Subject(subject_directory, subject_memory, config.number_of_sessions)
    subject_id = subject.get_subject_id()
    if verbose:
        print(f"Starting to process subject {subject_id}...")
    raw_data = aggregate_data(subject.get_main_directory())

    # raw_data = remove_invalid_data(raw_data)

    if should_remove_prefix:
        raw_data = remove_prefix(raw_data)

    if use_eyelink_parser:  # and features_already_extracted(subject.get_main_directory()):
        blinks_df = aggregate_blinks(subject.get_main_directory(), show=True)
        fixations_df = aggregate_fixations(subject.get_main_directory())
        saccades_df = aggregate_saccades(subject.get_main_directory())

    else:
        raw_data = mark_blinks(raw_data)
        blinks_df = extract_blinks_df(raw_data)
        fixations_df = calculate_engbert_fixations(raw_data)
        saccades_df = calculate_saccades(raw_data)

    # add extra columns:
    aggregated_rois = _get_aggregated_rois()
    video_dims = pd.read_pickle(config.VIDEO_DIMS_FILE_PATH)

    if not raw_data.empty:
        raw_data[config.DISTANCE], raw_data[config.DVA], raw_data[config.SQRT_DVA] = calculate_distances(raw_data,
                                                                                                         aggregated_rois,
                                                                                                         video_dims,
                                                                                                         False)
        raw_data = append_memory_to_subject_dataframe(subject_id, raw_data, subject.get_memory())
    if not blinks_df.empty:
        blinks_df = append_memory_to_subject_dataframe(subject_id, blinks_df, subject.get_memory())
    if not fixations_df.empty:
        fixations_df[config.DISTANCE], fixations_df[config.DVA], fixations_df[config.SQRT_DVA] = calculate_distances(
            fixations_df, aggregated_rois,
            video_dims,
            True)
        fixations_df = append_memory_to_subject_dataframe(subject_id, fixations_df, subject.get_memory())
    if not saccades_df.empty:
        saccades_df = append_memory_to_subject_dataframe(subject_id, saccades_df, subject.get_memory())

    if should_save:
        if verbose:
            print("\tSaving results...")
        from src.utils import save_df_to_pkl
        save_df_to_pkl(raw_data, f'{subject_id}_raw_gaze', config.data_dir, config.pickling_protocol)
        save_df_to_pkl(blinks_df, f'{subject_id}_blinks', config.data_dir, config.pickling_protocol)
        save_df_to_pkl(fixations_df, f'{subject_id}_fixations', config.data_dir, config.pickling_protocol)
        save_df_to_pkl(saccades_df, f'{subject_id}_saccades', config.data_dir, config.pickling_protocol)

    t_finish = time.perf_counter()  # in seconds
    if verbose:
        print(f'\tFinished in {(t_finish - t_start):.2f} seconds.')
        print('----------------------------------------------------')
    return raw_data, blinks_df, fixations_df, saccades_df


def _get_aggregated_rois() -> pd.DataFrame:
    return pd.read_pickle(os.path.join(config.rois_dir, 'aggregated_RoIs' + config.PKL))


def remove_invalid_data(raw_data):
    # Counts the number of invalid data points relative to the total number of data points
    # for each movie and session and prints the results
    if not raw_data.empty:
        for session in raw_data.index.get_level_values(config.SESSION).unique():
            session_data = raw_data.xs(session, level=config.SESSION)
            for movie in session_data.index.get_level_values(config.MOVIE).unique():
                # get the data for the current session and movie
                movie_data = session_data.xs(movie, level=config.MOVIE)
                # invalid data is a row with at least one NaN value
                invalid_data = movie_data[movie_data.isnull().any(axis=1)]
                ratio = len(invalid_data) / len(movie_data)
                # if the ratio is greater than config.INVALID_DATA_RATIO, remove the movie data from both sessions
                if ratio > config.INVALID_DATA_RATIO:
                    raw_data.drop(index=movie, level=config.MOVIE, inplace=True)
                    print(f'Removed {movie} data')

    # Remove the rows with at least one NaN value
    raw_data.dropna(how='any', inplace=True)

    # Remove the movies that are not appearing in both sessions
    for movie in raw_data.index.get_level_values(config.MOVIE).unique():
        if len(raw_data.xs(movie, level=config.MOVIE).index.get_level_values(config.SESSION).unique()) < 2:
            raw_data.drop(index=movie, level=config.MOVIE, inplace=True)
            print(f'Movie: {movie} appears only in one session. Removed from data.')
    return raw_data
