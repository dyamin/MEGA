import pandas as pd

from src import config
from src.features_extraction import config as fa_config
from src.features_extraction.services import DataService
from src.features_extraction.services.DataService import filter_data_to_pre_roi

global duration_grouped


def init():
    global duration_grouped

    filtered_blinks, _ = filter_data_to_pre_roi(DataService.blinks, DataService.rois, onset='Start_Time')

    duration_grouped = filtered_blinks[fa_config.duration].groupby(
        level=[config.SUBJECT, config.SESSION, config.MOVIE]
    )


def get_all_features():
    print("Calculating blinks features_extraction - Start")

    blinks_features = pd.concat([get_rates(),
                                 get_duration_mean(),
                                 get_duration_median(),
                                 get_duration_std(),
                                 get_duration_min(),
                                 get_duration_max(),
                                 # get_inter_blink_duration_mean(),
                                 # get_rate_assymetry(),
                                 # precentage_of_time_blinking()
                                 ], axis=1)

    print("Calculating blinks features_extraction - DONE")
    print("\n*******************************\n")

    return blinks_features


def get_counts() -> pd.Series:
    blinks_count = DataService.blinks.groupby(
        level=[config.SUBJECT, config.SESSION, config.MOVIE]
    ).size().fillna(0).rename('Blinks_Count')
    if 'Measured Eye' in blinks_count:
        return blinks_count.groupby('Measured Eye').mean()
    else:
        return blinks_count


def get_rates() -> pd.Series:
    counts = get_counts()
    movies = counts.index.get_level_values(config.MOVIE)
    durations = DataService.videos_dims.loc[movies, fa_config.duration] / 1000
    durations.index = counts.index

    return counts.divide(durations / 60).rename("Blinks_Per_Minute")


def get_duration_mean() -> pd.Series:
    global duration_grouped

    try:
        return duration_grouped.mean().rename('Blinks_Duration_Mean')
    except NameError:
        init()
        return duration_grouped.mean().rename('Blinks_Duration_Mean')


def get_duration_median() -> pd.Series:
    global duration_grouped

    try:
        return duration_grouped.median().rename('Blinks_Duration_Median')
    except NameError:
        init()
        return duration_grouped.median().rename('Blinks_Duration_Median')


def get_duration_std() -> pd.Series:
    global duration_grouped

    try:
        return duration_grouped.std().rename('Blink_Variability')
    except NameError:
        init()
        return duration_grouped.std().rename('Blink_Variability')


def get_duration_min() -> pd.Series:
    global duration_grouped

    try:
        return duration_grouped.min().rename('Blinks_Duration_Min')
    except NameError:
        init()
        return duration_grouped.min().rename('Blinks_Duration_Min')


def get_duration_max() -> pd.Series:
    global duration_grouped

    try:
        return duration_grouped.max().rename('Blinks_Duration_Max')
    except NameError:
        init()
        return duration_grouped.max().rename('Blinks_Duration_Max')
