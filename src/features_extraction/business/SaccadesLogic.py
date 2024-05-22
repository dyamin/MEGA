import pandas as pd

from src import config
from src.features_extraction import config as fa_config
from src.features_extraction.services import DataService
from src.features_extraction.services.DataService import filter_data_to_pre_roi

global duration_grouped, peak_vel_grouped, ampl_grouped


def init():
    global duration_grouped, peak_vel_grouped, ampl_grouped

    filtered_saccades, _ = filter_data_to_pre_roi(DataService.saccades, DataService.rois)

    duration = filtered_saccades[fa_config.duration]
    peak_vel = filtered_saccades[config.VELOCITY]
    ampl = filtered_saccades[config.AMPLITUDE]

    duration_grouped = duration.groupby(level=[config.SUBJECT, config.SESSION, config.MOVIE])
    peak_vel_grouped = peak_vel.groupby(level=[config.SUBJECT, config.SESSION, config.MOVIE])
    ampl_grouped = ampl.groupby(level=[config.SUBJECT, config.SESSION, config.MOVIE])


def get_all_features():
    print("Calculating saccades features_extraction - START")

    saccades_features = pd.concat([get_counts(),
                                   get_rates(),
                                   get_duration_mean(),
                                   get_duration_median(),
                                   get_duration_std(),
                                   get_mean_peak_vel_mean(),
                                   get_mean_peak_vel_median(),
                                   get_mean_peak_vel_std(),
                                   get_mean_ampl_mean(),
                                   get_mean_ampl_median(),
                                   get_mean_ampl_std(),
                                   ], axis=1)

    print("Calculating saccades features_extraction - DONE")
    print("\n*******************************\n")

    return saccades_features


def get_counts() -> pd.Series:
    return DataService.saccades.groupby(
        level=[config.SUBJECT, config.SESSION, config.MOVIE]
    ).size().fillna(0).rename('Saccades_Count')


def get_rates() -> pd.Series:
    counts = get_counts()
    movies = counts.index.get_level_values(config.MOVIE)
    durations = DataService.videos_dims.loc[movies, fa_config.duration] / 1000
    durations.index = counts.index

    return counts.divide(durations).rename("Saccades_Rate")


def get_duration_mean() -> pd.Series:
    global duration_grouped

    try:
        return duration_grouped.mean().rename('Saccades_Duration_Mean')
    except NameError:
        init()
        return duration_grouped.mean().rename('Saccades_Duration_Mean')


def get_duration_median() -> pd.Series:
    global duration_grouped

    try:
        return duration_grouped.median().rename('Saccades_Duration_Median')
    except NameError:
        init()
        return duration_grouped.median().rename('Saccades_Duration_Median')


def get_duration_std() -> pd.Series:
    global duration_grouped

    try:
        return duration_grouped.std().rename('Saccades_Duration_StDev')
    except NameError:
        init()
        return duration_grouped.std().rename('Saccades_Duration_StDev')


def get_mean_peak_vel_mean():
    global peak_vel_grouped

    try:
        return peak_vel_grouped.mean().rename('Saccades_Peak_Velocitiy_Mean')
    except NameError:
        init()
        return peak_vel_grouped.mean().rename('Saccades_Peak_Velocitiy_Mean')


def get_mean_peak_vel_median():
    global peak_vel_grouped

    try:
        return peak_vel_grouped.median().rename('Saccades_Peak_Velocitiy_Median')
    except NameError:
        init()
        return peak_vel_grouped.median().rename('Saccades_Peak_Velocitiy_Median')


def get_mean_peak_vel_std():
    global peak_vel_grouped

    try:
        return peak_vel_grouped.std().rename('Saccades_Peak_Velocitiy_StDev')
    except NameError:
        init()
        return peak_vel_grouped.std().rename('Saccades_Peak_Velocitiy_StDev')


def get_mean_ampl_mean():
    global ampl_grouped

    try:
        return ampl_grouped.mean().rename('Saccades_Amplitude_Mean')
    except NameError:
        init()
        return ampl_grouped.mean().rename('Saccades_Amplitude_Mean')


def get_mean_ampl_median():
    global ampl_grouped

    try:
        return ampl_grouped.median().rename('Saccades_Amplitude_Median')
    except NameError:
        init()
        return ampl_grouped.median().rename('Saccades_Amplitude_Median')


def get_mean_ampl_std():
    global ampl_grouped

    try:
        return ampl_grouped.std().rename('Saccades_Amplitude_StDev')
    except NameError:
        init()
        return ampl_grouped.std().rename('Saccades_Amplitude_StDev')
