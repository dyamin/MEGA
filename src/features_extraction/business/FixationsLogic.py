import pandas as pd

from src import config
from src.features_extraction import config as fa_config
from src.features_extraction.services import DataService
from src.features_extraction.services.DataService import filter_data_to_pre_roi

global duration_grouped, pre_pupil_grouped, post_pupil_grouped


def init():
    global duration_grouped, pre_pupil_grouped,post_pupil_grouped

    pre_fixations, post_fixations = filter_data_to_pre_roi(DataService.fixations, DataService.rois)

    duration_grouped = pre_fixations[fa_config.duration].groupby(
        level=[config.SUBJECT, config.SESSION, config.MOVIE]
    )
    pre_pupil_grouped = pre_fixations[config.PUPIL].groupby(
        level=[config.SUBJECT, config.SESSION, config.MOVIE]
    )
    post_pupil_grouped = post_fixations[config.PUPIL].groupby(
        level=[config.SUBJECT, config.SESSION, config.MOVIE]
    )


def get_all_features():
    print("Calculating fixations features_extraction - START")

    fixations_features = pd.concat([get_counts(),
                                    get_rates(),
                                    get_duration_mean(),
                                    get_duration_median(),
                                    get_duration_std(),
                                    get_pre_pupil_mean(),
                                    get_pre_pupil_median(),
                                    get_pre_pupil_std(),
                                    get_post_pupil_mean(),
                                    get_post_pupil_median(),
                                    get_post_pupil_std(),
                                    ], axis=1)

    print("Calculating fixations features_extraction - DONE")
    print("\n*******************************\n")

    return fixations_features


def get_counts() -> pd.Series:
    return DataService.fixations.groupby(
        level=[config.SUBJECT, config.SESSION, config.MOVIE]
    ).size().fillna(0).rename('Fixations_Count')


def get_rates() -> pd.Series:
    counts = get_counts()
    movies = counts.index.get_level_values(config.MOVIE)
    durations = DataService.videos_dims.loc[movies, fa_config.duration] / 1000
    durations.index = counts.index

    return counts.divide(durations).rename("Fixations_Rate")


def get_duration_mean() -> pd.Series:
    global duration_grouped

    try:
        return duration_grouped.mean().rename('Fixations_Duration_Mean_Pre')
    except NameError:
        init()
        return duration_grouped.mean().rename('Fixations_Duration_Mean_Pre')


def get_duration_median() -> pd.Series:
    global duration_grouped

    try:
        return duration_grouped.median().rename('Fixations_Duration_Median_Pre')
    except NameError:
        init()
        return duration_grouped.median().rename('Fixations_Duration_Median_Pre')


def get_duration_std() -> pd.Series:
    global duration_grouped

    try:
        return duration_grouped.std().rename('Fixations_Duration_StDev_Pre')
    except NameError:
        init()
        return duration_grouped.std().rename('Fixations_Duration_StDev_Pre')


def get_post_pupil_mean():
    global post_pupil_grouped

    try:
        return post_pupil_grouped.mean().rename('Fixation_Pupil_Mean_Post')
    except NameError:
        init()
        return post_pupil_grouped.mean().rename('Fixation_Pupil_Mean_Post')


def get_post_pupil_median():
    global pre_pupil_grouped

    try:
        return post_pupil_grouped.median().rename('Fixation_Pupil_Median_Post')
    except NameError:
        init()
        return post_pupil_grouped.median().rename('Fixation_Pupil_Median_Post')


def get_post_pupil_std():
    global post_pupil_grouped

    try:
        return post_pupil_grouped.std().rename('Fixation_Pupil_StDev_Post')
    except NameError:
        init()
        return post_pupil_grouped.std().rename('Fixation_Pupil_StDev_Post')


def get_pre_pupil_mean():
    global pre_pupil_grouped

    try:
        return pre_pupil_grouped.mean().rename('Fixation_Pupil_Mean_Pre')
    except NameError:
        init()
        return pre_pupil_grouped.mean().rename('Fixation_Pupil_Mean_Pre')


def get_pre_pupil_median():
    global pre_pupil_grouped

    try:
        return pre_pupil_grouped.median().rename('Fixation_Pupil_Median_Pre')
    except NameError:
        init()
        return pre_pupil_grouped.median().rename('Fixation_Pupil_Median_Pre')


def get_pre_pupil_std():
    global pre_pupil_grouped

    try:
        return pre_pupil_grouped.std().rename('Fixation_Pupil_StDev_Pre')
    except NameError:
        init()
        return pre_pupil_grouped.std().rename('Fixation_Pupil_StDev_Pre')
