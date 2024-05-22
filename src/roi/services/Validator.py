import os

import matplotlib.pyplot as plt
import pandas as pd

from src.roi import config
from src.roi.business import RectangleValidator, TimeValidator
from src.roi.services import DataService


def validate(metric: str, log: bool = False, load=False) -> None:
    valid_movies = None

    if metric == "rect 9":
        valid_movies, percentages = _by_rect(size=9, log=log)
    elif metric == "rect 16":
        valid_movies, percentages = _by_rect(size=16, log=log)
    elif metric == "time":
        valid_movies, percentages = by_t(std_window=config.t_std_window, log=log)
    elif metric == "graph":
        _draw_graph(log, load)
    else:
        print(f"The {metric} metric is not supported")

    print(valid_movies)


def _by_rect(size: int, log: bool = False) -> (list, dict):
    """
    :param size: relative size of RoI from the movie's screen (size n -> RoI is 1/n of the screen)
    :param log: log results iff True
    :return: list of valid movies and a dictionary with percentage of valid subjects per movie
    """

    if log:
        DataService.open_log(f"rect_{size}")

    valid_movies = list()
    percentages = dict()

    for mov in DataService.videos_dims.index:

        valid, proportion = RectangleValidator.is_valid(mov, log, roi_size=size)
        percentage = proportion * 100
        percentages[mov] = percentage

        if valid:
            valid_movies.append(mov)

        if log:
            # Writing to log file
            msg = "is valid" if valid else "is not valid"
            DataService.log("Movie {} {}:\n".format(mov, msg))
            DataService.log(
                "{}% of the subjects in movie '{}' were inside the defined RoI\n\n".format(percentage, mov))

    if log:
        DataService.log("There are {} valid movies :\n{}".format(len(valid_movies), valid_movies))
        DataService.close_log()

    return valid_movies, percentages


def by_t(std_window: int = 2, log: bool = False) -> (list, dict):
    """
    :param std_window: number of STDs to count pre RoI's as a range of validity
    :param log: log results iff True
    :return: list of valid movies and a dictionary with percentage of valid subjects per movie
    """

    if log:
        DataService.open_log(f"time")

    valid_movies = list()
    percentages = dict()

    for mov in DataService.videos_dims.index:

        valid, proportion = TimeValidator.is_valid(mov, log, std_window=std_window)
        percentage = proportion * 100
        percentages[mov] = percentage

        if valid:
            valid_movies.append(mov)

        if log:
            # Writing to log file
            msg = "is valid" if valid else "is not valid"
            DataService.log("Movie {} {}:\n".format(mov, msg))
            DataService.log(
                "{}% of the subjects in movie '{}' were inside the defined RoI\n\n".format(percentage, mov))

    if log:
        DataService.log("There are {} valid movies :\n{}".format(len(valid_movies), valid_movies))
        DataService.close_log()

    return valid_movies, percentages


def _draw_graph(log: bool, load: bool):
    if load:
        path = os.path.join(config.data_dir, config.validity_filename)
        validity = pd.read_pickle(path)
    else:
        _, percentages_rect9 = _by_rect(size=9, log=log)
        _, percentages_rect16 = _by_rect(size=16, log=log)
        _, percentages_time = by_t(std_window=config.t_std_window, log=log)

        validity = pd.concat([
            pd.Series(percentages_rect9).rename(config.rect9_column),
            pd.Series(percentages_rect16).rename(config.rect16_column),
            pd.Series(percentages_time).rename(config.time_column)
        ], axis=1)
        validity.reset_index(inplace=True)
        validity['index'] = validity['index'].apply(lambda ind: int(ind[3:]))

    _plot(validity, log=log, filterr=True)
    _plot(validity, log=log, filterr=False)


def _plot(validity: pd.DataFrame, log: bool, filterr: bool) -> None:
    graph_name = "Validity"

    if filterr:
        thresh = config.inbounds_threshold * 100
        validity = validity[validity['Rectangle 9 (% subjects)'] >= thresh]
        validity = validity[validity['Rectangle 16 (% subjects)'] >= thresh]
        validity = validity[validity['Time (% subjects)'] >= thresh]

    validity = validity.sort_values(by=['Time (% subjects)', 'Rectangle 16 (% subjects)',
                                        'Rectangle 9 (% subjects)'],
                                    ascending=False)

    print("Plotting graph - START")
    validity.plot(x="index", y=[config.rect9_column, config.rect16_column, config.time_column],
                  kind="bar", figsize=(24, 8))
    print("Plotting graph - DONE")

    if filterr:
        plt.ylim(70, 100)
        graph_name += "_filtered"

    plt.xlabel("Movie", fontsize=18)
    plt.ylabel("% Valid subjects", fontsize=18)
    plt.title(graph_name, fontsize=36)
    plt.grid(alpha=0.5, linestyle='--')

    if log:
        DataService.write_pickle(validity, config.validity_filename)
        fname = os.path.join(config.log_dir, f"{graph_name}.jpeg")
        plt.savefig(fname)

    plt.show()
