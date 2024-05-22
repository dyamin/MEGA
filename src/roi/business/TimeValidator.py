from src.roi import config
from src.roi.services import DataService


def is_valid(mov: str, log: bool, std_window: int) -> (bool, float):
    """
    This function validates movies by time. That means, that a movie is considered to be valid iff the number of subjects
        that marked their RoI's t inside the defined time window exceeded the defined threshold
    :return: if the movie is valid, and the proportion of subject in window
    """

    majority_threshold = config.inbounds_threshold

    rois_per_subjects = DataService.rois_per_subjects

    mov_rois = rois_per_subjects[rois_per_subjects[config.movie_level] == mov]
    num_subjects = len(mov_rois)
    window_start, window_end = _calc_window(mov, std_window, log)
    in_window_cnt = 0

    for subject_ind in range(num_subjects):

        t = mov_rois.iloc[subject_ind][config.t]

        if _is_in_window(t, window_start, window_end):
            in_window_cnt += 1
        elif log:
            DataService.log(f"Subject {subject_ind} t = {t}\n")

    proportion = in_window_cnt / num_subjects

    return proportion > majority_threshold, proportion


def _calc_window(mov: str, std_window: int, log: bool) -> (float, float):
    aggregated_rois = DataService.aggregated_rois
    t_mid = aggregated_rois.loc[mov, config.t_mid]
    t_std = aggregated_rois.loc[mov, config.t_std]

    t_deviation = (std_window / 2) * t_std
    t_end = t_mid + t_deviation
    t_start = t_mid - t_deviation

    msg = f"RoI time STD: {t_std}\nRoI time window in movie {mov}:\n[{t_start}, {t_end}]\n"
    print(msg)
    if log:
        DataService.log(msg)

    return t_start, t_end


def _is_in_window(t: float, start: float, end: float) -> bool:
    return start <= t <= end
