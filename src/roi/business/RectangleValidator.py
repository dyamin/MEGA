import numpy as np

from src.roi import config
from src.roi.models.Movie import *
from src.roi.models.Roi import *
from src.roi.services import DataService


def is_valid(mov: str, log: bool, roi_size: int) -> (bool, float):
    """
    This function validates movies by RoI rect. That means, that a movie is considered to be valid iff the number of subjects
        that marked their RoI inside the defined RoI exceeded the defined threshold
    :return: if the movie is valid, and the proportion of subject inbounds
    """

    majority_threshold = config.inbounds_threshold

    roi = _calc_roi(mov, log, roi_size)
    inbound_subjects, num_subjects = _count_inbounds_subjects(mov, roi, log)

    proportion = inbound_subjects / num_subjects

    return proportion > majority_threshold, proportion


def _extract_movie_dims(mov: str, attr: str = None) -> (int, int) or int:
    if attr is None:
        # Both dims will be returned
        width = DataService.videos_dims.loc[mov][config.width]
        height = DataService.videos_dims.loc[mov][config.height]
        return width, height
    elif attr == "width":
        attr = config.width
    elif attr == "height":
        attr = config.height
    else:
        raise AttributeError(f"Attribute {attr} is not a video dim")

    return DataService.videos_dims.loc[mov, attr]


def _get_absolute(ttype: str, attr: str, mov: str, subject: int = None) -> float:
    """
    :param ttype: type of data needed:
                    aggregated -> from aggregated roibased
                    pre subjects -> from rois per subjects
    :param attr: the attribute to convert from proportional to absolute
    :param mov: the movie of which the data is relevant to
    :param subject: the subject of whom the data is relevant to
    :return: the absolute value of the desired attribute for a specific movie\subject
    """
    if 'x' in attr.lower():
        baseline = _extract_movie_dims(mov, 'width')
    elif 'y' in attr.lower():
        baseline = _extract_movie_dims(mov, 'height')
    else:
        raise AttributeError(f"Absolute value for {attr} is not supported")

    if ttype == "aggregated":
        return baseline * DataService.aggregated_rois.loc[mov][attr] / 100
    elif ttype == "per subjects":
        rois_per_subjects = DataService.rois_per_subjects
        mov_rois = rois_per_subjects[rois_per_subjects[config.movie_level] == mov]
        return baseline * mov_rois.iloc[subject][attr] / 100


def _calc_roi(mov: str, log: bool, roi_size: int) -> Roi:
    """
    :param mov: movie for which roibased will be calculated
    :param log: will log results to file iff True
    :param roi_size: number that specifies to how many rectangles the movie's area is divided
    :return: Adjusted RoI for movie. If RoI deviates from movie's boundaries, it will be adjusted to be inside
        the movie's boundaries, without affecting its size
    """

    roi = _get_roi(mov, roi_size)
    movie = _get_movie(mov)

    # Check boundaries deviations
    roi.left = roi.mid.x - 0.5 * roi.width
    roi.right = roi.mid.x + 0.5 * roi.width
    if roi.left < movie.left:
        # RoI left boundary is out of movie bounds
        # RoI will be indented right
        deviation = abs(roi.left - movie.left)
        roi.left = movie.left
        roi.right = roi.right + deviation

    elif roi.right > movie.right:
        # RoI right boundary is out of movie bounds
        # RoI will be indented left
        deviation = abs(movie.right - roi.right)
        roi.right = movie.right
        roi.left = roi.left - deviation

    else:
        # RoI is inside movie_bounds
        # RoI will not be indented on X axis
        pass

    roi.top = roi.mid.y - 0.5 * roi.height
    roi.bottom = roi.mid.y + 0.5 * roi.height
    if roi.top < movie.top:
        # RoI upper boundary is out of movie bounds
        # RoI will be indented down
        deviation = abs(roi.top - movie.top)
        roi.top = movie.top
        roi.bottom = roi.bottom + deviation

    elif roi.bottom > movie.bottom:
        # RoI lower boundary is out of movie bounds
        # RoI will be indented up
        deviation = abs(movie.bottom - roi.bottom)
        roi.bottom = movie.bottom
        roi.top = roi.top + deviation

    else:
        # RoI is inside movie_bounds
        # RoI will not be indented on X axis
        pass

    msg = f"RoI boundaries in movie {mov}:\n{roi}\n"
    print(msg)
    if log:
        DataService.log(msg)

    return roi


def _get_movie(mov: str) -> Movie:
    width, height = _extract_movie_dims(mov)
    right = width
    left = 0
    top = 0
    bottom = height

    movie = Movie(width, height, left, right, top, bottom)
    print(f"Movie {mov}: {movie}")

    return movie


def _calculate_roi_dims(movie_width: int, movie_height: int, roi_size) -> (float, float):
    width = movie_width / np.sqrt(roi_size)
    height = movie_height / np.sqrt(roi_size)

    print("RoI dims:\nwidth:{}, height:{}".format(width, height))

    return width, height


def _is_inbounds(val: float, roi: Roi, axis: int = 0) -> bool:
    """
    :param val
    :param roi
    :param axis: 0 if horizontal, 1 if vertical
    :return: if val is in roibased according to axis
    """

    if axis == 0:
        return roi.left <= val <= roi.right
    if axis == 1:
        return roi.top <= val <= roi.bottom
    else:
        raise AttributeError(f"Axis {axis} is not supported; 0 - horizontal or 1 - vertical")


def _get_roi(mov: str, roi_size: int) -> Roi:
    """
    :param mov: the movie which the RoI will be returned for
    :param roi_size: number that specifies to how many rectangles the movie's area is divided
    :return: RoI from size 1/roi_size of the movie mov
    """

    movie_width, movie_height = _extract_movie_dims(mov)
    width, height = _calculate_roi_dims(movie_width, movie_height, roi_size)

    # Converting from percentage to absolute RoI mid (x,y) and std
    mid_x = _get_absolute('aggregated', 'X_median', mov)
    mid_y = _get_absolute('aggregated', 'Y_median', mov)
    std_x = _get_absolute('aggregated', 'X_StDev', mov)
    std_y = _get_absolute('aggregated', 'Y_StDev', mov)

    return Roi(width, height, mid_x, mid_y, std_x, std_y)


def _count_inbounds_subjects(mov: str, roi: Roi, log: bool) -> (int, int):
    """
    :return: How many subjects are inbound of a specific movie RoI
    """
    rois_per_subjects = DataService.rois_per_subjects
    mov_rois = rois_per_subjects[rois_per_subjects[config.movie_level] == mov]

    num_subjects = len(mov_rois)
    inbounds_subjects = 0

    for subject_ind in range(num_subjects):

        X = _get_absolute("per subjects", 'X', mov, subject=subject_ind)
        Y = _get_absolute("per subjects", 'Y', mov, subject=subject_ind)

        if _is_inbounds(X, roi, axis=0) and _is_inbounds(Y, roi, axis=1):
            inbounds_subjects += 1
        elif log:
            DataService.log(f"Subject {subject_ind} (X, Y) = ({X},{Y})\n")

    return inbounds_subjects, num_subjects
