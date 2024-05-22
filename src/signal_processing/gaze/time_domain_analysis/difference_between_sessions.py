import json

import numpy as np

from src.signal_processing import config
from src.signal_processing.gaze.utils import get_filtered_gaze_position_df
from src.signal_processing.utils import get_event_max_times


def distance(x, y):
    return np.sqrt(np.sum([(a - b) * (a - b) for a, b in zip(x, y)]))


def get_session_a_gaze(df, subj):
    return list(zip(df[('X_gaze', subj, 'Session A')], df[('Y_gaze', subj, 'Session A')]))


def difference_between_sessions():
    event_times = get_event_max_times()
    pupil_df = get_filtered_gaze_position_df()
    movie2avgdiff = {}
    for mov in config.valid_movies:
        movie_df = pupil_df.xs((mov), level='Movie')
        unstacked_df = movie_df.unstack(level=[0, 1])
        subjects = set([i[1] for i in unstacked_df.columns.values])
        max_event_time = event_times[mov]
        sum = 0
        for subj in subjects:
            for other_subj in subjects:
                if subj != other_subj:
                    subj_gaze = get_session_a_gaze(unstacked_df, subj)
                    other_subj_gaze = get_session_a_gaze(unstacked_df, other_subj)
                    dist = distance(subj_gaze, other_subj_gaze)
                    erp = erp[erp.index < max_event_time]
                    sum += erp.sum()
        movie2avgdiff[mov] = sum / len(subjects)

    print(json.dumps({k: v for k, v in sorted(movie2avgdiff.items(), key=lambda item: item[1])}, indent=4))


difference_between_sessions()
