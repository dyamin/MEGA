import matplotlib.pyplot as plt
import pandas as pd

from src.signal_processing import config
from src.signal_processing.distance_from_roi.utils import get_filtered_roi_df, get_filtered_mem_df
from src.signal_processing.utils import get_event_median_times


def difference_between_sessions():
    event_times = get_event_median_times()
    dist_df = get_filtered_roi_df()
    mov_subj_to_diff = {}
    for mov in config.valid_movies:
        movie_df = dist_df.xs((mov), level='Movie')
        unstacked_df = movie_df.unstack(level=[0, 1])
        subjects = set([i[0] for i in unstacked_df.columns.values])
        event_time = event_times[mov]
        for subj in subjects:
            diff_sum = 0
            session_a = unstacked_df[(subj, 'Session A')]
            session_b = unstacked_df[(subj, 'Session B')]
            erp = session_a - session_b
            erp = erp[erp.index < event_time]
            diff_sum += erp.sum()
            mov_subj_to_diff[(subj, mov)] = diff_sum

    mem_df = get_filtered_mem_df()
    dist_df = pd.DataFrame.from_dict(mov_subj_to_diff, orient='index', columns=['Distance'])
    dist_df.index = pd.MultiIndex.from_tuples(mov_subj_to_diff.keys(), names=['Subject', 'Movie'])
    merged_df = pd.merge(mem_df, dist_df, left_index=True, right_index=True)
    merged_df.plot(x='Memory', y='Distance', kind='scatter')
    plt.show()


difference_between_sessions()
