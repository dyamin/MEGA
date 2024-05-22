import json

import matplotlib.pyplot as plt
import pandas as pd

from src.signal_processing import config
from src.signal_processing.distance_from_roi.utils import get_filtered_roi_df, get_filtered_avg_mem_df
from src.signal_processing.utils import get_event_median_times


def difference_between_sessions():
    event_times = get_event_median_times()
    dist_df = get_filtered_roi_df()
    movie2avgdiff = {}
    for mov in config.valid_movies:
        movie_df = dist_df.xs((mov), level='Movie')
        unstacked_df = movie_df.unstack(level=[0, 1])
        subjects = set([i[0] for i in unstacked_df.columns.values])
        event_time = event_times[mov]
        sum = 0
        for subj in subjects:
            session_a = unstacked_df[(subj, 'Session A')]
            session_b = unstacked_df[(subj, 'Session B')]
            erp = session_a - session_b
            erp = erp[erp.index < event_time]
            sum += erp.sum()
        movie2avgdiff[mov] = sum / len(subjects)
    print(json.dumps({k: v for k, v in sorted(movie2avgdiff.items(), key=lambda item: item[1])}, indent=4))

    mem_df = get_filtered_avg_mem_df()
    mem_df['Distance'] = pd.Series(movie2avgdiff)
    mem_df.plot(x='Memory', y='Distance', kind='scatter')
    plt.show()


difference_between_sessions()
