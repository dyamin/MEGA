import json

from src.signal_processing import config
from src.signal_processing.distance_from_roi.utils import get_filtered_avg_mem_df
from src.signal_processing.utils import get_event_max_times


def difference_between_sessions():
    event_times = get_event_max_times()
    dist_df = get_filtered_avg_mem_df()
    movie2avgdiff = {}
    for mov in config.valid_movies:
        movie_df = dist_df.xs(mov, level='Movie')
        unstacked_df = movie_df.unstack(level=[0, 1])
        subjects = set([i[0] for i in unstacked_df.columns.values])
        max_event_time = event_times[mov]
        sum = 0
        for subj in subjects:
            session_a = unstacked_df[(subj, 'Session A')]
            session_b = unstacked_df[(subj, 'Session B')]
            erp = session_a - session_b
            erp = erp[erp.index < max_event_time]
            sum += erp.sum()
        movie2avgdiff[mov] = sum / len(subjects)

    print(json.dumps({k: v for k, v in sorted(movie2avgdiff.items(), key=lambda item: item[1])}, indent=4))


difference_between_sessions()
