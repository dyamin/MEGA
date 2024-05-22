import matplotlib.pyplot as plt

from src.signal_processing import config
from src.signal_processing.distance_from_roi.utils import get_filtered_roi_df
from src.signal_processing.utils import get_event_times

"""
Averaging over trials: I tried to find phase locked events (ERPs), 
the vertical line is the median surprise event time (as marked by the viewers)
"""


def avg_over_trials():
    event_times = get_event_times()
    distance_df = get_filtered_roi_df()
    session_a_mean = distance_df.groupby(["Movie", "Session", "TimeStamp"]).mean()
    unstacked_df = session_a_mean.unstack(level=[0, 1])
    # Here I assume that all the trials in Session A are phase lock
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.5)
    for n, mov in enumerate(config.valid_movies):
        ax = plt.subplot(int(len(config.valid_movies) / 2) + 1, 2, n + 1)
        session_a_ = unstacked_df[(mov, 'Session A')]
        session_a_.plot(ax=ax, color='blue', label='A')
        session_b_ = unstacked_df[(mov, 'Session B')]
        session_b_.plot(ax=ax, color='red', label='B')
        ax.axvline(x=event_times[mov])
        ax.set_title(mov)
    plt.tight_layout()
    plt.show()


avg_over_trials()
