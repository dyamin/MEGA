import matplotlib.pyplot as plt

from src.signal_processing.distance_from_roi.utils import get_filtered_roi_df

"""
Averaging over trials: I tried to find phase locked events (ERPs), 
the vertical line is the median surprise event time (as marked by the viewers)
"""


def avg_over_trials():
    event_times = get_event_times()
    distance_df = get_filtered_roi_df()
    mov = 'mov75'
    movie_df = distance_df.xs((mov), level='Movie')
    unstacked_df = movie_df.unstack(level=[0, 1])
    # Here I assume that all the trials in Session A are phase lock
    fig = plt.figure()
    plt.subplots_adjust(hspace=0.5)
    subjects = set([i[0] for i in unstacked_df.columns.values])
    for n, subj in enumerate(subjects):
        ax = plt.subplot(int(len(subjects) / 2) + 1, 2, n + 1)
        session_a_ = unstacked_df[(subj, 'Session A')]
        session_a_.plot(ax=ax, color='blue', label='A')
        session_b_ = unstacked_df[(subj, 'Session B')]
        session_b_.plot(ax=ax, color='red', label='B')
        ax.axvline(x=event_times[mov])
        ax.set_title(subj)
    plt.tight_layout()
    plt.show()


avg_over_trials()
