import matplotlib.pyplot as plt

from src.signal_processing import config
from src.signal_processing.distance_from_roi.utils import get_filtered_roi_df
from src.signal_processing.utils import get_event_median_times

"""
Removing Event Related Potential (Saccades to the ROI in Session A) from Session B, 
ERP is calculated by averaging session A over trials. 
In Blue it’s Session B before removing, in Red it’s after removing.
"""


def removing_erp_from_session_b():
    event_times = get_event_median_times()
    distance_df = get_filtered_roi_df()
    session_a_mean = distance_df.groupby(["Movie", "Session", "TimeStamp"]).mean()
    unstacked_df = session_a_mean.unstack(level=[0, 1])
    # Here I assume that all the trials in Session A are phase lock
    plt.figure()
    plt.subplots_adjust(hspace=0.5)
    for n, mov in enumerate(config.valid_movies):
        ax = plt.subplot(int(len(config.valid_movies) / 2) + 1, 2, n + 1)
        erp = unstacked_df[(mov, 'Session A')]
        session_b_ = unstacked_df[(mov, 'Session B')]
        non_phase_locked = session_b_ - erp
        session_b_.plot(ax=ax, color='blue', label='B before')
        non_phase_locked.plot(ax=ax, color='red', label='B after')
        plt.axvline(x=event_times[mov])
        ax.set_title(mov + ', mean: ' + str(round(non_phase_locked.mean(), 2)))
    plt.tight_layout()
    plt.show()


removing_erp_from_session_b()
