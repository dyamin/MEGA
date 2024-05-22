import matplotlib.pyplot as plt

from src.signal_processing import config
from src.signal_processing.pupil.utils import get_filtered_pupil_radius_df
from src.signal_processing.utils import get_event_times

"""
Is pupil dilation correlate with surprise?
Removing Event Related Potential (Session B ) from Session A
There is no surprise in Session B,
so presumably the average in it contains the components that come from the light (noise).
By removing them we hopefully stay with the surprise component (signal).
"""


def removing_erp_from_session_b():
    event_times = get_event_times()

    pupil_df = get_filtered_pupil_radius_df()
    session_a_mean = pupil_df.groupby(["Movie", "Session", "TimeStamp"]).mean()
    unstacked_df = session_a_mean.unstack(level=[0, 1])
    plt.figure()
    plt.subplots_adjust(hspace=0.5)
    for n, mov in enumerate(config.valid_movies):
        ax = plt.subplot(int(len(config.valid_movies) / 2) + 1, 2, n + 1)
        session_a = unstacked_df[(mov, 'Session A')]
        session_b = unstacked_df[(mov, 'Session B')]
        erp = session_a - session_b
        erp = erp.tail(erp.shape[0] - 500)
        erp.plot(ax=ax)
        plt.axvline(event_times[mov])
        ax.set_title(mov + ', mean: ' + str(round(erp.mean(), 2)))
    plt.tight_layout()
    plt.show()


removing_erp_from_session_b()
