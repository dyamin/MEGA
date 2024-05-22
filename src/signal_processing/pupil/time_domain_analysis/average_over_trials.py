import matplotlib.pyplot as plt

from src.signal_processing import config
from src.signal_processing.pupil.utils import get_filtered_pupil_radius_df
from src.signal_processing.utils import get_event_median_times

# The first and still most used method for analysing the pupil diameter (see Laeng and Alnaes
# (2019b) for a review) either disregards pupil data as time series or approximates it by dividing
# the pupil response into epochs or bins, typically based on an equal number of samples (e.g.,
# Bianco et al. (2019); Bochynska et al. (2021); Zavagno et al. (2017))

# Average over trials to find Event Related Potential (ERP)
# Pupil radius in Session A (blue) and Session B (red)
event_times = get_event_median_times()

pupil_df = get_filtered_pupil_radius_df()
session_a_mean = pupil_df.groupby(["Movie", "Session", "TimeStamp"]).mean()
unstacked_df = session_a_mean.unstack(level=[0, 1])

fig = plt.figure()
plt.subplots_adjust(hspace=0.5)
for n, mov in enumerate(config.valid_movies):
    ax = plt.subplot(int(len(config.valid_movies) / 2) + 1, 2, n + 1)
    session_a_ = unstacked_df[(mov, 'Session A')]
    session_a_ = session_a_.tail(session_a_.shape[0] - 500)
    session_a_.plot(ax=ax, color='blue', label='A')
    session_b_ = unstacked_df[(mov, 'Session B')]
    session_b_ = session_b_.tail(session_b_.shape[0] - 500)
    session_b_.plot(ax=ax, color='red', label='B')
    plt.axvline(event_times[mov])
    ax.set_title(mov)

plt.tight_layout()
plt.show()
