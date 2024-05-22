import matplotlib.pyplot as plt

from src.signal_processing import config
from src.signal_processing.gaze.utils import get_filtered_gaze_position_df

"""
Average over trials: Session A in blue “x”, and Session B in orange “o”.
There seems to be a repetition effect, in the 2nd viewing there is less scanning of the space
"""

gaze_df = get_filtered_gaze_position_df()
session_a_mean = gaze_df.groupby(["Movie", "Session", "TimeStamp"]).mean()
unstacked_df = session_a_mean.unstack(level=[0, 1])

fig = plt.figure()
plt.subplots_adjust(hspace=0.5)
for n, mov in enumerate(config.valid_movies):
    ax = plt.subplot(int(len(config.valid_movies) / 2) + 1, 2, n + 1)
    session_a_x = unstacked_df[('X_gaze', mov, 'Session A')]
    session_a_y = unstacked_df[('Y_gaze', mov, 'Session A')]
    ax.scatter(session_a_x, session_a_y, marker="x")
    session_b_x = unstacked_df[('X_gaze', mov, 'Session B')]
    session_b_y = unstacked_df[('Y_gaze', mov, 'Session B')]
    ax.scatter(session_b_x, session_b_y, marker="o")
    ax.set_title(mov)

plt.tight_layout()
plt.show()
