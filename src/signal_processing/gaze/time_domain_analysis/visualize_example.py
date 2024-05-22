import matplotlib.pyplot as plt

from src.signal_processing.gaze.utils import get_filtered_gaze_tuple_position_df

gaze_df = get_filtered_gaze_tuple_position_df()

# visualize example subject and movie
subject_movie = gaze_df.xs(('LG3', 'mov73'), level=['Subject', 'Movie'])
sessionA = subject_movie.xs(('Session A',), level=['Session'])
plt.subplots_adjust(hspace=1)
plt.subplot(211)
plt.scatter(*zip(*sessionA))
plt.title('Session A')
sessionB = subject_movie.xs(('Session B',), level=['Session'])
plt.subplot(212)
plt.scatter(*zip(*sessionB))
plt.title('Session B')
plt.show()
