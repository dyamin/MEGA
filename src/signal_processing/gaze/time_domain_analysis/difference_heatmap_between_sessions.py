import matplotlib.pyplot as plt
import pandas as pd

from src.signal_processing.gaze.utils import get_filtered_gaze_tuple_position_df

gaze_df = get_filtered_gaze_tuple_position_df()

# visualize example subject and movie
subject_movie = gaze_df.xs(('AH3', 'mov73'), level=['Subject', 'Movie'])
sessionA = subject_movie.xs(('Session A',), level=['Session'])
plt.subplots_adjust(hspace=1)
plt.subplot(211)
zip_session_a_ = pd.DataFrame(sessionA, columns=['X', 'Y'])
plt.scatter(zip_session_a_)
plt.title('Session A')
sessionB = subject_movie.xs(('Session B',), level=['Session'])
plt.subplot(212)
plt.scatter(*zip(*sessionB))
plt.title('Session B')
plt.show()
