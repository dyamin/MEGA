import matplotlib.pyplot as plt

from src.signal_processing.pupil.utils import get_filtered_pupil_radius_df
from src.signal_processing.utils import get_event_median_times

pupil_df = get_filtered_pupil_radius_df()
event_times = get_event_median_times()

# visualize example subject and movie
mov = 'mov73'
subject_movie = pupil_df.xs(('TC9', mov), level=['Subject', 'Movie'])
session_a = subject_movie.xs(('Session A',), level=['Session'])
plt.subplots_adjust(hspace=1)
plt.subplot(211)
plt.plot(session_a.index.get_level_values('TimeStamp'), session_a.values)
plt.axvline(event_times[mov])
plt.title('Session A')
session_b = subject_movie.xs(('Session B',), level=['Session'])
plt.subplot(212)
plt.plot(session_b.index.get_level_values('TimeStamp'), session_b.values)
plt.axvline(event_times[mov])
plt.title('Session B')
plt.show()
print(max(session_a.index.get_level_values('TimeStamp')))
