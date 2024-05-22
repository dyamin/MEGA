import matplotlib.pyplot as plt

from src import config
from src.statistical_analysis.utils import get_event_median_times, get_all_valid_subject_data_df

distance_df = get_all_valid_subject_data_df()

# visualize example subject and movie
movie = 'mov10'
subj = 'HB81'

subject_movie = distance_df.xs((subj, movie), level=['Subject', 'Movie'])
sessionA = subject_movie.xs(('Session A',), level=['Session'])
plt.subplots_adjust(hspace=1)
plt.subplot(211)
plt.plot(sessionA.index.get_level_values('TimeStamp'), sessionA[config.DVA])
plt.title('Session A')
plt.axvline(x=get_event_median_times()[movie])
sessionB = subject_movie.xs(('Session B',), level=['Session'])
plt.subplot(212)
plt.plot(sessionB.index.get_level_values('TimeStamp'), sessionB[config.DVA])
plt.title('Session B')
plt.axvline(x=get_event_median_times()[movie])
plt.show()
print(max(sessionA.index.get_level_values('TimeStamp')))
