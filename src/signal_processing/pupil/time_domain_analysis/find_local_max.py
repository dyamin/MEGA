import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from src.signal_processing.pupil.utils import get_filtered_pupil_radius_df

pupil_df = get_filtered_pupil_radius_df()
subject_movie = pupil_df.xs(('TC9', 'mov35'), level=['Subject', 'Movie'])
subject_movie_session = subject_movie.xs(('Session A',), level=['Session'])
df = pd.DataFrame({'Date': subject_movie_session.index,
                   'Pupil': subject_movie_session.values})
df['Date'] = pd.to_datetime(df.index * 10 ** 8)
df.set_index('Date', inplace=True)

# Finding local maxima
n = 5  # number of points to be checked before and after
df['max'] = df.iloc[argrelextrema(df.Pupil.values, np.greater_equal,
                                  order=n)[0]]['Pupil']
# Plot results
plt.scatter(df.index, df['max'], c='g')
plt.plot(df.index, df['Pupil'])

plt.show()
