import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from src.signal_processing.distance_from_roi.utils import get_filtered_roi_df

distance_df = get_filtered_roi_df()
subject_movie = distance_df.xs(('GG', 'mov73'), level=['Subject', 'Movie'])
subject_movie_session = subject_movie.xs(('Session A',), level=['Session'])
df = pd.DataFrame({'Date': subject_movie_session.index,
                   'Distance_from_RoI': subject_movie_session.values})
df['Date'] = pd.to_datetime(df.index * 10 ** 8)
df.set_index('Date', inplace=True)

# Finding local min
n = 30  # number of points to be checked before and after
df['min'] = df.iloc[argrelextrema(df.Distance_from_RoI.values, np.less_equal,
                                  order=n)[0]]['Distance_from_RoI']
# Plot results
plt.scatter(df.index, df['min'], c='r')
plt.plot(df.index, df['Distance_from_RoI'])
plt.show()
