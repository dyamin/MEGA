import matplotlib.pyplot as plt
import pandas as pd

from src.signal_processing.distance_from_roi.utils import get_filtered_roi_df
from src.signal_processing.utils import get_event_times

"""
The derivative was more negative and less positive in session B compared to session A. 
Probably due to fast saccades to the ROI, and slow ones when getting far from it. 
"""


def derivative_by_session(session):
    distance_df = get_filtered_roi_df()
    subject_movie = distance_df.xs(('GG', 'mov73'), level=['Subject', 'Movie'])
    subject_movie_session = subject_movie.xs((session,), level=['Session'])
    df = pd.DataFrame({'Date': subject_movie_session.index,
                       'Distance_from_RoI': subject_movie_session.values})
    df['Date'] = pd.to_datetime(df.index * 10 ** 8)
    df.set_index('Date', inplace=True)
    # Plot results
    df.diff().plot()
    plt.axvline(x=get_event_times()['mov73'])
    plt.title(session + ' derivative')
    plt.show()


derivative_by_session('Session A')
derivative_by_session('Session B')
