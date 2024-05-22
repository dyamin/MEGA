import matplotlib.pyplot as plt
import pandas as pd

from src.signal_processing.pupil.utils import get_filtered_pupil_radius_df


def derivative_by_session(session):
    pupil_df = get_filtered_pupil_radius_df()
    subject_movie = pupil_df.xs(('TC9', 'mov73'), level=['Subject', 'Movie'])
    subject_movie_session = subject_movie.xs((session,), level=['Session'])
    df = pd.DataFrame({'Date': subject_movie_session.index,
                       'Distance_from_RoI': subject_movie_session.values})
    df['Date'] = pd.to_datetime(df.index * 10 ** 8)
    df.set_index('Date', inplace=True)
    df = df.tail(df.shape[0] - 500)

    # Plot results
    df.diff().plot()
    plt.title(session + ' derivative')
    plt.show()


derivative_by_session('Session A')
derivative_by_session('Session B')
