import matplotlib.pyplot as plt

from src.signal_processing import config
from src.signal_processing.fixations.utils import get_filtered_fixations_df
from src.signal_processing.utils import get_event_median_times


def auc_fixation_distance():
    event_times = get_event_median_times()
    fixs_df = get_filtered_fixations_df()
    auc_a, auc_b_remembered, auc_b_not_remembered = dict(), dict(), dict()
    subjects = set([i[0] for i in fixs_df.index.values])
    for subj in subjects:
        subj_df = fixs_df.xs(subj, level='Subject')
        session_a = subj_df.xs('Session A', level='Session')
        session_a = session_a[session_a.index.isin([-4, -3], level='Memory')]
        session_b = subj_df.query("Session == 'Session B'")
        session_b.index = session_b.index.droplevel(0)
        session_b_remembered = session_b[session_b.index.isin([4, 3], level='Memory')]
        session_b_not_remembered = session_b[session_b.index.isin([-4, -3], level='Memory')]

        for mov in config.valid_movies:
            event_time = event_times[mov]
            mov_session_a = session_a.query(f"Movie == '{mov}'")
            mov_session_a = mov_session_a.loc[mov_session_a['Start_Time'] < event_time]
            mov_session_br = session_b_remembered.query(f"Movie == '{mov}'")
            mov_session_br = mov_session_br.loc[mov_session_br['Start_Time'] < event_time]
            mov_session_bnr = session_b_not_remembered.query(f"Movie == '{mov}'")
            mov_session_bnr = mov_session_bnr.loc[mov_session_bnr['Start_Time'] < event_time]

            for index, row in mov_session_a.iterrows():
                auc_a[(subj, mov)] = auc_a.get((subj, mov), 0) \
                                     + ((row['Duration'] * row['Distance']) / mov_session_a['Duration'].sum())
            for index, row in mov_session_br.iterrows():
                auc_b_remembered[(subj, mov)] = auc_b_remembered.get((subj, mov), 0) \
                                                + ((row['Duration'] * row['Distance']) / mov_session_br[
                    'Duration'].sum())
            for index, row in mov_session_bnr.iterrows():
                auc_b_not_remembered[(subj, mov)] = auc_b_not_remembered.get((subj, mov), 0) \
                                                    + ((row['Duration'] * row['Distance']) / mov_session_bnr[
                    'Duration'].sum())

    for k in auc_a:
        print(
            f"{k}- 1st: {auc_a.get(k, -1)}, not-remembered: {auc_b_not_remembered.get(k, -1)}, remembered: {auc_b_remembered.get(k, -1)}")

    plt.bar(['1st', 'remembered', 'not remembered'],
            [sum(auc_a.values()) / float(len(auc_a)), sum(auc_b_remembered.values()) / float(len(auc_b_remembered)),
             sum(auc_b_not_remembered.values()) / float(len(auc_b_not_remembered))])
    plt.xlabel("Sessions")
    plt.ylabel("Avg AUC")
    plt.show()


auc_fixation_distance()
