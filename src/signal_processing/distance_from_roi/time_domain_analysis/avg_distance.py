import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.statistical_analysis.utils import get_event_median_times, get_all_valid_subject_data_df


def auc_fixation_distance():
    event_times = get_event_median_times()
    all_df = get_all_valid_subject_data_df()
    auc_a, auc_b_remembered, auc_b_not_remembered = dict(), dict(), dict()
    subjects = set([i[0] for i in all_df.index.values])
    for subj in subjects:
        subj_df = all_df.xs(subj, level='Subject')
        session_a = subj_df.xs('Session A', level='Session')
        session_a = session_a[session_a.index.isin([-4, -3], level='Memory')]
        session_b = subj_df.query("Session == 'Session B'")
        session_b.index = session_b.index.droplevel(0)
        session_b_remembered = session_b[session_b.index.isin([4, 3], level='Memory')]
        session_b_not_remembered = session_b[session_b.index.isin([-4, -3], level='Memory')]

        for mov in config.valid_movies:
            event_time = event_times[mov]
            mov_session_a = session_a.query(f"Movie == '{mov}'")
            mov_session_a = mov_session_a[mov_session_a.index.get_level_values(1) < event_time]
            mov_session_a = mov_session_a.loc[200 < mov_session_a.index.get_level_values(1)]
            mov_session_br = session_b_remembered.query(f"Movie == '{mov}'")
            mov_session_br = mov_session_br.loc[mov_session_br.index.get_level_values(1) < event_time]
            mov_session_br = mov_session_br.loc[200 < mov_session_br.index.get_level_values(1)]
            mov_session_bnr = session_b_not_remembered.query(f"Movie == '{mov}'")
            mov_session_bnr = mov_session_bnr.loc[mov_session_bnr.index.get_level_values(1) < event_time]
            mov_session_bnr = mov_session_bnr.loc[200 < mov_session_bnr.index.get_level_values(1)]

            for index, row in mov_session_a.iterrows():
                auc_a[(subj, mov)] = auc_a.get((subj, mov), 0) \
                                     + (row['Distance_from_RoI'] / len(mov_session_a))
            for index, row in mov_session_br.iterrows():
                auc_b_remembered[(subj, mov)] = auc_b_remembered.get((subj, mov), 0) \
                                                + (row['Distance_from_RoI'] / len(mov_session_br))
            for index, row in mov_session_bnr.iterrows():
                auc_b_not_remembered[(subj, mov)] = auc_b_not_remembered.get((subj, mov), 0) \
                                                    + (row['Distance_from_RoI'] / len(mov_session_bnr))

    for k in auc_a:
        print(
            f"{k}- 1st: {auc_a.get(k, -1)}, not-remembered: {auc_b_not_remembered.get(k, -1)}, remembered: {auc_b_remembered.get(k, -1)}")

    plt.bar(['1st', 'remembered', 'not remembered'],
            [sum(auc_a.values()) / float(len(auc_a)), sum(auc_b_remembered.values()) / float(len(auc_b_remembered)),
             sum(auc_b_not_remembered.values()) / float(len(auc_b_not_remembered))])
    plt.xlabel("Sessions")
    plt.ylabel("Avg AUC")
    plt.show()

    df = pd.DataFrame(columns=['Session', 'Subject', 'Movie', 'Distance'])
    for (subj_key, mov_key), value in auc_a.items():
        new_row_df = pd.DataFrame({'Session': ['A'], 'Subject': [subj_key], 'Movie': [mov_key], 'Distance': [value]})
        df = pd.concat([df, new_row_df], ignore_index=True, axis=0)
    for (subj_key, mov_key), value in auc_b_not_remembered.items():
        new_row_df = pd.DataFrame({'Session': ['BNR'], 'Subject': [subj_key], 'Movie': [mov_key], 'Distance': [value]})
        df = pd.concat([df, new_row_df], ignore_index=True, axis=0)
    for (subj_key, mov_key), value in auc_b_remembered.items():
        new_row_df = pd.DataFrame({'Session': ['BR'], 'Subject': [subj_key], 'Movie': [mov_key], 'Distance': [value]})
        df = pd.concat([df, new_row_df], ignore_index=True, axis=0)

    # dfg_all = df.groupby(['Session'])['Distance'].mean().reset_index()
    # sns.set(style="whitegrid")
    # sns.barplot(x="Session", y="Distance", data=dfg_all, capsize=.1, ci="sd")
    # plt.show()

    fig, ax = plt.subplots()
    dfg_subj = df.groupby(['Session', 'Subject'])['Distance'].mean().reset_index()
    sns.set(style="whitegrid")
    sns.barplot(x="Session", y="Distance", data=dfg_subj, capsize=.1, ci="sd")
    sns.swarmplot(x="Session", y="Distance", data=dfg_subj, color="0", alpha=.35)
    plt.show()

    # dfg_mov = df.groupby(['Session', 'Movie'])['Distance'].mean().reset_index()
    # sns.set(style="whitegrid")
    # sns.barplot(x="Session", y="Distance", data=dfg_mov, capsize=.1, ci="sd")
    # sns.swarmplot(x="Session", y="Distance", data=dfg_mov, color="0", alpha=.35)
    # plt.show()

    dfg_subj_A = dfg_subj.query("Session == 'A'")['Distance']
    plt.hist(dfg_subj_A, alpha=0.5, label='A', bins=100, range=(0, 800))
    dfg_subj_BNR = dfg_subj.query("Session == 'BNR'")['Distance']
    plt.hist(dfg_subj_BNR, alpha=0.5, label='BNR', bins=100, range=(0, 800))
    dfg_subj_BR = dfg_subj.query("Session == 'BR'")['Distance']
    plt.hist(dfg_subj_BR, alpha=0.5, label='BR', bins=100, range=(0, 800))
    plt.show()


auc_fixation_distance()
