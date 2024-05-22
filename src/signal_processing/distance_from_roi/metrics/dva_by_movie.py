import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config as global_config
from src.signal_processing import config, unit_transformers
from src.signal_processing.utils import get_aggregated_roi_df
from src.signal_processing.utils import get_event_median_times, get_all_subject_data_df


def auc_fixation_distance():
    event_times = get_event_median_times()
    rois = get_aggregated_roi_df()

    all_df = get_all_subject_data_df()
    auc_a, auc_b_remembered, auc_b_not_remembered = dict(), dict(), dict()
    subjects = set([i[0] for i in all_df.index.values])
    for subj in subjects:
        subj_df = all_df.xs(subj, level='Subject')
        session_a = subj_df.xs('Session A', level='Session')
        # session_a = session_a[session_a.index.isin([-4, -3, -2, -1], level='Memory')]
        session_b = subj_df.query("Session == 'Session B'")
        session_b.index = session_b.index.droplevel(0)
        session_b_remembered = session_b[session_b.index.isin([4, 3, 2, 1], level='Memory')]
        session_b_not_remembered = session_b[session_b.index.isin([-4, -3, -2, -1], level='Memory')]

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

            x_roi, y_roi = get_movie_roi(mov, rois)

            for index, row in mov_session_a.iterrows():
                dva = unit_transformers.calculate_dva(row['X_gaze'], row['Y_gaze'], (x_roi, y_roi))
                auc_a[(subj, mov)] = auc_a.get((subj, mov), 0) \
                                     + (dva / len(mov_session_a))
            for index, row in mov_session_br.iterrows():
                dva = unit_transformers.calculate_dva(row['X_gaze'], row['Y_gaze'], (x_roi, y_roi))
                auc_b_remembered[(subj, mov)] = auc_b_remembered.get((subj, mov), 0) \
                                                + (dva / len(mov_session_br))
            for index, row in mov_session_bnr.iterrows():
                dva = unit_transformers.calculate_dva(row['X_gaze'], row['Y_gaze'], (x_roi, y_roi))
                auc_b_not_remembered[(subj, mov)] = auc_b_not_remembered.get((subj, mov), 0) \
                                                    + (dva / len(mov_session_bnr))

    for k in auc_a:
        print(
            f"{k}- 1st: {auc_a.get(k, -1)}, not-remembered: {auc_b_not_remembered.get(k, -1)}, remembered: {auc_b_remembered.get(k, -1)}")

    df = pd.DataFrame(columns=['Session', 'Subject', 'Movie', 'DVA'])
    for (subj_key, mov_key), value in auc_a.items():
        new_row_df = pd.DataFrame({'Session': ['A'], 'Subject': [subj_key], 'Movie': [mov_key], 'DVA': [value]})
        df = pd.concat([df, new_row_df], ignore_index=True, axis=0)
    for (subj_key, mov_key), value in auc_b_not_remembered.items():
        new_row_df = pd.DataFrame({'Session': ['BNR'], 'Subject': [subj_key], 'Movie': [mov_key], 'DVA': [value]})
        df = pd.concat([df, new_row_df], ignore_index=True, axis=0)
    for (subj_key, mov_key), value in auc_b_remembered.items():
        new_row_df = pd.DataFrame({'Session': ['BR'], 'Subject': [subj_key], 'Movie': [mov_key], 'DVA': [value]})
        df = pd.concat([df, new_row_df], ignore_index=True, axis=0)

    # dfg_mov = df.groupby(['Session', 'Movie'])['Distance'].mean().reset_index()
    # sns.set(style="whitegrid")
    # sns.barplot(x_roi="Session", y_roi="Distance", data=dfg_mov, capsize=.1, ci="sd")
    # sns.swarmplot(x_roi="Session", y_roi="Distance", data=dfg_mov, hue="Movie")
    # plt.legend().remove()
    # plt.show()

    dfg_subj = df.groupby(['Session', 'Subject'])['DVA'].mean().reset_index()
    sns.set(style="whitegrid")
    sns.barplot(x="Session", y="DVA", data=dfg_subj, capsize=.1, ci="sd")
    sns.swarmplot(x="Session", y="DVA", data=dfg_subj, hue="Subject")

    plt.legend().remove()
    plt.show()


def get_movie_roi(mov, rois):
    mov_roi = rois[rois.index.isin([mov])]
    videos_dims = pd.read_pickle(global_config.VIDEO_DIMS_FILE_PATH)[:global_config.last_repeating_movie_ind]
    movie_width, movie_height = videos_dims.loc[mov, ['Width', 'Height']]
    x_roi = (mov_roi['X_mean'] / 100) * movie_height
    y_roi = (mov_roi['Y_mean'] / 100) * movie_width
    return x_roi, y_roi


auc_fixation_distance()
