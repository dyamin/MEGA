import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config as global_config
from src.signal_processing import config
from src.signal_processing.utils import get_event_median_times, get_all_subject_data_df

videos_dims = pd.read_pickle(global_config.VIDEO_DIMS_FILE_PATH)[:global_config.last_repeating_movie_ind]
event_times = get_event_median_times()
all_df = get_all_subject_data_df()
auc_a, auc_br, auc_bf = dict(), dict(), dict()
subjects = set([i[0] for i in all_df.index.values])
for subj in subjects:
    subj_df = all_df.xs(subj, level='Subject')
    session_a = subj_df.xs('Session A', level='Session')
    session_b = subj_df.query("Session == 'Session B'")
    session_b.index = session_b.index.droplevel(global_config.SESSION)
    session_b_remembered = session_b.query("Memory > 0")
    session_b_remembered.index = session_b_remembered.index.droplevel(global_config.MEMORY)
    session_b_forget = session_b.query("Memory < 0")
    session_b_forget.index = session_b_forget.index.droplevel(global_config.MEMORY)

    for mov in config.valid_movies:
        event_time = event_times[mov]
        mov_session_a = session_a.query(f"Movie == '{mov}'")
        mov_session_a = mov_session_a[mov_session_a.index.get_level_values(global_config.TIMESTAMP) < event_time]
        mov_session_br = session_b_remembered.query(f"Movie == '{mov}'")
        mov_session_br = mov_session_br.loc[mov_session_br.index.get_level_values(global_config.TIMESTAMP) < event_time]
        mov_session_bf = session_b_forget.query(f"Movie == '{mov}'")
        mov_session_bf = mov_session_bf.loc[mov_session_bf.index.get_level_values(global_config.TIMESTAMP) < event_time]

        movie_width, movie_height = videos_dims.loc[mov, ['Width', 'Height']]
        mov_roi_distance = max(movie_width, movie_height) * 0.15
        a_len = len(mov_session_a)
        if a_len > 0:
            roi_session_a = mov_session_a[mov_session_a[global_config.DISTANCE] < mov_roi_distance]
            auc_a[(subj, mov)] = (len(roi_session_a) / a_len) * 100

        br_len = len(mov_session_br)
        if br_len > 0:
            roi_session_br = mov_session_br[mov_session_br[global_config.DISTANCE] < mov_roi_distance]
            auc_br[(subj, mov)] = (len(roi_session_br) / br_len) * 100

        bf_len = len(mov_session_bf)
        if bf_len > 0:
            roi_session_bf = mov_session_bf[mov_session_bf[global_config.DISTANCE] < mov_roi_distance]
            auc_bf[(subj, mov)] = (len(roi_session_bf) / bf_len) * 100

for k in auc_a:
    print(
        f"{k}- 1st: {auc_a.get(k, -1)}, not-remembered: {auc_bf.get(k, -1)}, remembered: {auc_br.get(k, -1)}")

df = pd.DataFrame(columns=['Session', 'Subject', 'Movie', 'Proportion of time in AOI'])
for (subj_key, mov_key), value in auc_a.items():
    new_row_df = pd.DataFrame(
        {'Session': ['A'], 'Subject': [subj_key], 'Movie': [mov_key], 'Proportion of time in AOI': [value]})
    df = pd.concat([df, new_row_df], ignore_index=True, axis=0)
for (subj_key, mov_key), value in auc_bf.items():
    new_row_df = pd.DataFrame(
        {'Session': ['BF'], 'Subject': [subj_key], 'Movie': [mov_key], 'Proportion of time in AOI': [value]})
    df = pd.concat([df, new_row_df], ignore_index=True, axis=0)
for (subj_key, mov_key), value in auc_br.items():
    new_row_df = pd.DataFrame(
        {'Session': ['BR'], 'Subject': [subj_key], 'Movie': [mov_key], 'Proportion of time in AOI': [value]})
    df = pd.concat([df, new_row_df], ignore_index=True, axis=0)

dfg_mean = df.groupby(['Session', 'Subject'])['Proportion of time in AOI'].mean().reset_index()
fig, ax = plt.subplots()
sns.set(style="whitegrid")
sns.barplot(x="Session", y="Proportion of time in AOI", data=dfg_mean, capsize=.1, ci="sd")
sns.swarmplot(x="Session", y="Proportion of time in AOI", data=dfg_mean, hue="Subject")
plt.legend().remove()
plt.show()
