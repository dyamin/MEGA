from src.statistical_analysis import utils

rois = utils.get_aggregated_roi_df()
t_st_dev_mean = rois.t_StDev.mean()
time_filtered = rois.loc[rois['t_StDev'] < t_st_dev_mean]
x_std_mean = rois.X_StDev.mean()
gaze_filtered = time_filtered.loc[time_filtered['X_StDev'] < x_std_mean]
y_std_mean = rois.Y_StDev.mean()
gaze_filtered = gaze_filtered.loc[gaze_filtered['Y_StDev'] < y_std_mean]
print(list(gaze_filtered.index.values))
