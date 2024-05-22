import os

# RoI
columns_names = ['Movie', 'X', 'Y', 't']
stats_median = ['X_median', 'Y_median', 't_median']
stats_mean = ['X_mean', 'Y_mean', 't_mean']

# Validity
inbounds_threshold = 0.7  # 70% of the subject are inside
t_std_window = 2
t = 't'  # from rois_per_subject
validity_filename = "validity_metrics.pkl"
rect9_column = "Rectangle 9 (% subjects)"
rect16_column = "Rectangle 16 (% subjects)"
time_column = "Time (% subjects)"

# Log
log_dir = os.path.join("resources", "log", "roibased")
