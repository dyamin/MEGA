from src import config

# General
marked_videos_dir = r'marked'

# Flags
speed = 1
subjects_to_show = ['LG3'] # ['HR58'] # ['HB81']  # ['LG3']  # ['NP2']
num_subjects_if_random = 1  # number of random subjects to choose in case of no subjects mentioned
sessions_to_show = [config.SESSION_A, config.SESSION_B]
draw_roi = False
draw_rect = False
draw_fixations = True
draw_raw = True
draw_memory = False
draw_agd = True
draw_all_fixations = False

# RoI
roi_size = 9  # 9th of the screen
rois_rects_file = f'Rects_{roi_size}th.pkl'

# Fixations
fixations_file = 'all_subject_fixations.pkl'
fixation_X = config.gaze_X
fixation_Y = config.gaze_Y
fixation_duration = "duration"
fixation_t0 = "onset"
