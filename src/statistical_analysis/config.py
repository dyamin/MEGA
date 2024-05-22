# Plotting Constants:
FIG_SIZE = (20, 11)
SUP_TITLE_FONTSIZE, SUB_TITLE_FONTSIZE, AXIS_LABEL_FONTSIZE, LEGEND_FONTSIZE, TICKS_LABEL_FONTSIZE = 35, 18, 35, 20, 30
LIGHT_LINESTYLE, MEDIUM_LINESTYLE, DASHED_LINESTYLE = ':', "-.", "--"
EVENT_TIME_LINE_COLOR, EVENT_TIME_FACE_COLOR = '#668cff', '#99b3ff'
LINE_WIDTH_BOLD, LINE_WIDTH_MEDIUM, LINE_WIDTH_NARROW = 2.5, 1.7, 1
LINE_ALPHA_HIGH, LINE_ALPHA_MEDIUM, LINE_ALPHA_LOW = 0.8, 0.6, 0.3
SURFACE_ALPHA_HIGH, SURFACE_ALPHA_MEDIUM, SURFACE_ALPHA_LOW = 0.5, 0.2, 0.08
SESSION_A_COLOR1, SESSION_A_COLOR2, SESSION_B_COLOR1, SESSION_B_COLOR2 = '#33ff33', '#99ff99', '#ff3333', '#ff9999'
DEFAULT_SUPTITLE, DEFAULT_SUBTITLE = 'Averaging across all movies for each subject', ''
# DEFAULT_SUPTITLE, DEFAULT_SUBTITLE = 'Memory Performance', ''
# DEFAULT_SUPTITLE, DEFAULT_SUBTITLE = 'Explicitly remembered trials', ''
# DEFAULT_SUPTITLE, DEFAULT_SUBTITLE = 'Explicitly forgotten trials', ''
DEFAULT_X_LABEL, DEFAULT_Y_LABEL, Y_LABEL_DIFFERENCE = 'Time from Event (ms)', 'Distance (dva)', 'Distance Difference ' \
                                                                                                 '(dva)'
ADD_LEGENDS = True

# Plotting flags and defaults
should_add_error = False
should_zoom_on_high_sample_count = True
min_percent_samples = 70
DEFAULT_ROI_TIME, DEFAULT_ROI_DURATION = 0, 1500  # in ms, i.e 1.5sec
should_aggregate_by_subject = True
should_normalize = False
splitted_violin = False
STARTING_TIME = 0 # 1760
SHOULD_FILTER_SUBJECTS = False
should_plot_time_series_of_trial = True

# Plotting by Session
session_subtitle = 'Grouped by Session'
default_session_filename = 'Distances_by_Session'
default_aoi_duration_suptitle_filename = 'AOI_Duration_by_Session'
default_averaged_dva_suptitle_filename = 'Averaged_DVA_by_Session'
default_averaged_pupil_suptitle_filename = 'Averaged_pupil_by_Session'
default_na_nap_vs_nap_suptitle_filename = 'no_nap_vs_nap'

# Plotting by Session & Memory
should_plot_not_remembered = True
should_plot_session_A = True
memory_labels = ['2nd - Remembered', '2nd - Not Remembered',
                 '1st - Remembered', '1st - Not Remembered']
memory_subtitle = 'Grouped by Session & Memory'
default_memory_filename = 'Distances_by_Session_and_Memory'
aoi_duration_suptitle = 'Proportion of time inside AOI'
averaged_dva_suptitle = 'MEGA score'
# averaged_dva_suptitle = 'Mead distance to the ROI (dva)'
averaged_pupil_suptitle = 'Mean Pupil changes from the beginning'
ADD_SEM_SHADE = False

# Plotting Session Comparison for Remembered Subject
memory_comparison_labels = {'BR': '2nd viewing (reported remembered in 2nd viewing)',
                            'AR': '1st viewing (reported remembered in 2nd viewing)',
                            'BF': '2nd viewing (reported not remembered in 2nd viewing)',
                            'AF': '1st (reported not remembered in 2nd viewing)'}

default_memory_comparison_filename = 'Distances_by_Memory_between_Sessions'
memory_comparison_filename = ''  # 'Comparing Sessions for Memory Report in B'

# Bootstrapping Constants:
subtraction_nan_filler = None
NUM_ITERATIONS = 1000
verbose = True
verbose_every = 50
bootstrapping_aggregator = 'area'
bootstrapping_levels = []  # [MOVIE, SUBJECT] | [SUBJECT, MOVIE] | [SUBJECT] | [MOVIE]
bootstrap_start_time, bootstrap_end_time = -1500, 0
