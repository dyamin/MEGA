FIXATIONNUMBER = 'FixationNumber'
BLINKNUMBER = 'BlinkNumber'
SACCADENUMBER = 'SaccadeNumber'
EVENT_TIME = 'EventTime'

# Pre-Processing Arguments:
epsilon = 0.05
num_arguments_for_numerical_derivation = 3
save_each_subject = True
verbose_each_subject = True
should_remove_initial_pupil_per_movie = False
use_eyelink_parser = True

min_blink_time = 3
ignore_blinks_in_movie_start_and_end = True
blink_speed_stdev_threshold = 2
min_samples_per_blink_epoch = 3
samples_between_blink_epochs = 2

should_mark_fixations_on_original_data = True
engberts_lambda = 5
max_difference_within_fixation_epoch = 6
min_fixation_duration = 50

max_difference_within_saccade = 6
min_saccade_duration = 6

specific_subjects = []
