import utils
from src import config as g_config
from src.statistical_analysis.business.DistanceFocusingLogic import adjust_pupil_data
from src.statistical_analysis.services.PrePlottingService import sessions, remembered_within_subject, \
    aoi_duration_from_beginning, average_per_trial, memory, find_optimal_params, find_best_movies, BRemember_BForgot


def run(args):
    roi, subjects_data = _retrieve_roi_and_subjects_data(args[0])

    # if g_config.POPULATION == 'nap':
    #     subjects_data = filter_to_subjects_that_nap(subjects_data)

    if args[1] == 'gaze' or args[1] == 'all':
        distance = subjects_data[[g_config.DVA, g_config.SQRT_DVA, g_config.DISTANCE]]

        if args[2] == 'all':
            average_per_trial(roi, distance, 'rememberedAveragedDva', 'forgotAveragedDva')
            aoi_duration_from_beginning(roi, distance, 'rememberedAoiDuration', 'forgotAoiDuration')
            sessions(roi, distance, 'dva_session')
            memory(roi, distance, 'dva_memory')
            remembered_within_subject(roi, distance, 'dva_memory_within')
            BRemember_BForgot(roi, distance, 'BRemembered_BForgot')
            # find_optimal_params(roibased, distance, 'optimal_general_prefix_cut')
            # find_best_movies(roibased, distance)

        elif args[2] == 'session':
            sessions(roi, distance, 'dva_session')
        elif args[2] == 'memory':
            memory(roi, distance, 'dva_memory')
        elif args[2] == 'memory_within':
            remembered_within_subject(roi, distance, 'dva_memory_within')
        elif args[2] == 'aoi_duration_from_beginning':
            aoi_duration_from_beginning(roi, distance, 'rememberedAoiDuration', 'forgotAoiDuration')
        elif args[2] == 'average_from_beginning':
            average_per_trial(roi, distance, 'rememberedAveragedDva', 'forgotAveragedDva')
        elif args[2] == 'BRemembered_BForgot':
            BRemember_BForgot(roi, distance, 'BRemembered_BForgot')
        elif args[2] == 'optimal_general_prefix_cut':
            find_optimal_params(roi, distance, args[2])
        elif args[2] == 'find_best_movies':
            find_best_movies(roi, distance)
        else:
            AttributeError(f"Plotting distances by {args[2]} is not supported.")

    if args[1] == 'pupil' or args[1] == 'all':
        pupil = subjects_data[[g_config.PUPIL]]
        pupil = adjust_pupil_data(pupil)

        if args[2] == 'all':
            average_per_trial(roi, pupil, 'rememberedAveragedPupil', 'forgotAveragedPupil')
            sessions(roi, pupil, 'pupil_session')
            memory(roi, pupil, 'pupil_memory')
            remembered_within_subject(roi, pupil, 'pupil_memory_within')
        elif args[2] == 'session':
            sessions(roi, pupil, 'pupil_session')
        elif args[2] == 'memory':
            memory(roi, pupil, 'pupil_memory')
        elif args[2] == 'memory_within':
            remembered_within_subject(roi, pupil, 'pupil_memory_within')
        elif args[2] == 'average_from_beginning':
            average_per_trial(roi, pupil, 'rememberedAveragedPupil', 'forgotAveragedPupil')
        else:
            AttributeError(f"Plotting distances by {args[2]} is not supported.")


def _retrieve_roi_and_subjects_data(requested_movie: str):
    rois = utils.get_aggregated_roi_df()
    aggregated_distances = utils.get_all_valid_subject_data_df()
    if requested_movie == 'all':
        return rois, aggregated_distances
    if requested_movie == 'valid':
        roi_drop_movies = set(rois.index) - set(g_config.valid_movies)
        distance_drop_movies = roi_drop_movies.union({
            f'mov{idx}' for idx in range(g_config.num_repeating_movies + 1, g_config.total_recorded_movies + 1)})
        valid_rois = rois.drop(roi_drop_movies)
        valid_distances = aggregated_distances.drop(index=distance_drop_movies, level=g_config.MOVIE, errors='ignore')
        return valid_rois, valid_distances
    assert (int(requested_movie) >= 1) and (int(requested_movie) <= g_config.num_repeating_movies), \
        f'Requested movie must be an int between 1 and {g_config.num_repeating_movies}, you provided {requested_movie}'
    movID = f'mov{int(requested_movie)}'
    mov_roi = rois.loc[movID]
    mov_distances = aggregated_distances.xs(movID, level='Movie',
                                            drop_level=False)  # returned Series has level 'Movie' in it
    return mov_roi, mov_distances


run(["valid", "gaze", "all"])
