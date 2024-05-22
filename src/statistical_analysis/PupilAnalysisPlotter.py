from src.statistical_analysis import config, utils
from src.statistical_analysis.services.PlottingService import plot_memory, plot_averaged_sessions, plot_memory_within, \
    plot_sessions


def run(args):
    rois = utils.get_aggregated_roi_df()

    if args[0] == 'all':
        plot_memory_within(rois, 'pupil_memory_within')
        plot_sessions(rois, 'pupil_session')
        plot_memory(rois, 'pupil_memory')
        plot_averaged_sessions('rememberedAveragedPupil', 'forgotAveragedPupil', config.averaged_pupil_suptitle,
                               config.default_averaged_pupil_suptitle_filename, 'Sessions')

    elif args[0] == 'pupil_session':
        plot_sessions(rois, 'pupil_session')
    elif args[0] == 'pupil_memory':
        plot_memory(rois, 'pupil_memory')
    elif args[0] == 'pupil_memory_within':
        plot_memory_within(rois, 'pupil_memory_within')
    elif args[0] == 'average_pupil_from_beginning':
        plot_averaged_sessions('rememberedAveragedPupil', 'forgotAveragedPupil', config.averaged_pupil_suptitle,
                               config.default_averaged_pupil_suptitle_filename)
    else:
        AttributeError(f"Plotting distances by {args[1]} is not supported.")
    return


run(["all"])
