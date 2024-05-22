import os

import pandas as pd

import src.config as g_config
from src.statistical_analysis import config, utils
from src.statistical_analysis.services.PlottingService import plot_optimal_starting_time, \
    plot_averaged_sessions, plot_memory_within, plot_sessions, plot_memory, plot_memory_with_sem_shade, \
    plot_bootstrap_dist


def run(args):
    rois = utils.get_aggregated_roi_df()
    labels = config.memory_comparison_labels

    if args[0] == 'all':
        plot_memory_within(rois, 'dva_memory_within')
        plot_memory_with_sem_shade(rois, 'dva_memory_within')
        plot_averaged_sessions('rememberedAoiDuration', config.aoi_duration_suptitle,
                               config.default_aoi_duration_suptitle_filename, [labels['AR'], labels['BR']],
                               'Sessions (reported remembered in 2nd viewing)')
        plot_averaged_sessions('forgotAoiDuration', config.aoi_duration_suptitle,
                               config.default_aoi_duration_suptitle_filename, [labels['AF'], labels['BF']],
                               'Sessions (reported remembered in 2nd viewing)')
        plot_averaged_sessions('rememberedAveragedDva', config.averaged_dva_suptitle,
                               config.default_averaged_dva_suptitle_filename, [labels['AR'], labels['BR']],
                               'Sessions (reported remembered in 2nd viewing)')
        plot_averaged_sessions('forgotAveragedDva', config.averaged_dva_suptitle,
                               config.default_averaged_dva_suptitle_filename, [labels['AF'], labels['BF']],
                               'Sessions (reported forgotten in 2nd viewing)')
        plot_averaged_sessions('BRemembered_BForgot', config.averaged_dva_suptitle,
                               config.default_averaged_dva_suptitle_filename, [labels['BR'], labels['BF']],
                               '2nd viewing')
        plot_sessions(rois, 'dva_session')
        plot_memory(rois, 'dva_memory')
        # plot_optimal_starting_time('optimal_general_prefix_cut')
        # Features:
        # df = pd.read_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
        #                                  f"features.pkl"))
        # # Get all column names except the last one (which is the Confidence column)
        # features = df.columns[:-1]
        # for feature in features:
        #     plot_averaged_sessions('features', feature,
        #                        'features', ['1st viewing', '2nd viewing'],
        #                        'Sessions', feature=feature)

    elif args[0] == 'dva_session':
        plot_sessions(rois, 'dva_session')
    elif args[0] == 'dva_memory':
        plot_memory(rois, 'dva_memory')
    elif args[0] == 'dva_memory_within':
        plot_memory_within(rois, 'dva_memory_within')
    elif args[0] == 'dva_memory_with_sem_shade':
        plot_memory_with_sem_shade(rois, 'dva_memory_within')
    elif args[0] == 'aoi_duration_remembered':
        plot_averaged_sessions('rememberedAoiDuration', config.aoi_duration_suptitle,
                               config.default_aoi_duration_suptitle_filename, [labels['AR'], labels['BR']],
                               'Sessions (reported remembered in 2nd viewing)')
    elif args[0] == 'aoi_duration_forgot':
        plot_averaged_sessions('forgotAoiDuration', config.aoi_duration_suptitle,
                               config.default_aoi_duration_suptitle_filename, [labels['AF'], labels['BF']],
                               'Sessions (reported remembered in 2nd viewing)')
    elif args[0] == 'average_remembered':
        plot_averaged_sessions('rememberedAveragedDva', config.averaged_dva_suptitle,
                               config.default_averaged_dva_suptitle_filename, [labels['AR'], labels['BR']],
                               'Sessions (reported remembered in 2nd viewing)', feature='DVA')
    elif args[0] == 'average_dva':
        plot_averaged_sessions(['rememberedAveragedDva', 'forgotAveragedDva'], config.averaged_dva_suptitle,
                               config.default_averaged_dva_suptitle_filename, ['1st viewing', '2nd viewing'],
                               'Sessions', feature=g_config.DVA)
    elif args[0] == 'average_forgot':
        plot_averaged_sessions('forgotAveragedDva', config.averaged_dva_suptitle,
                               config.default_averaged_dva_suptitle_filename, [labels['AF'], labels['BF']],
                               'Sessions (reported forgotten in 2nd viewing)', feature='DVA')
    elif args[0] == 'BRemembered_BForgot':
        plot_averaged_sessions('BRemembered_BForgot', config.averaged_dva_suptitle,
                               config.default_averaged_dva_suptitle_filename, [labels['BR'], labels['BF']],
                               '2nd viewing', feature=g_config.DVA)
    elif args[0] == 'dva_bootstrap':
        plot_bootstrap_dist(['rememberedAveragedDva', 'forgotAveragedDva'], config.averaged_dva_suptitle, args[0],
                            'Sessions')
    elif args[0] == 'features':
        df = pd.read_pickle(os.path.join(g_config.statistical_analysis_resource_dir,
                                         f"features.pkl"))
        # Get all column names except the last one (which is the Confidence column)
        features = ['Saccades_Duration_Median', 'Confidence']  # df.columns[:-1]
        for feature in features:
            plot_averaged_sessions('features', 'Saccades_Duration_Median',  # feature,
                                   'features', ['1st viewing', '2nd viewing'],
                                   'Sessions', feature=features)
    elif args[0] == 'optimal_general_prefix_cut':
        plot_optimal_starting_time(args[0])
    else:
        AttributeError(f"Plotting {args[1]} is not supported.")
    return


run(["average_remembered"])
