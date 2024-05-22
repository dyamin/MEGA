from src.statistical_analysis import config, utils
from src.statistical_analysis.services.PlottingService import plot_nap_vs_no_nap, plot_elderly_vs_mci_ad, \
    plot_elderly_mci_ad_correlation


def run(args):
    rois = utils.get_aggregated_roi_df()
    labels = config.memory_comparison_labels

    if args[0] == 'all':
        plot_nap_vs_no_nap('rememberedAveragedDva', config.averaged_dva_suptitle,
                           config.default_na_nap_vs_nap_suptitle_filename, [labels['AR'], labels['BR']],
                           'Sessions (reported remembered in 2nd viewing)')
        plot_nap_vs_no_nap('forgotAveragedDva', config.averaged_dva_suptitle,
                           config.default_na_nap_vs_nap_suptitle_filename, [labels['AF'], labels['BF']],
                           'Sessions (reported forgotten in 2nd viewing)')
        plot_elderly_vs_mci_ad('rememberedAveragedDva', config.averaged_dva_suptitle,
                               config.default_na_nap_vs_nap_suptitle_filename, [labels['AR'], labels['BR']],
                               'Sessions (reported remembered in 2nd viewing)')
        plot_elderly_vs_mci_ad('forgotAveragedDva', config.averaged_dva_suptitle,
                               config.default_na_nap_vs_nap_suptitle_filename, [labels['AF'], labels['BF']],
                               'Sessions (reported forgotten in 2nd viewing)')
        plot_elderly_vs_mci_ad(['rememberedAveragedDva', 'forgotAveragedDva'], config.averaged_dva_suptitle,
                               config.default_averaged_dva_suptitle_filename, ['1st viewing', '2nd viewing'],
                               'Sessions')
    elif args[0] == 'nap_vs_no_nap':
        plot_nap_vs_no_nap(['rememberedAveragedDva', 'forgotAveragedDva'], config.averaged_dva_suptitle,
                           config.default_averaged_dva_suptitle_filename, ['1st viewing', '2nd viewing'],
                           'All trials')
    elif args[0] == 'remembered_nap_vs_no_nap':
        plot_nap_vs_no_nap('rememberedAveragedDva', config.averaged_dva_suptitle,
                           config.default_averaged_dva_suptitle_filename, ['1st viewing', '2nd viewing'],
                           'Remembered trials')
    elif args[0] == 'forgot_nap_vs_no_nap':
        plot_nap_vs_no_nap('forgotAveragedDva', config.averaged_dva_suptitle,
                           config.default_averaged_dva_suptitle_filename, ['1st viewing', '2nd viewing'],
                           'Not-Remembered trials')
    elif args[0] == 'remembered_elderly_vs_mci_ad':
        plot_elderly_vs_mci_ad('rememberedAveragedDva', config.averaged_dva_suptitle,
                               config.default_averaged_dva_suptitle_filename, [labels['AR'], labels['BR']],
                               'Sessions (reported remembered in 2nd viewing)')
    elif args[0] == 'forgot_elderly_vs_mci_ad':
        plot_elderly_vs_mci_ad('forgotAveragedDva', config.averaged_dva_suptitle,
                               config.default_averaged_dva_suptitle_filename, [labels['AF'], labels['BF']],
                               'Sessions (reported forgotten in 2nd viewing)')
    elif args[0] == 'elderly_vs_mci_ad':
        plot_elderly_vs_mci_ad(['rememberedAveragedDva', 'forgotAveragedDva'], config.averaged_dva_suptitle,
                               config.default_averaged_dva_suptitle_filename, ['1st viewing', '2nd viewing'],
                               'Sessions')
    elif args[0] == 'elderly_mci_ad_correlation':
        plot_elderly_mci_ad_correlation(['rememberedAveragedDva', 'forgotAveragedDva'], config.averaged_dva_suptitle,
                                        config.default_averaged_dva_suptitle_filename,
                                        x_label='MMSE')
        plot_elderly_mci_ad_correlation(['rememberedAveragedDva', 'forgotAveragedDva'], config.averaged_dva_suptitle,
                                        config.default_averaged_dva_suptitle_filename,
                                        x_label='MOCA')
        plot_elderly_mci_ad_correlation(['rememberedAveragedDva', 'forgotAveragedDva'], config.averaged_dva_suptitle,
                                        config.default_averaged_dva_suptitle_filename,
                                        x_label='HitRate')
    else:
        AttributeError(f"Plotting distances by {args[1]} is not supported.")
    return


run(["nap_vs_no_nap"])
