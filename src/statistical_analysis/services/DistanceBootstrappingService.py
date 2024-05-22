import datetime as datetime
import os

import pandas as pd

from src import config as g_config
from src.statistical_analysis import config
from src.statistical_analysis.business.DifferenceCalculatorLogic import calculate_differences
from src.statistical_analysis.business.DistanceBootstrappingLogic import gen_bootstrap, get_bootstrapping_agg_func
from src.statistical_analysis.business.DistanceFocusingLogic import recenter_distances
from src.statistical_analysis.business.PartitionGenerationLogic import gen_partitioned_data
from src.utils import round_down_to_closest_even, save_df_to_pkl


def generate_dva_between_session_distribution():
    print('Initiating Session bootstrapping calculations...')
    session_bootstrap, stat_value, p_value = generate_dva_boostrap_distribution(
        ['A', '', ''], ['B', '', ''],
        bootstrap_name='Bootstrap_Session')
    return session_bootstrap, stat_value, p_value


def generate_between_session_distribution():
    print('Initiating Session bootstrapping calculations...')
    session_bootstrap, stat_value, p_value = generate_dva_boostrap_distribution(
        ['A', '', ''], ['B', '', ''],
        bootstrap_name='Bootstrap_Session',
        start_time=config.bootstrap_start_time,
        end_time=config.bootstrap_end_time)
    return session_bootstrap, stat_value, p_value


def generate_session_B_memory_distribution():
    print('Initiating Memory within Session B bootstrapping calculations...')
    session_b_memory_bootstrap, stat_value, p_value = generate_boostrap_distribution(
        ['B', 'N', ''], ['B', 'Y', ''],
        bootstrap_name='Bootstrap_Memory_in_Session_B',
        start_time=config.bootstrap_start_time,
        end_time=config.bootstrap_end_time)
    return session_b_memory_bootstrap, stat_value, p_value


def generate_boostrap_distribution(partition_subtract_from: list, partition_to_subtract: list,
                                   start_time: float, end_time: float, bootstrap_name: str):
    start_time = None if start_time is None else round_down_to_closest_even(start_time)
    end_time = None if end_time is None else round_down_to_closest_even(end_time)

    rois = pd.read_pickle(os.path.join(g_config.rois_dir, g_config.AGGRGATED_ROI_FILE))
    all_gaze_data = pd.read_pickle(os.path.join(g_config.data_dir, g_config.RAW_GAZE_FILE))
    distances = recenter_distances(all_gaze_data[g_config.DISTANCE], rois, False)

    subtract_from = gen_partitioned_data(distances, partition_subtract_from)
    subtractor = gen_partitioned_data(distances, partition_to_subtract)
    differences = calculate_differences(subtract_from=subtract_from, subtractor=subtractor,
                                        from_timestamp=start_time, to_timestamp=end_time)

    bootstrapped_dist = gen_bootstrap(differences,
                                      aggregator=config.bootstrapping_aggregator,
                                      levels=config.bootstrapping_levels)
    stat_value, p_value = _calculate_stat_and_p_value(differences, bootstrapped_dist)

    _save_bootstrapping_distribution(bootstrapped_dist, bootstrap_name)
    return bootstrapped_dist, stat_value, p_value


def _calculate_stat_and_p_value(data: pd.Series, bootstrap_dist: pd.Series) -> (float, float):
    agg_func = get_bootstrapping_agg_func(config.bootstrapping_aggregator)
    stat_value = agg_func(data.groupby(level=[config.TIMESTAMP]).mean())
    p_value = 1 - (bootstrap_dist.searchsorted(stat_value) / len(bootstrap_dist))
    print(f"\tStat Value:\t{stat_value:.2f}\n\tSingle sided p-value:\t{p_value:.4f}")
    return stat_value, p_value


def _save_bootstrapping_distribution(bootstrap_dist: pd.Series, bootstrap_name: str) -> None:
    filename = f'{bootstrap_name}_{datetime.now().strftime("%d/%m/%Y_%H:%M:%S")}'
    save_df_to_pkl(bootstrap_dist, filename, config.data_dir, config.pickling_protocol)
    return
