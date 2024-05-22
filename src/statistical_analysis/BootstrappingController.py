from src.statistical_analysis.services.DistanceBootstrappingService import generate_session_B_memory_distribution, \
    generate_boostrap_distribution, generate_dva_between_session_distribution, generate_between_session_distribution


def run(args):
    if len(args) == 1 and (args[0] == 'DVA_Sessions'):
        return generate_dva_between_session_distribution()
    if len(args) == 1 and (args[0] == 'Sessions'):
        return generate_between_session_distribution()
    elif len(args) == 1 and (args[0] == 'Memory'):
        return generate_session_B_memory_distribution()
    assert (len(args) == 9), f'Not enough arguments to calculate a new bootstrapping distribution.'
    partition_subtract_from = [args[0], args[1], args[2]]
    partition_to_subtract = [args[3], args[4], args[5]]
    start_time, end_time = args[6], args[7]
    bootstrap_name = args[8]
    return generate_boostrap_distribution(partition_subtract_from, partition_to_subtract,
                                          start_time, end_time, bootstrap_name)


run(['DVA_Sessions'])
