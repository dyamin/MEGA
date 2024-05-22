from src.statistical_analysis.business.DataSplittingLogic import split
from src.statistical_analysis.models.DataPartition import DataPartition

'''
This module splits the data and returns only part of the data that matches the requested partition
'''


def gen_partitioned_data(data, partition_by: list):
    partition = DataPartition.make(partition_by)
    split_data = split(data, partition.get_splitter_arguments()[0])

    if partition.ses.lower() == 'A'.lower():
        split_data = split_data[:len(split_data) // 2]
    elif partition.ses.lower() == 'B'.lower():
        split_data = split_data[len(split_data) // 2:]

    if partition.mem.lower() == 'Y'.lower():
        split_data = split_data[:len(split_data) // 2]
    elif partition.mem.lower() == 'N'.lower():
        split_data = split_data[len(split_data) // 2:]

    if partition.conf.lower() == 'H'.lower():
        split_data = split_data[:len(split_data) // 2]
    elif partition.conf.lower() == 'L'.lower():
        split_data = split_data[len(split_data) // 2:]

    return split_data[0]
