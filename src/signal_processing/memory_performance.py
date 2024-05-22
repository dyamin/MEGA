"""
Calculate hit_rate for each subject in the memory_self_report DataFrame.
hit rate = number of hits / number of targets

Save the hit_rate values in a pkl file.
"""

import os

import pandas as pd

from src import config
from src.config import get_project_root

# Load the memory report DataFrame
memory_report = pd.read_pickle(os.path.join(config.data_dir, 'raw_memory_report.pkl'))
# Load the valid movies without the prefix 'mov' (e.g. 'mov1' -> 1)
valid_movies_by_number = [int(movie[3:]) for movie in config.valid_movies]
# Filter and keep only the valid movies (i.e. movies that have RoIs)
memory_report = memory_report[memory_report.index.get_level_values('Movie #').isin(valid_movies_by_number)]

# Calculate dprime for each subject, index name will be Subject
hit_rate_df = pd.Series(dtype=float, name='HitRate',
                        index=memory_report.index.get_level_values(config.SUBJECT).unique())

for subject in memory_report.index.get_level_values(config.SUBJECT).unique():
    # Get the memory report for the given subject
    subject_memory_report = memory_report[memory_report.index.get_level_values(config.SUBJECT) == subject]

    # Get the memory report for session B
    session_b_memory_report = subject_memory_report['Seen? B']

    # Get the number of targets by counting the number of rows
    num_targets = session_b_memory_report.count()

    # Get the number of hits by summing the rows that have a value of 1 in the 'Seen? B' column
    num_hits = session_b_memory_report.sum()

    # Calculate hit rate and false alarm rate
    hit_rate = num_hits / num_targets

    hit_rate_df[subject] = hit_rate

# Save the dprime values in a pkl file
hit_rate_df.to_pickle(os.path.join(config.statistical_analysis_resource_dir, 'hit_rate.pkl'))

# Read memory performance csv
memory_performance = pd.read_csv(os.path.join(config.memory_performance_csv))

# Add Hit rate column to memory performance by join the subject-
# column in the memory performance df, and index in the hit rate series
hit_rate_elderly = pd.read_pickle(
    os.path.join(get_project_root(), "resources", "elderly", "statistical_analysis", 'hit_rate.pkl'))
hit_rate_mci_ad = pd.read_pickle(
    os.path.join(get_project_root(), "resources", "mci_ad", "statistical_analysis", 'hit_rate.pkl'))
hit_rate_series = pd.concat([hit_rate_elderly, hit_rate_mci_ad])

memory_performance = memory_performance.join(hit_rate_series, on='Name')

# Save the memory performance df as pkl
memory_performance.to_pickle(
    os.path.join(get_project_root(), "resources", 'elderly', "statistical_analysis", "memory_performance.pkl"))
