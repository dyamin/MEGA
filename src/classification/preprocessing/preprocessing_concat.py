import os

import pandas as pd

from src import config

if config.POPULATION == 'young_multiple_populations':
    animation_features_df = pd.read_pickle(
        os.path.join(config.get_project_root(), "resources", 'animation', "statistical_analysis", 'features.pkl'))
    animation_features_df.index = animation_features_df.index.set_levels(
        [f'animation_{x}' for x in animation_features_df.index.levels[0]], level=0)
    no_nap_features_df = pd.read_pickle(
        os.path.join(config.get_project_root(), "resources", 'no_nap', "statistical_analysis", 'features.pkl'))
    no_nap_features_df.index = no_nap_features_df.index.set_levels(
        [f'no_nap_{x}' for x in no_nap_features_df.index.levels[0]], level=0)
    nap_features_df = pd.read_pickle(
        os.path.join(config.get_project_root(), "resources", 'nap', "statistical_analysis", 'features.pkl'))
    nap_features_df.index = nap_features_df.index.set_levels([f'nap_{x}' for x in nap_features_df.index.levels[0]],
                                                             level=0)

    # Get common columns
    common_columns = animation_features_df.columns.intersection(no_nap_features_df.columns).intersection(
        nap_features_df.columns)
    # print columns name in no_nap_features_df that are not in animation_features_df
    print(no_nap_features_df.columns.difference(animation_features_df.columns))

    # Select common columns in both dataframes
    animation_features_df_common = animation_features_df[common_columns]
    no_nap_features_df_common = no_nap_features_df[common_columns]
    nap_features_df_common = nap_features_df[common_columns]

    # Concatenate
    multiple_populations_features_df = pd.concat(
        [animation_features_df_common, no_nap_features_df_common, nap_features_df_common])

    multiple_populations_features_df.to_pickle(
        os.path.join(config.get_project_root(), "resources", 'multiple_populations', "statistical_analysis",
                     'features.pkl'))

elif config.POPULATION == 'elderly_multiple_populations':
    elderly_features_df = pd.read_pickle(
        os.path.join(config.get_project_root(), "resources", 'elderly', "statistical_analysis", 'features.pkl'))
    elderly_features_df.index = elderly_features_df.index.set_levels(
        [f'elderly_{x}' for x in elderly_features_df.index.levels[0]], level=0)
    mci_ad_features_df = pd.read_pickle(
        os.path.join(config.get_project_root(), "resources", 'mci_ad', "statistical_analysis", 'features.pkl'))
    mci_ad_features_df.index = mci_ad_features_df.index.set_levels(
        [f'mci_ad_{x}' for x in mci_ad_features_df.index.levels[0]], level=0)

    # Get common columns
    common_columns = elderly_features_df.columns.intersection(mci_ad_features_df.columns)
    # print columns name in no_nap_features_df that are not in animation_features_df
    print(mci_ad_features_df.columns.difference(elderly_features_df.columns))

    # Select common columns in both dataframes
    elderly_features_df_common = elderly_features_df[common_columns]
    mci_ad_features_df_common = mci_ad_features_df[common_columns]

    # Concatenate
    multiple_populations_features_df = pd.concat([elderly_features_df_common, mci_ad_features_df_common])

    multiple_populations_features_df.to_pickle(
        os.path.join(config.get_project_root(), "resources", 'multiple_populations', "statistical_analysis",
                     'features.pkl'))
