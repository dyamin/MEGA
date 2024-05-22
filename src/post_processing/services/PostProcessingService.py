import os

import pandas as pd

from src import config
from src.post_processing import config as preprcessing_config
from src.post_processing.business.DemographicsAndMemoryReportsLogic import aggregate_data
from src.post_processing.services.SubjectProcessingService import process_subject
from src.utils import save_df_to_pkl


def process(memory_report_dir: str, gaze_dir: str):
    assert (os.path.isdir(memory_report_dir)), f'Couldn\'t find a directory in path {memory_report_dir}'
    assert (os.path.isdir(memory_report_dir)), f'Couldn\'t find a directory in path {gaze_dir}'
    all_subjects_demog, all_subjects_memory, memory_series = aggregate_data(memory_report_dir)

    gaze_dict, blinks_dict, fixations_dict, saccades_dict = dict(), dict(), dict(), dict()

    subj_dirs = [os.path.join(config.decentralized_data_dir, dirname) for dirname in
                 os.listdir(config.decentralized_data_dir)
                 if os.path.isdir(os.path.join(config.decentralized_data_dir, dirname))]

    if preprcessing_config.specific_subjects:
        filtered_subj = []
        for d in subj_dirs:
            if d.endswith(tuple(preprcessing_config.specific_subjects)):
                filtered_subj.append(d)
        subj_dirs = filtered_subj

    for subj_dir in subj_dirs:
        subject_id = os.path.basename(subj_dir)
        subject_memory = memory_series.xs(subject_id, level=config.SUBJECT)
        marked_blinks_data, blinks_df, fixations_df, saccades_df = process_subject(subj_dir, subject_memory,
                                                                                   preprcessing_config.should_remove_initial_pupil_per_movie,
                                                                                   preprcessing_config.save_each_subject,
                                                                                   preprcessing_config.verbose_each_subject)
        gaze_dict[subject_id] = marked_blinks_data
        blinks_dict[subject_id] = blinks_df
        fixations_dict[subject_id] = fixations_df
        saccades_dict[subject_id] = saccades_df

    # concat all single-subject DFs
    all_subjects_gaze_df = pd.concat(gaze_dict.values())
    all_subjects_blinks_df = pd.concat(blinks_dict.values())
    all_subjects_fixations_df = pd.concat(fixations_dict.values())
    all_subjects_saccades_df = pd.concat(saccades_dict.values())

    # save the results
    save_df_to_pkl(all_subjects_demog, config.demographic_data, config.data_dir, config.pickling_protocol)
    save_df_to_pkl(all_subjects_memory, 'raw_' + config.memory_report, config.data_dir, config.pickling_protocol)
    save_df_to_pkl(memory_series, config.memory_report, config.data_dir, config.pickling_protocol)
    save_df_to_pkl(all_subjects_gaze_df, config.all_subject + config.gaze, config.data_dir, config.pickling_protocol)
    save_df_to_pkl(all_subjects_blinks_df, config.all_subject + config.blinks,
                   config.data_dir, config.pickling_protocol)
    save_df_to_pkl(all_subjects_fixations_df, config.all_subject + config.fixations,
                   config.data_dir, config.pickling_protocol)
    save_df_to_pkl(all_subjects_saccades_df, config.all_subject + config.saccades,
                   config.data_dir, config.pickling_protocol)
    return (gaze_dict, blinks_dict, fixations_dict, saccades_dict,
            all_subjects_gaze_df, all_subjects_blinks_df, all_subjects_fixations_df, all_subjects_saccades_df)
