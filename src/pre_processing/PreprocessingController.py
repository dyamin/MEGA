import fnmatch
import os

from src import config
from src.pre_processing.business.CiliPreprocLogic import preproc_sub
from src.pre_processing.utils import decentralize_raw_data, decentralize_features

content = config.raw_data_dir
preproc_output = os.path.join(content, 'output')


def preproc_subjects():
    for file in os.listdir(content):
        if fnmatch.fnmatch(file, '*.asc'):
            print(f'Preprocessing {file}')
            preproc_sub(content, file, preproc_output)


def decentralize():
    for file in os.listdir(preproc_output):
        if fnmatch.fnmatch(file, '*_data.pkl'):
            print(f'Decentralizing {file}')
            bs_per_movie = decentralize_raw_data(os.path.join(preproc_output, file), config.decentralized_data_dir)
            decentralize_features(bs_per_movie, os.path.join(preproc_output, file), config.decentralized_data_dir)


preproc_subjects()
decentralize()
