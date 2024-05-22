import datetime as datetime
import os

import pandas as pd

from src import config as global_config


def aggregate_data(rootdir: str, show=False) -> (pd.DataFrame, pd.DataFrame, pd.Series):
    '''
    Params: path -> path the the data file
    Returns: DataFrame of demographic data and DataFrame of behavioral data

    This function iterates over the files in rootdir, and from each one of them
    extracts both the demographic data and the behavioral data.


    Params: rootdir is an absolute path the the directory which contains all experimental data.
            show is a boolean flag the indicates if function should print status along the way
    Returns: Aggregated DataFrames of all data in rootdir

    This function iterates over all the files and aggregates all the data into 2 DataFrames:

    1. Demographic data, which will look like the following:
                             Data
                   Age  Gender  TimeA  TimeB
                 |---------------------------|
    Subject |                           |
    -------------|                           |
                 |                           |
                 |                           |
                 |                           |
                 |                           |
                 |---------------------------|

    2. Behavioral DataFrame, which will look like the following:
                                           Data
                            Seen? A | Conf A | Seen? B | Conf B
                         |--------------------------------------|
    Subject | Movie |                                      |
    ---------------------|                                      |
                 |       |                                      |
                 |       |                                      |
                 |       |
                  |
                 |       |                                      |
                 |       |--------------------------------------|

    3. Behavioral Series, which will look like the following:
    VALUES are {-4,-3,-2,-1,1,2,3,4} based on memory*confidence:
        negative = subject said they didn't watch before
        positive = subject said they did watch before
        |1|=low confidence ... |4|=high confidence

     Subject   |  Session  | Movie | Confidence  |
    -------------------------------|-------------|
    (SubjectA) | Session A | mov1  | {value_AA1} |
               |           | mov2  | {value_AA2} |
    (SubjectB) | Session A | mov1  | {value_BA1} |
       ...     |     ...   | ...   |    ...      |
    (SubjectA) | Session B | mov1  | {value_AB1} |
               |           | mov2  | {value_AB2} |
    (SubjectB) | Session B | mov1  | {value_BB1} |
       ...     |     ...   | ...   |    ...      |


    Data API
    ----------------------------------------------------------------------
    rootdir.dir-> name1_TimeA
                  name1_TimeB
                  name2_TimeA
                  name2_TimeB
                  .
                  .
                  .
                  nameN_TimeA
                  nameN_TimeB
    ----------------------------------------------------------------------
    '''
    all_subjects_demog = dict()
    all_subjects_memory = dict()
    subject_names = list()

    # walks over the directory tree, with root = rootdir
    # subdir - current dir
    # dirs - list of dirs in subdir
    # files - lits of files in subdir
    for subdir, dirs, files in os.walk(rootdir):

        # file looks like the following: name_year_month_day_hour.csv
        for file in files:

            print('Start processing file: ', file)
            path = os.path.join(rootdir, file)
            subject_raw_data = pd.read_csv(path)

            subject_name, subject_demog = _extract_demographic_from_data(subject_raw_data.columns[0])
            # subject_demog looks like the following: {name: [age, gender, time]}
            subject_behave_raw = extract_memory_from_data(subject_raw_data[1:])
            # subject_behave_raw looks like the following: [...,[movie #, seen?, confidence],...]

            subject_names.append(subject_name)

            print(subject_demog[subject_name])
            # changes time string to time_stamp
            subject_demog[subject_name][2] = datetime.datetime.now().strptime(subject_demog[subject_name][2],
                                                                              global_config.DATE_FORMAT)

            # adds subject's demographic data to the aggregated form
            if subject_name not in all_subjects_demog:
                all_subjects_demog.update(subject_demog)
            else:
                # adds the second session Time_stamp to the subject's demographic data
                time_stamp_A = all_subjects_demog[subject_name][2]
                time_stamp_B = subject_demog[subject_name][2]
                second_session_appears_after_first = time_stamp_B > time_stamp_A
                if second_session_appears_after_first:
                    all_subjects_demog[subject_name].append(subject_demog[subject_name][2])
                else:
                    all_subjects_demog[subject_name].insert(2, subject_demog[subject_name][2])

            # adds subject's behavioral data to the aggregated form
            if subject_name not in all_subjects_memory:
                all_subjects_memory.update({subject_name: subject_behave_raw})
            else:
                # adds each movie's second session (Seen? , Confidence) to the subject's behavioral data
                for movie in range(len(all_subjects_memory[subject_name])):
                    if second_session_appears_after_first:
                        seen_B = subject_behave_raw[movie][1]
                        confidence_B = subject_behave_raw[movie][2]
                        all_subjects_memory[subject_name][movie].extend([seen_B, confidence_B])
                    else:
                        seen_A = subject_behave_raw[movie][1]
                        confidence_A = subject_behave_raw[movie][2]
                        all_subjects_memory[subject_name][movie].insert(1, confidence_A)
                        all_subjects_memory[subject_name][movie].insert(1, seen_A)

                # creates a DataFrame for behavioral data
                subject_behave = pd.DataFrame(all_subjects_memory[subject_name],
                                              columns=['Movie #', 'Seen? A', 'Confidence A', 'Seen? B', 'Confidence B'])
                subject_behave.set_index('Movie #', inplace=True)
                all_subjects_memory[subject_name] = subject_behave

    # creates a DataFrame for demographic data
    all_subjects_demog = pd.DataFrame.from_dict(all_subjects_demog, orient='index')
    all_subjects_demog.columns = ['Age', 'Gender', 'Time A', 'Time B']
    all_subjects_demog.index.name = global_config.SUBJECT
    # calculates time differences between sessions
    all_subjects_demog['Time_Between_Sessions'] = all_subjects_demog['Time B'] - all_subjects_demog['Time A']

    # concatenate subject's behavioral data Data Frames into one DataFrame
    all_subjects_memory = pd.concat(all_subjects_memory, names=["Subject", 'Movie #'])

    # changes 'Seen? A' & 'Seen? B' all_sbuject_behave columns' values to 1 and 0 (from 1 and 2)
    all_subjects_memory['Seen? A'] = all_subjects_memory['Seen? A'].apply(lambda val: 0 if val == 2 else 1)
    all_subjects_memory['Seen? B'] = all_subjects_memory['Seen? B'].apply(lambda val: 0 if val == 2 else 1)

    # create the behavioral-data Series from the behavioral DataFrame:
    memory_series = _convert_memory_dataframe_to_series(all_subjects_memory)

    if show:
        print("Demographics:")
        print(all_subjects_demog.head())
        print(".\n.\n.")
        print(all_subjects_demog.tail())

        print("Behavioral:")
        print(all_subjects_memory.head())
        print(".\n.\n.")
        print(all_subjects_memory.tail())

    return all_subjects_demog, all_subjects_memory, memory_series


def extract_memory_from_data(behave_data_raw: pd.DataFrame) -> pd.DataFrame:
    '''
    behave_data_raw API:

                    col
        |-------------------------|
        | movie1 seen? confidnece |
        | movie2 seen? confidnece |
        |            .            |
        |            .            |
        |            .            |
        | movieN seen? confidnece |
        |-------------------------|
    '''
    data_col = behave_data_raw.columns[0]
    behave_data = list()
    for ind, movie_data in enumerate(behave_data_raw[data_col]):
        seen_confidence = movie_data.split()[1:]
        seen = seen_confidence[0].split("'")[1]
        confidence = seen_confidence[1].split("'")[1]
        behave_data.append([ind + 1, int(seen), int(confidence)])

    if len(behave_data) == 0:
        for ind in range(1, global_config.NUMBER_OF_ANIMATIONS):
            behave_data.append([ind, 2, 4])

    return behave_data


def _convert_memory_dataframe_to_series(behave_df):
    # rename movies to the formula 'mov{number}'
    copy_df = behave_df.reset_index()
    copy_df['Movie #'] = copy_df['Movie #'].apply(lambda idx: f'mov{idx}')
    copy_df.set_index(['Subject', 'Movie #'], inplace=True)

    copy_df['Seen? A'].replace({1: 1, 0: -1}, inplace=True)
    copy_df['Seen? B'].replace({1: 1, 0: -1}, inplace=True)
    sesA = copy_df['Seen? A'].multiply(copy_df['Confidence A'])
    sesB = copy_df['Seen? B'].multiply(copy_df['Confidence B'])
    series = pd.concat([sesA, sesB], axis=0, keys=['Session A', 'Session B']).rename('Confidence')
    series.index.names = [global_config.SESSION, global_config.SUBJECT, global_config.MOVIE]
    return series.reorder_levels([global_config.SUBJECT, global_config.SESSION, global_config.MOVIE])


def _extract_demographic_from_data(demog_data: list) -> (str, dict):
    '''demog_data API: 'Name Age Gender Seesion-Time' '''
    subject_demog = demog_data.split()
    subject_demog_dict = {subject_demog[0]: subject_demog[1:]}
    return subject_demog[0], subject_demog_dict
