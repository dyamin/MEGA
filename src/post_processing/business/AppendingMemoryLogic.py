import pandas as pd

from src import config


def append_memory_to_subject_dataframe(subj_id: str,
                                       apendee_dataframe: pd.DataFrame,
                                       appended_memory: pd.Series) -> pd.DataFrame:
    assert (subj_id in apendee_dataframe.index.unique(level=config.SUBJECT)
            ), f'Couldn\'t find subject {subj_id} in the provided DataFrame'
    unique_index_level = apendee_dataframe.index.names[-1]
    appended_memory.columns = [config.MEMORY]
    appended_memory.name = config.MEMORY
    apendee_dataframe = apendee_dataframe.join(appended_memory)
    apendee_dataframe.set_index([config.MEMORY], append=True, inplace=True)
    apendee_dataframe = apendee_dataframe.reorder_levels(
        [config.SUBJECT, config.SESSION, config.MOVIE, config.MEMORY, unique_index_level])
    return apendee_dataframe
