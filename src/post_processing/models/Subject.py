import pandas as pd


class Subject:

    def __init__(self, subj_path: str, subj_memory: pd.Series, num_sessions: int = 2) -> None:
        import os
        assert (os.path.isdir(subj_path)), "Couldn't find a directory in the given path.\n"
        self.main_dir = subj_path
        self.subject_id = os.path.basename(subj_path)
        self.number_of_sessions = num_sessions
        self.memory = subj_memory

        subdirs = [dirname for dirname in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, dirname))]
        assert (len(subdirs) == self.number_of_sessions
                ), f'Was expecting exactly {self.number_of_sessions} sub-directories, found {len(subdirs)}.\n'
        assert (len([dirname for dirname in subdirs if dirname.split('_')[0] == self.subject_id]) == len(subdirs)
                ), f'Subject\'s sessions directories should be named like {self.subject_id}_A.\n'

        import string
        for i, dirname in enumerate(subdirs):
            ses = f'session_{string.ascii_uppercase[i]}_dir'
            setattr(self, ses, os.path.join(subj_path, dirname))
        return

    def __str__(self) -> str:
        subject_id = f'Subject {self.subject_id}'
        number_of_sessions = f'Number of sessions: {self.number_of_sessions}'
        main_dir = f'Main directory: {self.main_dir}'
        session_dirs = dict()
        for key, value in vars(self).items():
            if key.startswith('session_') & key.endswith('_dir'):
                session = key.split('_')[1]
                session_dirs[f'Session {session} directory'] = value
        sessions_strings = '\n\t'.join([f'\t{key}: {value}' for (key, value) in session_dirs.items()])

        return f'{subject_id}:\n\t{number_of_sessions}\n\t{main_dir}\n\t{sessions_strings}'

    def get_subject_id(self) -> str:
        return self.subject_id

    def get_main_directory(self) -> str:
        return self.main_dir

    def get_memory(self) -> pd.Series:
        return self.memory
