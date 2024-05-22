import os

import numpy as np
import scipy.io as sio
from visbrain.io.rw_hypno import (swap_hyp_values, _write_hypno_txt_sample)

directory_str = r'C:\Users\user\PycharmProjects\gaze\Gaze\resources\nap\scoring'
directory = os.fsencode(directory_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    __suffix = "_scoringNew.mat"
    if filename.endswith(__suffix):
        subname = filename.split('_')[0]
        path = os.path.join(directory_str, filename)
        # load the matlab file into numpy
        matlab_scoring = sio.loadmat(path)
        scoring_np = matlab_scoring['scoring']  # transpose()
        scoring_np = np.asarray(scoring_np[0], dtype=int)
        scoring_1sec = scoring_np[1:-1:250].copy()
        hypno_filename = subname + '_hypnoWholeFile_visFormatFromAlice.txt'
        hypno_path = os.path.join(directory_str, 'output', hypno_filename)
        np.savetxt(hypno_path, scoring_1sec, fmt='%d')

        # desc={'W': 0, 'N1':1,'N2':2,'N3':3,'REM':4,'Art':-1}
        desc = {'W': 200, 'N1': -100, 'N2': -200, 'N3': -300, 'REM': 100, 'Art': 0}
        hypno_converted = swap_hyp_values(scoring_1sec, desc)
        # saving:
        _write_hypno_txt_sample(hypno_path, hypno_converted, 1)
    else:
        continue
