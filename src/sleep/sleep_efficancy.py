import os
import pickle

sleep_efficiency_dict = {}
directory_str = r'C:\Users\user\PycharmProjects\gaze\Gaze\resources\nap\scoring\output'
directory = os.fsencode(directory_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    __suffix = "Alice.txt"
    if filename.endswith(__suffix):
        subname = filename.split('_')[0]
        sleep_efficiency_dict[subname] = 0
        path = os.path.join(directory_str, filename)
        # load
        try:
            with (open(path) as hypno_file):
                # Number of lines in the file
                sleep_onset, sleep_offset = 0, 0
                print(f"Processing {path}")
                for line_num, line in enumerate(hypno_file):
                    # Sleep efficency defined as the ratio of time spent asleep out of the time between sleep onset and final awakening
                    if int(line) > 1:
                        sleep_onset = line_num if sleep_onset == 0 else sleep_onset
                        sleep_offset = line_num
                        sleep_efficiency_dict[subname] += 1
                sleep_efficiency_dict[subname] = (sleep_efficiency_dict[subname] / (sleep_offset - sleep_onset)) if sleep_offset > sleep_onset else 0

        except PermissionError:
            print(f"Permission denied: {path}")
    else:
        continue

with open(os.path.join(directory_str, 'proc', f"sleep_efficiency_dict.pkl"), 'wb') as f:
    pickle.dump(sleep_efficiency_dict, f)
