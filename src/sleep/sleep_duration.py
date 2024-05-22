import os
import pickle

sleep_count_dict = {}
directory_str = r'C:\Users\user\PycharmProjects\gaze\Gaze\resources\nap\scoring\output'
directory = os.fsencode(directory_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    __suffix = "_description.txt"
    if not filename.endswith(__suffix):
        subname = filename.split('_')[0]
        sleep_count_dict[subname] = 0
        path = os.path.join(directory_str, filename)
        # load
        try:
            with open(path) as hypno_file:
                for line in hypno_file:
                    if int(line) > 1:
                        sleep_count_dict[subname] += 1
        except PermissionError:
            print(f"Permission denied: {path}")
    else:
        continue

with open(os.path.join(directory_str, 'proc', f"sleep_count_dict.pkl"), 'wb') as f:
    pickle.dump(sleep_count_dict, f)
