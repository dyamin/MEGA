import os
import pickle

n1_dict, n2_dict, n3_dict, rem_dict = {}, {}, {}, {}
directory_str = r'C:\Users\user\PycharmProjects\gaze\Gaze\resources\nap\scoring\output'
directory = os.fsencode(directory_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    __suffix = "Alice.txt"
    if filename.endswith(__suffix):
        subname = filename.split('_')[0]
        n1_dict[subname] = 0
        n2_dict[subname] = 0
        n3_dict[subname] = 0
        rem_dict[subname] = 0
        path = os.path.join(directory_str, filename)
        # load
        try:
            with (open(path) as hypno_file):
                # Number of lines in the file
                sleep_onset, sleep_offset = 0, 0
                print(f"Processing {path}")
                for line_num, line in enumerate(hypno_file):
                    # Sleep efficency defined as the ratio of time spent asleep out of the time between sleep onset and final awakening
                    if int(line) == 1:
                        sleep_onset = line_num if sleep_onset == 0 else sleep_onset
                        sleep_offset = line_num
                        n1_dict[subname] += 1
                    elif int(line) == 2:
                        n2_dict[subname] += 1
                    elif int(line) == 3:
                        n3_dict[subname] += 1
                    elif int(line) == 4:
                        rem_dict[subname] += 1

        except PermissionError:
            print(f"Permission denied: {path}")
    else:
        continue

with open(os.path.join(directory_str, 'proc', f"n1_dict.pkl"), 'wb') as f:
    pickle.dump(n1_dict, f)
with open(os.path.join(directory_str, 'proc', f"n2_dict.pkl"), 'wb') as f:
    pickle.dump(n2_dict, f)
with open(os.path.join(directory_str, 'proc', f"n3_dict.pkl"), 'wb') as f:
    pickle.dump(n3_dict, f)
with open(os.path.join(directory_str, 'proc', f"rem_dict.pkl"), 'wb') as f:
    pickle.dump(rem_dict, f)