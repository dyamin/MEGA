import os
import pickle

import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt

directory_str = r'C:\Users\user\PycharmProjects\gaze\Gaze\resources\nap\scoring\output\proc'

mega_score = pd.read_pickle(os.path.join(directory_str, "mega_score_df.pkl"))
# Transform the mega score to a series with the subject as the index
mega_score = mega_score.set_index([mega_score['Subject']])
mega_score = mega_score['Result']

with open(os.path.join(directory_str, f"sleep_efficiency_dict.pkl"), 'rb') as f:
    sleep_count_dict = pickle.load(f)
sleep_count = pd.Series(list(sleep_count_dict.values()), index=sleep_count_dict.keys())
# Filter out subjects with sleep efficiency < 0.5
sleep_count = sleep_count[sleep_count > 0.5]

# Check if MEGA score is normal distribution
print(scipy.stats.normaltest(mega_score))
# Plot MEGA score distribution
plt.hist(mega_score)
plt.show()

# Check for variance homogeneity between the two groups
print(scipy.stats.levene(mega_score, sleep_count))


# Check if there is a correlation between sleep efficiency and MEGA score
df = pd.concat([sleep_count, mega_score], axis=1)
df.dropna(how='any', inplace=True)
# new plot- scatter plot
plt.scatter(df[0], df['Result'])
plt.xlabel('Sleep efficiency')
plt.ylabel('MEGA score')
plt.show()

# Check if there is a correlation between sleep efficiency and MEGA score
print(scipy.stats.pearsonr(df[0], df['Result']))

# Read the sleep duration dictionary
with open(os.path.join(directory_str, f"sleep_count_dict.pkl"), 'rb') as f:
    sleep_count_dict = pickle.load(f)

sleep_count = pd.Series(list(sleep_count_dict.values()), index=sleep_count_dict.keys())
# Filter out subjects with sleep duration == 0
sleep_count = sleep_count[sleep_count > 0]
# Intersect MEGA and sleep duration
intesc = mega_score.index.intersection(sleep_count.index)
sleep_count = sleep_count[intesc]
mega_score = mega_score[intesc]

# Check if there is a correlation between n1 duration and MEGA score
with open(os.path.join(directory_str, f"n1_dict.pkl"), 'rb') as f:
    n1_dict = pickle.load(f)

n1_duration = pd.Series(list(n1_dict.values()), index=n1_dict.keys())
# Filter out subjects with n1 duration == 0
# n1_duration = n1_duration[n1_duration > 0]
# Intersect MEGA and n1 duration
intesc = mega_score.index.intersection(n1_duration.index)
n1_duration = n1_duration[intesc]
mega_score = mega_score[intesc]

# Check if there is a correlation between (n1 duration / sleep count) and MEGA score
# Intersect n1 duration and sleep count
intesc = n1_duration.index.intersection(sleep_count.index)
n1_duration = n1_duration[intesc]
sleep_count = sleep_count[intesc]
print("n1 duration / sleep count correlation with MEGA score:", scipy.stats.pearsonr(n1_duration / sleep_count, mega_score))
# Check if there is a correlation between n1 duration and MEGA score
print("n1 duration correlation with MEGA score:", scipy.stats.pearsonr(n1_duration, mega_score))

# Check if there is a correlation between n2 duration and MEGA score
with open(os.path.join(directory_str, f"n2_dict.pkl"), 'rb') as f:
    n2_dict = pickle.load(f)

n2_duration = pd.Series(list(n2_dict.values()), index=n2_dict.keys())
# Filter out subjects with n2 duration == 0
# n2_duration = n2_duration[n2_duration > 0]
# Intersect MEGA and n2 duration
intesc = mega_score.index.intersection(n2_duration.index)
n2_duration = n2_duration[intesc]
mega_score = mega_score[intesc]

# Check if there is a correlation between (n2 duration / sleep count) and MEGA score
# Intersect n2 duration and sleep count
intesc = n2_duration.index.intersection(sleep_count.index)
n2_duration = n2_duration[intesc]
sleep_count = sleep_count[intesc]
print("n2 duration / sleep count correlation with MEGA score:", scipy.stats.pearsonr(n2_duration / sleep_count, mega_score))
# Check if there is a correlation between n2 duration and MEGA score
print("n2 duration correlation with MEGA score:", scipy.stats.pearsonr(n2_duration, mega_score))

# Plot n2 and MEGA score scatter plot
plt.scatter(n2_duration, mega_score)
plt.xlabel('n2 duration')
plt.ylabel('MEGA score')
plt.show()

# Check if there is a correlation between n3 duration and MEGA score
with open(os.path.join(directory_str, f"n3_dict.pkl"), 'rb') as f:
    n3_dict = pickle.load(f)

n3_duration = pd.Series(list(n3_dict.values()), index=n3_dict.keys())
# Filter out subjects with n3 duration == 0
# n3_duration = n3_duration[n3_duration > 0]
# Intersect MEGA and n3 duration
intesc = mega_score.index.intersection(n3_duration.index)
n3_duration = n3_duration[intesc]
mega_score = mega_score[intesc]

# Check if there is a correlation between (n3 duration / sleep count) and MEGA score
# Intersect n3 duration and sleep count
intesc = n3_duration.index.intersection(sleep_count.index)
n3_duration = n3_duration[intesc]
sleep_count = sleep_count[intesc]
print("n3 duration / sleep count correlation with MEGA score:", scipy.stats.pearsonr(n3_duration / sleep_count, mega_score))
# Check if there is a correlation between n3 duration and MEGA score
print("n3 duration correlation with MEGA score:", scipy.stats.pearsonr(n3_duration, mega_score))

# Check if there is a correlation between rem duration and MEGA score
with open(os.path.join(directory_str, f"rem_dict.pkl"), 'rb') as f:
    rem_dict = pickle.load(f)

rem_duration = pd.Series(list(rem_dict.values()), index=rem_dict.keys())
# Filter out subjects with rem duration == 0
# rem_duration = rem_duration[rem_duration > 0]
# Intersect MEGA and rem duration
intesc = mega_score.index.intersection(rem_duration.index)
rem_duration = rem_duration[intesc]
mega_score = mega_score[intesc]

# Check if there is a correlation between (rem duration / sleep count) and MEGA score
# Intersect rem duration and sleep count
intesc = rem_duration.index.intersection(sleep_count.index)
rem_duration = rem_duration[intesc]
sleep_count = sleep_count[intesc]
print("rem duration / sleep count correlation with MEGA score:", scipy.stats.pearsonr(rem_duration / sleep_count, mega_score))
# Check if there is a correlation between rem duration and MEGA score
print("rem duration correlation with MEGA score:", scipy.stats.pearsonr(rem_duration, mega_score))


