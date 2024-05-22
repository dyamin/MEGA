import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from src.signal_processing import config
from src.signal_processing.distance_from_roi.utils import get_filtered_roi_df

distance_df = get_filtered_roi_df()
subject_movie = distance_df.xs(('TC9', 'mov73'), level=['Subject', 'Movie'])
data = subject_movie.xs(('Session A',), level=['Session'])
npnts = len(data)
nyq = config.sampling_rate / 2
plt.subplots_adjust(hspace=1)

# Convolution in the time domain == Multiplication in the frequency domain
# I'll use the multiplication in the frequency domain flow, because it's more efficient

# 1. Length of the result of Convolution: len(signal) + len(kernel) - 1
# 2. FFTs of the signal and kernel
# 3. multiply spectra
# 4. IFFT
# 5. Cut of "wings"

# Convolution as spectral filter:

plt.plot(data)
y250 = gaussian_filter1d(data, 250)
y100 = gaussian_filter1d(data, 100)
plt.plot(data, 'k', label='original data')
plt.plot(data.index, y250, '--', label='filtered, sigma=250')
plt.plot(data.index, y100, ':', label='filtered, sigma=100')
plt.legend()
plt.grid()
plt.show()

plt.subplot(211)
plt.plot(data)
b, a = signal.butter(1, 1 / nyq, btype='low', analog=False)
butter_filtered = signal.filtfilt(b, a, data)
plt.plot(data.index, butter_filtered)

plt.subplot(212)
plt.plot(data)
window = signal.windows.boxcar(150)
# filter the data using convolution
boxcar_filtered = np.convolve(window / len(window), data, mode="same")
plt.plot(data.index, boxcar_filtered)
plt.show()
