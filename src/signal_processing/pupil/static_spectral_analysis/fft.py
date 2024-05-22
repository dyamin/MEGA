import matplotlib.pyplot as plt
import numpy as np

from src.signal_processing import config
from src.signal_processing.pupil.utils import get_filtered_pupil_radius_df

pupil_df = get_filtered_pupil_radius_df()
# visualize example subject and movie
subject_movie = pupil_df.xs(('TC9', 'mov73'), level=['Subject', 'Movie'])
plt.subplots_adjust(hspace=1)

sessionA = subject_movie.xs(('Session A',), level=['Session'])
plt.subplot(211)
# FFT
npnts = len(sessionA.values)
fft_signal = np.fft.fft(sessionA.values) / npnts
hz = np.linspace(0, config.sampling_rate / 2, int(np.floor(npnts / 2) + 1))
# Extract amplitude
signal_amp = np.abs(fft_signal[0:len(hz)])
signal_amp[1:] = 2 * signal_amp[1:]
# Frequency domain representation
plt.title('Fourier transform: Session A')
plt.plot(hz, signal_amp[0:len(hz)])
plt.xlabel('Frequency')
plt.xlim([0, 5])
plt.ylabel('Amplitude')

sessionB = subject_movie.xs(('Session B',), level=['Session'])
npnts = len(sessionB.values)
plt.subplot(212)
# FFT
fft_signal = np.fft.fft(sessionB.values) / npnts
hz = np.linspace(0, config.sampling_rate / 2, int(np.floor(npnts / 2) + 1))
# Extract amplitude
signal_amp = np.abs(fft_signal[0:len(hz)])
signal_amp[1:] = 2 * signal_amp[1:]
# Frequency domain representation
plt.title('Fourier transform: Session B')
plt.plot(hz, signal_amp[0:len(hz)])
plt.xlabel('Frequency')
plt.xlim([0, 5])
plt.ylabel('Amplitude')

plt.show()
