import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
np.random.seed(0)

# Sine sample with a frequency of 5hz and add some noise
sr = 32  # sampling rate
y = np.linspace(0, 5 * 2*np.pi, sr)
y = np.tile(np.sin(y), 5)
y += np.random.normal(0, 1, y.shape)
t = np.arange(len(y)) / float(sr)

# Generate frquency spectrum
spectrum, freqs, _ = plt.magnitude_spectrum(y, sr)

# Calculate percentage for a frequency range 
lower_frq, upper_frq = 4, 6
ind_band = np.where((freqs > lower_frq) & (freqs < upper_frq))
plt.fill_between(freqs[ind_band], spectrum[ind_band], color='red', alpha=0.6)
frq_band_perc = auc(freqs[ind_band], spectrum[ind_band]) / auc(freqs, spectrum)
print('{:.1%}'.format(frq_band_perc))
# plt.plot(frq_band_perc)
plt.show()