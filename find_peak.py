import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import math


# Extract Raw Audio from Wav File
# signal = "".split(",")
signal = [0.0,
 82.763077974032,
 0.0,
 -85.333141628561,
 -84.3362937419794,
 -80.13419305691559,
 -83.4430535018366,
 0.0,
 -68.1985905136482,
 0.0]
signal_positive = np.array([math.ceil(abs(float(x))) for x in signal])
signal_original = np.array([math.ceil(float(x)) for x in signal])

plt.title("Signal Wave...")
plt.plot(signal_original)
plt.show(origin="upper")

## find max peaks in signals
peaks, _ = find_peaks(signal_positive, distance=10)
# peaks = find_peaks_cwt(signal, widths=[1, 25])

plt.plot(signal_original)
plt.plot(peaks, signal_original[peaks], "x")
fig1 = plt.gcf()
plt.show()
fig1.savefig("peak.png")

print(sorted(signal_positive))

print(peaks)
print(signal_original[peaks])
print(max(signal_original))
print(min(signal_original))