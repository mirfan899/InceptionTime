import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import math


# Extract Raw Audio from Wav File
signal = "1030.0,-154978.112660385,-175307.704094207,0.0,0.0,1029.5,-694540.565048932,-249483.827087739,32.0053832080835,32.0053832080835,1031.75,-614147.7731592361,-301578.676526524,0.0,49.3987053549955,1030.75,-537944.093104776,-335345.164609131,41.185925165709705,32.0053832080835,1033.75,-476757.77291967295,-355546.965796352,0.0,47.726310993906296,1034.75,-393151.303670211,-360919.01406404603,64.7988763545249,47.290610042638505,1035.5,-294909.389901756,-351489.067755147,59.036243467926504,46.0050860052542,1038.75,919782.886576012,-169878.78856498198,64.1336432059055,52.6960517220166,1038.0,387519.562301705,-90250.4527268836,56.30993247402021,47.290610042638505,1039.25,739263.839638235,28251.5890395619,0.0,47.726310993906296,1038.25,669025.460271689,119790.713501294,7.1250163489018,42.273689006093704,1038.5,613050.3829645771,190256.38056747802".split(",")
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