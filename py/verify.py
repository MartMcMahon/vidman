import numpy as np
import matplotlib.pyplot as plt
raw = np.fromfile("../cap.cs8", dtype=np.int8)
iq = raw[::2].astype(np.float32) + 1j*raw[1::2].astype(np.float32)
plt.specgram(iq, Fs=2.4e6)
plt.show()
