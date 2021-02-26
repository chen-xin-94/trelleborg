from pathlib import Path

import numpy as np
import matplotlib.pylab as plt
import h5py

plt.ion()
def avg(x, n=10): return np.convolve(x, np.ones(n), mode="valid")/n

cplot = lambda x, *args, **kwargs: plt.plot(x.real, x.imag, *args, **kwargs)

filename_1 = Path("C:/LINHC/VersucheDBs/Trelleborg/20210127_Phase_A-D_LEM1.h5")
filename_2 = Path("C:/LINHC/VersucheDBs/Trelleborg/20210127_Phase_A-D_LEM2.h5")

h51 = h5py.File(filename_1, "r")
h52 = h5py.File(filename_2, "r")

print("ATTR:")
print(h51.attrs.keys())

print("Keys:")
print(h51.keys())

fidx = 5
plt.figure("LEM1 vs LEM2")

plt.subplot(2,1,1)
cplot(h51["S21_P1_LEM1"][:10000,fidx],".")
cplot(h51["S21_P1_LEM1"][-10000:,fidx],".")

plt.grid(1)
plt.xlim(-0.6, 0.6)
plt.ylim(-0.6, 0.6)

plt.subplot(2,1,2)
cplot(h52["S21_P1_LEM2"][:10000,fidx],".")
cplot(h52["S21_P1_LEM2"][-10000:,fidx],".")

plt.grid(1)
plt.xlim(-0.6, 0.6)
plt.ylim(-0.6, 0.6)

h51.close()
h52.close()