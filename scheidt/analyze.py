import L737
logger = L737.getLogger(__name__)

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
plt.ion()

def avg(x, n=10): return np.convolve(x, np.ones(n), mode="valid")/n

cplot = lambda x, *args, **kwargs: plt.plot(x.real, x.imag, *args, **kwargs)


filename = Path("C:\\LINHC\\VersucheDBs\\Trelleborg\\merge_resampled_1s_dropna.pkl")
filename = Path("C:\\LINHC\\VersucheDBs\\Trelleborg\\2020-05-28_resampled_50ms.pkl")
filename = Path("C:\\LINHC\\VersucheDBs\\Trelleborg\\merge_resampled_160ms_dropna.pkl")

data = pd.read_pickle(filename)
tv = data.Teilversuch
versuch = data.Versuch
pos1 = data['Ist-MG10: Kammer1-Istpos.[mm]'].values
pos2 = data['Ist-MG11: Kammer2-Istpos.[mm]'].values

tv_test=5

plt.figure("Reflection Test")
test_ctl = [
    # tuple with following information order col_rfl, col_thr
    ("S11_LEM1", "S21_P1_LEM1"),
    ("S22_LEM1", "S21_P2_LEM1"),
    ("S11_LEM2", "S21_P1_LEM2"),
    ("S22_LEM2", "S21_P2_LEM2"),
]
f_idx=15; 
for ktest, (col_rfl, col_thr) in enumerate(test_ctl):
    plt.plot(abs(data[col_rfl,f_idx].values[tv==tv_test]-data[col_thr, f_idx].values[tv==tv_test]), label=col_thr)

plt.legend(loc=0)

plt.figure("Transmission Test")
test_ctl = [
    # tuple with following information order col_rfl, col_thr
    ("S11_LEM1", "S21_P1_LEM1"),
    ("S22_LEM1", "S21_P2_LEM1"),
    ("S11_LEM2", "S21_P1_LEM2"),
    ("S22_LEM2", "S21_P2_LEM2"),
]
f_idx=15; 
for ktest, (col_rfl, col_thr) in enumerate(test_ctl):
    plt.plot(abs(data[col_thr, f_idx].values[tv==tv_test]), ".-", label=col_thr)

plt.legend(loc=0)


