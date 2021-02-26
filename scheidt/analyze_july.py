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

filename = Path("C:\\LINHC\\VersucheDBs\\Trelleborg\\2020-07-14_merge_resampled_80ms_dropna.pkl")
#filename = Path("C:\\LINHC\\VersucheDBs\\Trelleborg\\2020-07-06_merge_resampled_160ms_dropna.pkl")

data = pd.read_pickle(filename)
ts = data['Ist-Ablauf-Zeit, ca.'].values
pos1 = data['Ist-MG10: Kammer1-Istpos.[mm]'].values
pos2 = data['Ist-MG11: Kammer2-Istpos.[mm]'].values

freq = data['Ist-MG1: Teller-Drehfrequenz [Hz]'].values

p1 = data["Ist-MG3: Kammer1-Druck[bar]"].values
p2 = data["Ist-MG5: Kammer2-Druck[bar]"].values

f1 = data["Ist-MG6: Kammer1-Kraft[kN]"].values
f2 = data["Ist-MG7: Kammer2-Kraft[kN]"].values

s21_1 = data['S21_P1_LEM1'].values
s21_2 = data['S21_P1_LEM2'].values


plt.figure("Signal S21 @500MHz vs Pos1/2")
ax1 = plt.subplot(2,1,1)
plt.title("Signal S21 @500MHz vs Pos1/2")
ax2 = plt.subplot(2,1,2, sharex=ax1)
ax1.plot(pos1[10000:12000],abs(s21_1[10000:12000,20]),label="Kammer 1 0.1Hz")
ax1.plot(pos2[10000:12000],abs(s21_2[10000:12000,20]),label="Kammer 2 0.1Hz")
ax2.plot(pos1[25000:27000],abs(s21_1[25000:27000,20]),label="Kammer 1 0.2Hz")
ax2.plot(pos2[25000:27000],abs(s21_2[25000:27000,20]),label="Kammer 2 0.2Hz")
ax1.set_ylabel("Abs(S21)")
ax1.legend(loc=0)
ax2.set_xlabel("Pos1/2 (mm)")
ax2.set_ylabel("Abs(S21)")
ax2.legend(loc=0)

plt.figure("Signal S21 @1000MHz vs Pos1/2")
plt.plot(pos1[10000:12000],abs(s21_1[10000:12000,70]),label="Kammer 1")
plt.plot(pos2[10000:12000],abs(s21_2[10000:12000,70]),label="Kammer 2")
plt.xlabel("Pos1/2 (mm)")
plt.ylabel("Abs(S21)")
plt.legend(loc=0)