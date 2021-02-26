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

filename = Path("C:\\LINHC\\VersucheDBs\\Trelleborg\\2020-08-20_Konstantdruck_merge_resampled_80ms_dropna.pkl")

data = pd.read_pickle(filename)
ts = data['Ist-Ablauf-Zeit, ca.'].values
pos1 = data['Ist-MG10: Kammer1-Istpos.[mm]'].values
pos2 = data['Ist-MG11: Kammer2-Istpos.[mm]'].values

speed1 = (pos1[1:] - pos1[:-1]) / 80e-3
speed2 = (pos2[1:] - pos2[:-1]) / 80e-3
freq = data['Ist-MG1: Teller-Drehfrequenz [Hz]'].values

p1 = data["Ist-MG3: Kammer1-Druck[bar]"].values
p2 = data["Ist-MG5: Kammer2-Druck[bar]"].values

f1 = data["Ist-MG6: Kammer1-Kraft[kN]"].values
f2 = data["Ist-MG7: Kammer2-Kraft[kN]"].values

phi = data["Ist-MG14: Motor-Istwinkel[°]"].values
s21_1 = data[('S21_P1_LEM1')].values
s21_2 = data[('S21_P1_LEM2')].values

idx1 = np.arange(10000,11000, dtype=int)
idx2 = np.arange(60000,61000, dtype=int)

p_idx1 = np.mean(p1[idx1])
p_idx2 = np.mean(p1[idx2])

# signalverhältnisse vs. frequenz
fmess = np.linspace(300,1500,121)
plt.figure("Signalbeträge vs f")
ax1 = plt.subplot(2,1,1)
plt.title("Max/Min Signalbeträge vs f")
ax2 = plt.subplot(2,1,2, sharex=ax1)
ax1.plot(fmess,np.amin(abs(s21_1[idx1,:]),axis=0),"b",label=f"Kammer 1 p {p_idx1:.1f}bar")
ax1.plot(fmess,np.amax(abs(s21_1[idx1,:]),axis=0),"b")
ax1.plot(fmess,np.amin(abs(s21_1[idx2,:]),axis=0),"r",label=f"Kammer 1 p {p_idx2:.1f}bar")
ax1.plot(fmess,np.amax(abs(s21_1[idx2,:]),axis=0),"r")

ax2.plot(fmess,np.amin(abs(s21_2[idx1,:]),axis=0),"b",label=f"Kammer 2 p {p_idx1:.1f}bar")
ax2.plot(fmess,np.amax(abs(s21_2[idx1,:]),axis=0),"b")
ax2.plot(fmess,np.amin(abs(s21_2[idx2,:]),axis=0),"r",label=f"Kammer 2 p {p_idx2:.1f}bar")
ax2.plot(fmess,np.amax(abs(s21_2[idx2,:]),axis=0),"r")

ax1.legend(loc=0)
ax2.legend(loc=0)
ax1.set_ylabel("Abs(S21)")
ax2.set_ylabel("Abs(S21)")
ax2.set_xlabel("Frequency (MHz)")

so = 2
for f_idx in [20, 40, 60]:
    plt.figure(f"Signal S21 @{fmess[f_idx]}MHz vs Pos1/2")
    ax1 = plt.subplot(2,1,1)
    plt.title(f"Signal S21 @{fmess[f_idx]}MHz vs Pos1/2")
    ax2 = plt.subplot(2,1,2, sharex=ax1, sharey=ax1)
    ax1.plot(pos1[idx1][:-so],abs(s21_1[idx1,f_idx])[so:],label=f"Kammer 1 {p_idx1:.1f}bar 0.1Hz")
    ax1.plot(pos2[idx1][:-so],abs(s21_2[idx1,f_idx])[so:],label=f"Kammer 2 {p_idx1:.1f}bar 0.1Hz")
    ax1.plot(pos1[idx2][:-so],abs(s21_1[idx2,f_idx])[so:],label=f"Kammer 1 {p_idx2:.1f}bar 0.1Hz")
    ax1.plot(pos2[idx2][:-so],abs(s21_2[idx2,f_idx])[so:],label=f"Kammer 2 {p_idx2:.1f}bar 0.1Hz")
    ax1.set_ylabel("Abs(S21)")
    ax1.legend(loc=0)
    ax1.set_xlabel("Pos1/2 (mm)")
    ax1.set_ylabel("Abs(S21)")
    ax1.legend(loc=0)
    ax1.grid(True)
    plt.tight_layout()

    plt.figure(f"Signal S21 @{fmess[f_idx]}MHz vs Pos1/2")
    ax1 = plt.subplot(2,1,1)
    plt.title(f"Signal S21 @{fmess[f_idx]}MHz vs Pos1/2")
    ax2 = plt.subplot(2,1,2)
    ax2.plot(phi[idx1],abs(s21_1[idx1,f_idx]), ".", label=f"Kammer 1 {p_idx1:.1f}bar 0.1Hz")
    ax2.plot(phi[idx1],abs(s21_2[idx1,f_idx]), ".", label=f"Kammer 2 {p_idx1:.1f}bar 0.1Hz")
    ax2.plot(phi[idx2],abs(s21_1[idx2,f_idx]), ".", label=f"Kammer 1 {p_idx2:.1f}bar 0.1Hz")
    ax2.plot(phi[idx2],abs(s21_2[idx2,f_idx]), ".", label=f"Kammer 2 {p_idx2:.1f}bar 0.1Hz")
    ax2.set_ylabel("Abs(S21)")
    ax2.legend(loc=0)
    ax2.set_xlabel("phi1/2 (°)")
    ax2.set_ylabel("Abs(S21)")
    ax2.legend(loc=0)
    ax2.grid(True)
    plt.tight_layout()
