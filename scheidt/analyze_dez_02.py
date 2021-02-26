import L737
logger = L737.getLogger(__name__)

import os
import h5py as h5py
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
plt.ion()

def avg(x, n=10): return np.convolve(x, np.ones(n), mode="valid")/n

cplot = lambda x, *args, **kwargs: plt.plot(x.real, x.imag, *args, **kwargs)

filename = Path("C:\\LINHC\\VersucheDBs\\Trelleborg\\2020-12-02_LEM2_dropna.hdf5")
h5 = h5py.File(filename, "r")

t1 = time.perf_counter()
raise Exception()
t2 = time.perf_counter()

ts = data['Ist-Ablauf-Zeit, ca.'].values
pos1 = data['Ist-MG10: Kammer1-Istpos.[mm]'].values
pos2 = data['Ist-MG11: Kammer2-Istpos.[mm]'].values

# speed1 = (pos1[1:] - pos1[:-1]) / (ts[1:] - ts[:-1])
# speed2 = (pos2[1:] - pos2[:-1]) / (ts[1:] - ts[:-1])

speed1 = data["Ist-MG15: Durchschnittsgeschw.1[mm/s]"].values
speed2 = data["Ist-MG16: Durchschnittsgeschw.2[mm/s]"].values

freq = data['Ist-MG1: Teller-Drehfrequenz [Hz]'].values

p1 = data["Ist-MG3: Kammer1-Druck[bar]"].values
p2 = data["Ist-MG5: Kammer2-Druck[bar]"].values

f1 = data["Ist-MG6: Kammer1-Kraft[kN]"].values
f2 = data["Ist-MG7: Kammer2-Kraft[kN]"].values

phi = data["Ist-MG14: Motor-Istwinkel[°]"].values

if 'S21_P1_LEM1' in data:
    lem_tag = "1"
    s21 = data[('S21_P1_LEM1')].values
    s22 = data[('S22_LEM1')].values
    s11 = data[('S11_LEM1')].values
else:
    lem_tag = "2"
    s21 = data[('S21_P1_LEM2')].values
    s22 = data[('S22_LEM2')].values
    s11 = data[('S11_LEM2')].values


plt.figure("Latest Signal")
cplot((s11[-10000:,20]),".", label=f"S11 LEM{lem_tag} Nov 12")
cplot((s22[-10000:,20]),".", label=f"S22 LEM{lem_tag} Nov 12")
cplot((s21[-10000:,20]),".", label=f"S21 LEM{lem_tag} Nov 12")
plt.legend(loc=0)
plt.grid(1)
plt.title("500MHz")

lv_move=np.sum(abs(s21[1:]-s21[:-1]),axis=1)

t3 = time.perf_counter()

cc_index = 10
cm22_tem = s22[:, cc_index] - s21[:,cc_index]
cm11_tem = s11[:, cc_index] - s21[:,cc_index]

# if there will be other cable this has to be adjusted
k1_damp = -0.0301110989
k2_damp = 2.12499618e-10
k3_damp = 2.78397924e-05
c_cable = 200000000.0

l_cable = 0.5
delta_l_cable = 0

f_act = cc_index * 10e6 + 300e6

for k1 in range(10):
    cable_att = 10**(-(k1_damp + k2_damp * f_act +
                    k3_damp * np.sqrt(f_act))/20*l_cable)
    l_cable_TX = l_cable - delta_l_cable
    l_cable_RX = l_cable + delta_l_cable
    CTX = np.exp(-2j*np.pi*f_act*l_cable_TX/c_cable) * cable_att
    CRX = np.exp(-2j*np.pi*f_act*l_cable_RX/c_cable) * cable_att

    cmtx = (s11[:, cc_index].T / CTX / CTX - s21[:, cc_index].T / CTX / CRX).T
    cmrx = (s22[:, cc_index].T / CRX / CRX - s21[:, cc_index].T / CTX / CRX).T

    # plt.figure("fft")
    # plt.plot(abs(np.fft(n_sig.windows.hamming(121)*cmtx[:,0])))
    # plt.plot(abs(np.fft(n_sig.windows.hamming(121)*cmrx[:,0])))

    dltx = ((-np.pi - np.angle(cmtx) + np.pi) % (2*np.pi) - np.pi) / (2*np.pi) * c_cable / 2 / f_act
    dlrx = ((-np.pi - np.angle(cmrx) + np.pi) % (2*np.pi) - np.pi) / (2*np.pi) * c_cable / 2 / f_act

    # plt.figure("phi")
    # plt.plot(dltx, "b")
    # plt.plot(dlrx, "r")

    delta_l_cable = np.mean(dlrx - dltx)/2
    l_cable = l_cable + np.mean(dlrx + dltx) / 2


f_act = np.linspace(300e6, 1500e6, 121)

cable_att = 10**(-(k1_damp + k2_damp * f_act +
                k3_damp * np.sqrt(f_act))/20*l_cable)
CTX = np.exp(-2j*np.pi*f_act*l_cable_TX/c_cable) * cable_att
CRX = np.exp(-2j*np.pi*f_act*l_cable_RX/c_cable) * cable_att

c21 = s21 / CTX / CRX
cmtx = (s11 / CTX / CTX - c21)
cptx = (s11 / CTX / CTX + c21)



raise Exception()
idx1 = np.arange(57750,58000, dtype=int)
idx2 = np.arange(58100,58300, dtype=int)

# idx1 = np.arange(42000,43000, dtype=int)
# idx2 = np.arange(91000,92000, dtype=int)


p_idx1 = np.mean(p1[idx1])
p_idx2 = np.mean(p1[idx2])

# signalverhältnisse vs. frequenz
fmess = np.linspace(300,1500,121)
plt.figure("Signalbeträge vs f")
ax1 = plt.subplot(2,1,1)
plt.title("Max/Min Signalbeträge vs f")
ax2 = plt.subplot(2,1,2, sharex=ax1)
ax1.plot(fmess,np.amin(abs(s21[idx1,:]),axis=0),"b",label=f"Kammer 1 p {p_idx1:.1f}bar")
ax1.plot(fmess,np.amax(abs(s21[idx1,:]),axis=0),"b")
ax1.plot(fmess,np.amin(abs(s21[idx2,:]),axis=0),"r",label=f"Kammer 1 p {p_idx2:.1f}bar")
ax1.plot(fmess,np.amax(abs(s21[idx2,:]),axis=0),"r")

ax1.legend(loc=0)
ax2.legend(loc=0)
ax1.set_ylabel("Abs(S21)")
ax2.set_ylabel("Abs(S21)")
ax2.set_xlabel("Frequency (MHz)")

delta_step = 2
if delta_step >= 0:
    t_idx10 = np.arange(delta_step, idx1.shape[0] - 1, dtype=int)
    t_idx11 = np.arange(0, idx1.shape[0]-delta_step - 1, dtype=int)
    t_idx20 = np.arange(delta_step, idx2.shape[0] - 1, dtype=int)
    t_idx21 = np.arange(0, idx2.shape[0]-delta_step - 1, dtype=int)
else:
    t_idx11 = np.arange(-delta_step, idx1.shape[0] - 1, dtype=int)
    t_idx10 = np.arange(0, idx1.shape[0] + delta_step - 1, dtype=int)
    t_idx21 = np.arange(-delta_step, idx2.shape[0] - 1, dtype=int)
    t_idx20 = np.arange(0, idx2.shape[0] + delta_step - 1, dtype=int)

for f_idx in [10,110]:
    plt.figure(f"Signal S21 @{fmess[f_idx]}MHz vs Pos1/2")
    ax1 = plt.subplot(2,1,1)
    plt.title(f"Signal S21 @{fmess[f_idx]}MHz vs Pos1/2")
    ax2 = plt.subplot(2,1,2, sharex=ax1, sharey=ax1)
    pos10_cor = pos1[idx1][t_idx10] +speed1[idx1][t_idx10]*2*f_idx/3*1e-3
    pos11_cor = pos1[idx2][t_idx10] +speed1[idx2][t_idx10]*2*f_idx/3*1e-3
    pos20_cor = pos2[idx1][t_idx20] +speed2[idx1][t_idx20]*2*f_idx/3*1e-3
    pos21_cor = pos2[idx2][t_idx20] +speed2[idx2][t_idx20]*2*f_idx/3*1e-3
    ax1.plot(pos10_cor,abs(s21_1[idx1,f_idx])[t_idx11], ".",label=f"Kammer 1 {p_idx1:.1f}bar 0.1Hz")
    ax1.plot(pos20_cor,abs(s21_2[idx1,f_idx])[t_idx21], ".",label=f"Kammer 2 {p_idx1:.1f}bar 0.1Hz")
    ax1.plot(pos11_cor,abs(s21_1[idx2,f_idx])[t_idx11], ".",label=f"Kammer 1 {p_idx2:.1f}bar 0.1Hz")
    ax1.plot(pos21_cor,abs(s21_2[idx2,f_idx])[t_idx21], ".",label=f"Kammer 2 {p_idx2:.1f}bar 0.1Hz")
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

