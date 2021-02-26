import L737
logger = L737.getLogger(__name__)

import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
from matplotlib.dates import datestr2num
plt.ion()

def avg(x, n=10): return np.convolve(x, np.ones(n), mode="valid")/n

cplot = lambda x, *args, **kwargs: plt.plot(x.real, x.imag, *args, **kwargs)
header_str = "Zeitstempel;Ist-Ablauf-Zeit, ca.;Ist-MG1: Teller-Drehfrequenz [Hz];Soll-MG1: Teller-Drehfrequenz [Hz];Ist-MG2: Kammer1-Temp[°C];Soll-MG2: Kammer1-Temp[°C];Ist-MG3: Kammer1-Druck[bar];Soll-MG3: Kammer1-Druck[bar];Ist-MG4: Kammer2-Temp.[°C];Soll-MG4: Kammer2-Temp.[°C];Ist-MG5: Kammer2-Druck[bar];Soll-MG5: Kammer2-Druck[bar];Ist-MG6: Kammer1-Kraft[kN];Soll-MG6: Kammer1-Kraft[kN];Ist-MG7: Kammer2-Kraft[kN];Soll-MG7: Kammer2-Kraft[kN];Ist-MG8: Kammer1-Außentemp.[°C];Soll-MG8: Kammer1-Außentemp.[°C];Ist-MG9: Kammer2-Außentemp.[°C];Soll-MG9: Kammer2-Außentemp.[°C];Ist-MG10: Kammer1-Istpos.[mm];Soll-MG10: Kammer1-Istpos.[mm];Ist-MG11: Kammer2-Istpos.[mm];Soll-MG11: Kammer2-Istpos.[mm];Ist-MG12: Hub1[mm];Soll-MG12: Hub1[mm];Ist-MG13: Hub2[mm];Soll-MG13: Hub2[mm];Ist-MG14: Motor-Istwinkel[°];Soll-MG14: Motor-Istwinkel[°];Ist-MG15: Durchschnittsgeschw.1[mm/s];Soll-MG15: Durchschnittsgeschw.1[mm/s];Ist-MG16: Durchschnittsgeschw.2[mm/s];Soll-MG16: Durchschnittsgeschw.2[mm/s];"
headers = dict()
for k1, name in enumerate(header_str.split(";")):
    headers[k1] = name

filename = Path("C:\\LINHC\\VersucheDBs\\Trelleborg\\200818_Daten E3 Prüfstand\\Versuch 13 - Feder 1 Konstant- und Nulldruck\\AD-07-18_LiView_V13_50Hz_T60°C_200bar-konstant-null_mit Bewegung_v50mms-200mms_2020_08_17_SSB_1.rdm")
str2num = lambda s: 0 if s==b"" else float(s.replace(b",",b"."))
converters = dict()
converters[0] = lambda s: 0
for k1 in range(1,35):
    converters[k1] = str2num

teststand=np.loadtxt(filename,delimiter=";",skiprows=5,converters=converters)

idx_k1_pos = 20
idx_k2_pos = 22

idx_k1_p = 6
idx_k2_p = 10

idx_k1_f = 12
idx_k2_f = 14

filename_lv = Path("C:\\LINHC\\VersucheDBs\\Trelleborg\\200818_Daten Liview\\2020-08-17 Test Feder 1 Konstant- und Nulldruck\\2020-08-17_10-55-38.pkl")


signal_names = ['S21_P1_LEM1',
 'S21_P1_time_LEM1',
 'S21_P2_LEM1',
 'S21_P2_time_LEM1',
 'S22_LEM1',
 'S22_time_LEM1',
 'S11_LEM1',
 'S11_time_LEM1',
 'S21_P1_LEM2',
 'S21_P1_time_LEM2',
 'S21_P2_LEM2',
 'S21_P2_time_LEM2',
 'S22_LEM2',
 'S22_time_LEM2',
 'S11_LEM2',
 'S11_time_LEM2']


lv_data_dir = Path("C:\\LINHC\\VersucheDBs\\Trelleborg\\200818_Daten Liview\\2020-08-17 Test Feder 1 Konstant- und Nulldruck")

filenames_to_load = list()
for name in lv_data_dir.iterdir():
    filenames_to_load.append(name)

lv_signal = dict()
lv_signal_count = dict()

for filename_lv in filenames_to_load[:]:
    print(filename_lv)
    with open(filename_lv, "br") as fp:
        data_lv = pickle.load(fp)

    for k1, name in enumerate(signal_names):
        zw = data_lv[data_lv.attr==name].iloc[:,0:121]
        if name not in lv_signal:
            lv_signal[name] = np.zeros((90000, 121), dtype = np.complex128)
            lv_signal_count[name] = 0

        index = lv_signal_count[name]
        length = zw.shape[0]
        if index+length>lv_signal[name].shape[0]:
            break
        else:
            lv_signal[name][index:index+length] = zw
            lv_signal_count[name] = lv_signal_count[name] + length


