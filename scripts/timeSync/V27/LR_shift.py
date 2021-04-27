from pathlib import Path
import itertools
import numpy as np
import matplotlib.pylab as plt
import h5py
import json
import pickle
import glob
import socket
import os
from sklearn.linear_model import LinearRegression

hostname = socket.gethostname()
if hostname == 'HAITI' or 'Gorleben':
    DIR = "C:/Users/xin/OneDrive - bwstaff/xin/trelleborg"
    DATASET = "D:/xin/datasets/Trelleborg/V27/*.h5"
if hostname == 'BALI':
    DIR = "/home/xin/projects/trelleborg"
    DATASET = "/storage/xin/datasets/Trelleborg/V27/*.h5"
if hostname == 'LAPTOP-1FOJITEG':
    DIR = "C:/Users/xinch/OneDrive - bwstaff/xin/trelleborg"
    DATASET = "C:/LINHC/VersucheDBs/Trelleborg/V27/*.h5"
DIR = os.path.abspath(DIR)
DATASET = os.path.abspath(DATASET)

file_list = []
for file in glob.glob(DATASET):
    file_list.append(file)
file_list = sorted(file_list)
file_list_LEM1 = [file for file in file_list if 'LEM1' in file]
file_list_LEM1
        
for file in file_list_LEM1:
## Loading files
    filename = file.split(os.sep)[-1][:-3]
    shift_opts_freqs = pickle.load( open( DIR + "/data/V27/shift_opts_freqs/" +filename + ".pkl", "rb" ) )
    with h5py.File(file, "r") as h51:
        # all low speed area timestamps as "low"
        low = np.where(h51[SPD][:]>-40)[0]
        # skip the first 450 points because of the duration of 1st lsa is always a bit different
        # thees points are also skipped for the consistency in the following codes
        low = low[low>450]
        pos_low = h51['pos1'][low]
        low_sep = np.where(np.diff(low)>3000)[0] # check if 3000 fit for all datasets
        pos_low_sep = np.split(pos_low,low_sep+1)
        Ls = np.append(low[low_sep],low[-1])
        Fs = np.append(low[0],low[low_sep+1])

## LR for all frequencies
        shift_opts_freqs_all = []
        for i in range (121):
            X = np.array(h51['t'][Fs]).reshape(-1,1)
            y = np.array(shift_opts_freqs[i]).reshape(-1,1)
            reg = LinearRegression().fit(X, y)
            shift = reg.predict(np.array(h51['t']).reshape(-1,1)).reshape(-1)
            # LR predicted shift "shfit" should not be more than 2 points away from 
            # grid-search-based "shift "shift_opts_freqs[i]"
            assert all(abs(shift[Fs]-shift_opts_freqs[i]))<2
            shift_opts_freqs_all.append(shift)
        shift_opts_freqs_all = np.array(shift_opts_freqs_all).squeeze().round().astype(int)
        assert shift_opts_freqs_all.shape[0] == 121
        folder = DIR + '/data/V27/shift_opts_freqs_all/'
        Path(folder).mkdir(parents=True, exist_ok=True)
        pklName = folder + filename + '.pkl'
        pickle.dump(shift_opts_freqs_all, open(pklName, 'wb')) 
