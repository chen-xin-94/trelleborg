from pathlib import Path
import itertools
import numpy as np
import matplotlib.pylab as plt
import h5py
import json
import pickle
import glob
from sklearn.linear_model import LinearRegression

# DIR = "C:/LINHC/Software/Python/L737/scribble/xin/trelleborg"
DIR = "/home/xin/projects/trelleborg"

DATASET = "/storage/xin/datasets/Trelleborg/2021-01-27-V24/*/*.h5"

file_list = []
for file in glob.glob(DATASET):
    file_list.append(file)

for file in file_list:
## Loading files
    # filename = file.split('\\')[-1][:-3]
    filename = file.split('/')[-1][:-3]
    shift_opts_freqs = pickle.load( open( DIR + "/data/shift_opts_freqs/" +filename + ".pkl", "rb" ) )
    with h5py.File(file, "r") as h51:
        low = np.where(h51['spd1'][:]>-50)[0]
        # skip the lsa in first 10000 points because some wierd patterns in speed, 
        # check 'C:\\LINHC\\VersucheDBs\\Trelleborg\\2021-01-27-V24\\2_Phase_A-D\\20210127_Phase_A-D_LEM2.h5', 
        low = low[low>10000]
        pos_low = h51['pos1'][low]
        low_sep = np.where(np.diff(low)>3000)[0] # check if 3000 fit for all datasets
        pos_low_sep = np.split(pos_low,low_sep+1)
        Ls = np.append(low[low_sep],low[-1])
        Fs = np.append(low[0],low[low_sep+1])
        if filename in ["20210218_7_Phase_C-F_LEM1","20210218_7_Phase_C-F_LEM2"]:
            Ls = Ls[:-1]
            Fs = Fs[:-1]
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
        pklName = DIR + '/data/shift_opts_freqs_all/' + filename + '.pkl'
        pickle.dump(shift_opts_freqs_all, open(pklName, 'wb')) 
