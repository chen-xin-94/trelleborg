import numpy as np
import h5py
import json
import pickle
import glob
import itertools

file_list = []
for file in glob.glob(r"C:\LINHC\VersucheDBs\Trelleborg\2021-01-27-V24\*\*.h5"):
    file_list.append(file)

for file in file_list:
    h51 = h5py.File(file, "r")
    filename = file.split('\\')[-1][:-3]

    if 'LEM1' in filename:
        POS = 'pos1'
        SPD = 'spd1'
        S21 = 'S21_P1_LEM1'
        T = 'k1t'
        P = 'k1p'
        F =  'k1f'
        AT = 'k1at'
        PHI = 'phi'
        IP1 = 'ip1k1'
        IP2 = 'ip2k1'

        pos1=[]
        spd1=[]
        k1t=[]
        k1p=[]
        k1f=[]
        k1at=[]
        phi=[]
        ip1k1=[]
        ip2k1=[]
    elif 'LEM2' in filename:
        POS = 'pos2'
        SPD = 'spd2'
        S21 = 'S21_P1_LEM2'
        T = 'k2t'
        P = 'k2p'
        F =  'k2f'
        AT = 'k2at'
        PHI = 'phi'
        IP1 = 'ip1k2'
        IP2 = 'ip2k2'
        pos2=[]
        spd2=[]
        k2t=[]
        k2p=[]
        k2f=[]
        k2at=[]
        phi=[]
        ip1k2=[]
        ip2k2=[]
    features_setup = [    
                        T ,
                        P ,
                        F ,
                        AT ,
                        PHI ,
                        IP1 ,
                        IP2 
    ]
    s21=[[] for _ in range(121)] # s21 for all frequendcies
    
## load the shifts    
    shift_opts_freqs = pickle.load( open( "./data/shift_opts_freqs/" +filename + ".pkl", "rb" ) )

## Find the low speed areas
    low = np.where(h51[SPD][:]>-50)[0]
    # skip the lsa in first 10000 points because some wierd patterns in speed, 
    # check 'C:\\LINHC\\VersucheDBs\\Trelleborg\\2021-01-27-V24\\2_Phase_A-D\\20210127_Phase_A-D_LEM2.h5', 
    low = low[low>10000]
    pos_low = h51[POS][low]
    low_sep = np.where(np.diff(low)>10000)[0]
    pos_low_sep = np.split(pos_low,low_sep+1)
    Ls = np.append(low[low_sep],low[-1])
    Fs = np.append(low[0],low[low_sep+1])
    
    # for dataset 20210218_7_Phase_C-F_LEM1
    # and dataset 20210218_7_Phase_C-F_LEM2
    # delete the last the lsa because the last lsa contains only 1900+ points instead of 2700+ for some reasons
    if filename in ["20210218_7_Phase_C-F_LEM1","20210218_7_Phase_C-F_LEM2"]:
        Ls = Ls[:-1]
        Fs = Fs[:-1]


## create a list of arrays max_sep, each array contains peak values of the corresponding low speed area.
    max_sep = []
    for k in range(len(pos_low_sep)):
        # temporary sequency ts
        ts = pos_low_sep[k] 

        # find top 50 highest values
        max_20 = np.argsort(ts)[-50:][::-1]

        # filter the max value for each period
        temp =[max_20[0]]
        for i in max_20:
            if all([abs(j-i)>50 for j in temp]):        
                temp.append(i)
        max_sep.append(np.sort(temp))
        
    # There should be 4 max values in each interval,
    # except for num = 10,i.e. dataset 20210218_7_Phase_C-F_LEM1
    #        and num = 11 i.e. dataset 20210218_7_Phase_C-F_LEM2
    # because lsa because the last lsa contains only 1900+ points instead of 2700+ for some reasons
    if filename not in ["20210218_7_Phase_C-F_LEM1","20210218_7_Phase_C-F_LEM2"]:
        assert all([len(max_sep[i])==4 for i in range(len(max_sep))]) 


## choose intervals with safe points (2000 points after the first peak value, 
    # 2000 is to make sure no points in hsa are included, peak value is to make sure the same starting point for each interval)
    pos_low_sep_safe=[]
    s21_low_sep_safe=[]
    for i,F in enumerate(Fs):
        pos_low_sep_safe.append(h51[POS][F+max_sep[i][0] : F+max_sep[i][0]+2000])
        s21_low_sep_safe.append((h51[S21][F+max_sep[i][0] : F+max_sep[i][0]+2000, :]).transpose())


    

## data extraction
    for i,Fc in enumerate(Fs):
        F = Fc + max_sep[i][0] 
        L = F + 2000
        for feature in features_setup:
            eval(feature).append(h51[feature][F:L])

    for freq in range (121):
        FF = [F - shift for shift in shift_opts_freqs[freq]]
        LL = [L - shift for shift in shift_opts_freqs[freq]]
        for j in range(len(FF)):
            s21[freq].append(h51[S21][FF[j]:LL[j], freq])


## save the file as hdf5
    saved_hdf5 = 'C:/LINHC/VersucheDBs/Trelleborg/2021-01-27-V24_lsa_shifted/' + filename + '_lsa_shifted.h5'


    with h5py.File(saved_hdf5, 'w') as f:
        setup = f.create_group("setup")
        liview = f.create_group("liview")
        
        liview.create_dataset('s21',data=np.array(s21).reshape(121,-1))
        
        for feature in features_setup:
            tmp = np.array(eval(feature)).reshape(-1)
            setup.create_dataset(feature,data=tmp)
                
        feature_lookup={}
        
        for j in range(242):
            if j%2 == 0:
                key = 's21-'+str(j//2)+'-real'
            else:
                key = 's21-'+str(j//2)+'-imag'
            feature_lookup[key] = j
        
        i = 242
        for key in setup.keys():
            feature_lookup[key] = i
            i += 1

        f.attrs["feature_lookup"] = json.dumps(feature_lookup)     
        
        f.attrs["name_lookup_rev"] = h51.attrs["name_rev_lookup"]
