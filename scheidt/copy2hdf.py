import time
import json
import pandas as pd
from pathlib import Path

import numpy as np
import h5py as h5py

filename = Path(
    "C:\\LINHC\\VersucheDBs\\Trelleborg\\2020-12-02_LEM2_dropna.pkl")
h5filename = filename.parent / (filename.name.split(".")[0] + ".hdf5")

# Name lookup

# some helpers if new values are needed

# lower case name_lookup
# for key in name_lookup:
#     name_lookup[key] = name_lookup[key].lower()

# name_rev_lookup = dict()
# for key in name_lookup:
#     name_rev_lookup[name_lookup[key]] = key

name_lookup = {
    'Ist-Ablauf-Zeit, ca.': 't',
    'Ist-MG1: Teller-Drehfrequenz [Hz]': 'tdf', 
    'Soll-MG1: Teller-Drehfrequenz [Hz]': 'stdf', 
    'Ist-MG2: Kammer1-Temp[°C]': 'k1t',
    'Soll-MG2: Kammer1-Temp[°C]': 'sk1t', 
    'Ist-MG3: Kammer1-Druck[bar]': 'k1p', 
    'Soll-MG3: Kammer1-Druck[bar]': 'sk1p', 
    'Ist-MG4: Kammer2-Temp.[°C]': 'k2t', 
    'Soll-MG4: Kammer2-Temp.[°C]': 'sk2t', 
    'Ist-MG5: Kammer2-Druck[bar]': 'k2p', 
    'Soll-MG5: Kammer2-Druck[bar]': 'sk2p', 
    'Ist-MG6: Kammer1-Kraft[kN]': 'k1f', 
    'Soll-MG6: Kammer1-Kraft[kN]': 'sk1f', 
    'Ist-MG7: Kammer2-Kraft[kN]': 'k2f', 
    'Soll-MG7: Kammer2-Kraft[kN]': 'sk2f', 
    'Ist-MG8: Kammer1-Außentemp.[°C]': 'k1at', 
    'Soll-MG8: Kammer1-Außentemp.[°C]': 'sk1at', 
    'Ist-MG9: Kammer2-Außentemp.[°C]': 'k2at', 
    'Soll-MG9: Kammer2-Außentemp.[°C]': 'sk2at', 
    'Ist-MG10: Kammer1-Istpos.[mm]': 'pos1', 
    'Soll-MG10: Kammer1-Istpos.[mm]': 'spos1', 
    'Ist-MG11: Kammer2-Istpos.[mm]': 'pos2', 
    'Soll-MG11: Kammer2-Istpos.[mm]': 'spos2', 
    'Ist-MG12: Hub1[mm]': 'hub1', 
    'Soll-MG12: Hub1[mm]': 'shub1', 
    'Ist-MG13: Hub2[mm]': 'hub2', 
    'Soll-MG13: Hub2[mm]': 'shub2', 
    'Ist-MG14: Motor-Istwinkel[°]': 'phi', 
    'Soll-MG14: Motor-Istwinkel[°]': 'sphi', 
    'Ist-MG15: Durchschnittsgeschw.1[mm/s]': 'spd1', 
    'Soll-MG15: Durchschnittsgeschw.1[mm/s]': 'sspd1', 
    'Ist-MG16: Durchschnittsgeschw.2[mm/s]': 'spd2', 
    'Soll-MG16: Durchschnittsgeschw.2[mm/s]': 'sspd2'
    }

name_rev_lookup = {
 't': 'Ist-Ablauf-Zeit, ca.',
 'itdf': 'Ist-MG1: Teller-Drehfrequenz [Hz]',
 'stdf': 'Soll-MG1: Teller-Drehfrequenz [Hz]',
 'ik1t': 'Ist-MG2: Kammer1-Temp[°C]',
 'sk1t': 'Soll-MG2: Kammer1-Temp[°C]',
 'ik1p': 'Ist-MG3: Kammer1-Druck[bar]',
 'sk1p': 'Soll-MG3: Kammer1-Druck[bar]',
 'ik2t': 'Ist-MG4: Kammer2-Temp.[°C]',
 'sk2t': 'Soll-MG4: Kammer2-Temp.[°C]',
 'ik2p': 'Ist-MG5: Kammer2-Druck[bar]',
 'sk2p': 'Soll-MG5: Kammer2-Druck[bar]',
 'ik1f': 'Ist-MG6: Kammer1-Kraft[kN]',
 'sk1f': 'Soll-MG6: Kammer1-Kraft[kN]',
 'ik2f': 'Ist-MG7: Kammer2-Kraft[kN]',
 'sk2f': 'Soll-MG7: Kammer2-Kraft[kN]',
 'ik1at': 'Ist-MG8: Kammer1-Außentemp.[°C]',
 'sk1at': 'Soll-MG8: Kammer1-Außentemp.[°C]',
 'ik2at': 'Ist-MG9: Kammer2-Außentemp.[°C]',
 'sk2at': 'Soll-MG9: Kammer2-Außentemp.[°C]',
 'ipos1': 'Ist-MG10: Kammer1-Istpos.[mm]',
 'spos1': 'Soll-MG10: Kammer1-Istpos.[mm]',
 'ipos2': 'Ist-MG11: Kammer2-Istpos.[mm]',
 'spos2': 'Soll-MG11: Kammer2-Istpos.[mm]',
 'ihub1': 'Ist-MG12: Hub1[mm]',
 'shub1': 'Soll-MG12: Hub1[mm]',
 'ihub2': 'Ist-MG13: Hub2[mm]',
 'shub2': 'Soll-MG13: Hub2[mm]',
 'iphi': 'Ist-MG14: Motor-Istwinkel[°]',
 'sphi': 'Soll-MG14: Motor-Istwinkel[°]',
 'ispd1': 'Ist-MG15: Durchschnittsgeschw.1[mm/s]',
 'sspd1': 'Soll-MG15: Durchschnittsgeschw.1[mm/s]',
 'ispd2': 'Ist-MG16: Durchschnittsgeschw.2[mm/s]',
 'sspd2': 'Soll-MG16: Durchschnittsgeschw.2[mm/s]'
 }


t1 = time.perf_counter()
data = pd.read_pickle(filename)
t2 = time.perf_counter()

print(f"reading took {t2-t1}s")

ts = data[('Ist-Ablauf-Zeit, ca.', '')]

h5file = h5py.File(h5filename, "w")
h5file.attrs["name_lookup"] = json.dumps(name_lookup)
h5file.attrs["name_rev_lookup"] = json.dumps(name_rev_lookup)

t1 = time.perf_counter()
for key in data.keys():
    if key[0] in name_lookup:
        print(key)
        h5file[name_lookup[key[0]]] = data[key].values
        h5file.flush()
    else:
        if key[0] not in h5file:
            print(key)
            h5file[key[0]] = np.array(data[key[0]], dtype="csingle")
            h5file.flush()

t2 = time.perf_counter()

print(f"copyying data took {t2-t1}s")

h5file.close()
