"""
Created on Tue May  7 11:16:48 2019

@author: legscmf
"""
import L737
logger = L737.getLogger(__name__)

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd


def avg(x, n=10): return np.convolve(x, np.ones(n), mode="valid")/n

cplot = lambda x, *args, **kwargs: plt.plot(x.real, x.imag, *args, **kwargs)

DYNAMIC_LIMIT = 1e-6


REFDATA_INTERVAL_FINE_CNT = 2000
REFDATA_INTERVAL_COARSE_CNT = 200

class IP_COMPLEX:
    def __init__(self):
        self.v = 1+1j
        self.df = 1+1j


class POLAR:
    def __init__(self):
        self.amplitude = 0.0
        self.phase = 0.0


class DataManagementValuesEOLBlock:
    def __init__(self):
        self.software = "20"
        self.specification_version = "04"
        self.data_version = "00"
        self.hw_version = "D01"

        self.ProductName = "EOL_BLOCK_______"


class EolParams:

    def __init__(self):
        self.AttnCurves = EolAttenuation()
        self.PortCalib = EolPortCalib()
        self.LemCalib = EolLemCalib()
        self.SingleParams = EolSingleParams()

        self.data_management_values = DataManagementValuesEOLBlock()


class EolAttenuation:
    def __init__(self, n=REFDATA_INTERVAL_COARSE_CNT):
        self.length = n # only useful if all members have the same length, otherwise delete it

        # create default data in the first element, the others are derived from this
        self.attn_Rng0 = np.concatenate(((34 + 12) * np.ones(60, dtype="int"), (18 + 12) * np.ones(n-60, dtype="int")))
        self.attn_Rng1 = np.zeros((n), dtype="int")
        self.attn_Rng2 = np.zeros((n), dtype="int")
        self.attn_Rng3 = np.zeros((n), dtype="int")
        self.attn_Rng4 = np.zeros((n), dtype="int")


class EolPortCalib:
    def __init__(self, n=REFDATA_INTERVAL_COARSE_CNT):
        self.length = n # only useful if all members have the same length, otherwise delete it

        self.OpaTx = [IP_COMPLEX() for k in range(n)]
        self.OpbTx = [IP_COMPLEX() for k in range(n)]
        self.OpcTx = [IP_COMPLEX() for k in range(n)]
        self.Sw3Tx = [IP_COMPLEX() for k in range(n)]

        # values for Rx path (between TNC connector) and mixer */
        self.OpaRx = [IP_COMPLEX() for k in range(n)]
        self.OpbRx = [IP_COMPLEX() for k in range(n)]
        self.OpcRx = [IP_COMPLEX() for k in range(n)]
        self.Sw3Rx = [IP_COMPLEX() for k in range(n)]

        # values for through path */
        self.ThrRx = [IP_COMPLEX() for k in range(n)]

        self.ThrTx_Rng0 = [IP_COMPLEX() for k in range(n)]
        self.ThrTx_Rng1 = [IP_COMPLEX() for k in range(n)]
        self.ThrTx_Rng2 = [IP_COMPLEX() for k in range(n)]
        self.ThrTx_Rng3 = [IP_COMPLEX() for k in range(n)]
        self.ThrTx_Rng4 = [IP_COMPLEX() for k in range(n)]

        self.Kthr_Rng0 = [IP_COMPLEX() for k in range(n)]
        self.Kthr_Rng1 = [IP_COMPLEX() for k in range(n)]
        self.Kthr_Rng2 = [IP_COMPLEX() for k in range(n)]
        self.Kthr_Rng3 = [IP_COMPLEX() for k in range(n)]
        self.Kthr_Rng4 = [IP_COMPLEX() for k in range(n)]


class EolLemCalib:
    def __init__(self, n=REFDATA_INTERVAL_FINE_CNT):
        self.length = n # only useful if all members have the same length, otherwise delete it

        self.ref_amp = np.zeros((n), dtype="float")
        self.xtalk_norm = np.zeros((n), dtype="complex")
        self.noise_norm = np.zeros((n), dtype="float")
        self.s11_norm = [POLAR() for k in range(n)]
        self.s22_norm = [POLAR() for k in range(n)]
        self.saw_norm = [POLAR() for k in range(n)]



class EolSingleParams(object):
    """ Contains:
        - single value EoL calibration parameters
    """
    def __init__(self):
        self.RefAmpT0 = 0
        self.S11NormT0 = 0
        self.S22NormT0 = 0
        self.SawNormT0 = 0
        self.SensorDiffTemp = 0


class SPC:

    def __init__(self, lem_eol_fn: str):

#        self.eol = pickle.load(Path(__file__).parent.joinpath("eol_data", lem_eol_fn).open(mode="br"))
        self.eol = RenamingUnpickler(Path(__file__).parent.joinpath("eol_data", lem_eol_fn).open(mode="br")).load()


    def S11_eff(self, mrfltx):
        return 1/(self.OPCTX - self.OPBTX / (self.OPATX - mrfltx))

    def S22_eff(self, mrfltx):
        return 1/(self.OPCRX - self.OPBRX / (self.OPARX - mrfltx))


    def correct(self, mTHR_raw, seff, SW3, KTHR, THRRX11, THRTX22):
        # sq = sqrt(4*CRX**4*CTX**4*CYL21**2*SW3RX**2 + CRX**4*S11eff**2*SW3RX**2 - 2*CRX**2*CTX**2*S11eff*SW3RX + CTX**4)
        # CYL11 = (CRX**2*S11eff*SW3RX + CTX**2 -+ sq)/(2*CRX**2*CTX**2*SW3RX)
        # CYL21 = mTHR*(-CRX**2*CYL11*THRRX11 - CTX**2*CYL11*THRTX22 + 1)/(CRX*CTX*KTHR)

       # eta = 1
        mTHR = np.where(abs(mTHR_raw) < DYNAMIC_LIMIT, DYNAMIC_LIMIT, mTHR_raw)
        a = (1+seff*SW3)/SW3
        b = - seff/SW3
        m = KTHR/mTHR
        m2 = m**2
        AA = m2 - (a*THRRX11*THRTX22-THRTX22-THRRX11)**2
        BB = -m2*a-2*(1+THRRX11*THRTX22*b)*(a*THRRX11*THRTX22-THRTX22-THRRX11)
        CC = -m2*b-(1+THRRX11*THRTX22*b)**2

        DD = np.sqrt(BB**2-4*AA*CC)
        y1 = (-BB+DD)/(2*AA)
        y2 = (-BB-DD)/(2*AA)
                
        y = np.where(abs(y2)<=abs(y1), y2, y1)

        dut11 = y
        dut22 = dut11

        dut21 = 1/m*(1+(a*y+b)*THRRX11*THRTX22-y*(THRRX11+THRTX22))

        return (dut11, dut21, dut22)


    def interpolate_parameter(self, ip_complex, f_frac):
        return ip_complex.v + f_frac * ip_complex.df

    def on_evaluate(self, path_info2, frq_rf, mw_cycle1, mw_cycle2):
        """Will be called in each measurment cycle. Data are exchanged. By global data objects.
        """

        if (path_info2 != "S11" and
                path_info2 != "S22"):
            return

        # calculate the right parameters
        f_index = int((frq_rf - 100e6) / 10e6)
        f_frac = (frq_rf % 10e6)

        self.OPARX = self.interpolate_parameter(self.eol.PortCalib.OpaRx[f_index], f_frac)
        self.OPBRX = self.interpolate_parameter(self.eol.PortCalib.OpbRx[f_index], f_frac)
        self.OPCRX = self.interpolate_parameter(self.eol.PortCalib.OpcRx[f_index], f_frac)

        self.OPATX = self.interpolate_parameter(self.eol.PortCalib.OpaTx[f_index], f_frac)
        self.OPBTX = self.interpolate_parameter(self.eol.PortCalib.OpbTx[f_index], f_frac)
        self.OPCTX = self.interpolate_parameter(self.eol.PortCalib.OpcTx[f_index], f_frac)

        self.SW3TX = self.interpolate_parameter(self.eol.PortCalib.Sw3Tx[f_index], f_frac)
        self.SW3RX = self.interpolate_parameter(self.eol.PortCalib.Sw3Rx[f_index], f_frac)

        self.THRRX11 = self.interpolate_parameter(self.eol.PortCalib.ThrRx[f_index], f_frac)

        self.KTHR = self.interpolate_parameter(self.eol.PortCalib.Kthr_Rng0[f_index], f_frac)
        self.THRTX22 = self.interpolate_parameter(self.eol.PortCalib.ThrTx_Rng0[f_index], f_frac)


        if path_info2 == "S11":
            self.seff11 = self.S11_eff(mw_cycle2)
            seff = self.seff11
            sw3 = self.SW3RX
        else:
            self.seff22 = self.S22_eff(mw_cycle2)
            seff = self.seff22
            sw3 = self.SW3TX

        result = self.correct(
            mw_cycle1,
            seff,
            sw3,
            self.KTHR,
            self.THRRX11,
            self.THRTX22)

        return (result[0], result[1])

class RenamingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        print(module, name)
        if module == 'L737.devices.LEG.LEM.parameter_v01':
            module = '__main__'

        return super().find_class(module, name)

def correct_data(filename: str) -> pd.DataFrame:
    """Correct a pickled pandas data file with LiView Data
    pickle pandas frame to file with nearly same name but extended by corr
    
    Arguments:
        filename {str} -- complete filename to the pickled pandas data file
    
    Returns:
        [pd.DataFrame] -- the data frame where scattering parameters are compensated for electronics effects
    """
    data = pd.read_pickle(filename)
    lem1_eol_fn = "200324_1319_LEM_A3XPB4_eolblock.pck"
    lem2_eol_fn = "200324_1319_LEM_A3XPB6_eolblock.pck"
    f_plan = np.arange(300e6, 1500e6, 10e6)
    spc1 = SPC(lem1_eol_fn)
    spc2 = SPC(lem2_eol_fn)
    convert_ctl = [
        # tuple with following information order meas_path, col_rfl, col_thr, spc
        ("S11", "S11_LEM1", "S21_P1_LEM1", spc1),
        ("S22", "S22_LEM1", "S21_P2_LEM1", spc1),
        ("S11", "S11_LEM2", "S21_P1_LEM2", spc2),
        ("S22", "S22_LEM2", "S21_P2_LEM2", spc2),
    ]
    t1 = time.perf_counter()
    for kctl, (meas_path, col_rfl, col_thr, spc) in enumerate(convert_ctl):
        for kf, f in enumerate(f_plan[0:]):
            if col_thr in data:
                zw=(spc.on_evaluate(meas_path,f, data[(col_thr,kf)].values, data[(col_rfl, kf)].values))
                data[(col_rfl,kf)] = zw[0]
                data[(col_thr,kf)] = zw[1]

    t2 = time.perf_counter()

    pickle.dump(data, Path(filename).parent.joinpath(Path(filename).stem + "_corr.pkl").open("bw"))

    return data

if __name__ =="__main__":
    import matplotlib.pylab as plt
    plt.ion()

    filename = "C:\\LINHC\\VersucheDBs\\Trelleborg\\merge_resampled_60s.pkl"
    filename = "C:\\LINHC\\VersucheDBs\\Trelleborg\\merge_resampled_1s_dropna.pkl"
    filename = "C:\\LINHC\\VersucheDBs\\Trelleborg\\2020-05-28_resampled_50ms.pkl"
    filename = Path("C:\\LINHC\\VersucheDBs\\Trelleborg\\merge_resampled_160ms_dropna.pkl")
    filename = "C:\\LINHC\\VersucheDBs\\Trelleborg\\2020-11-24_LEM1_dropna.pkl"
    filename = "C:\\LINHC\\VersucheDBs\\Trelleborg\\2020-11-24_LEM2_dropna.pkl"

    data = correct_data(filename)
    # correct names
    col_names = [name[0][0] if name[1]=="" else name  for name in data.columns]
    data.columns = col_names

