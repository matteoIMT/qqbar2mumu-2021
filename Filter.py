import pandas as pd
import uproot
import awkward as ak
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math

from collections import Counter
from itertools import combinations
from tqdm import tqdm
from time import time

import ProjectPackage.DataExtraction as de
import ProjectPackage.Kinematic as km
from ProjectPackage import Cut

data_folder = 'D:/Data_muons/dimuonData_LHC18m'
run_number = 291944


def all_filters(data_folder, run, N_cut=4, all_P_T=True, p_T_min=0, p_T_max=8) -> pd.DataFrame:
    t0 = time()
    events = de.read_root_file(data_folder, run=run)

    events = cut_events(events)
    df = de.muon_df(events)

    df = cut_tracks(df, N_cut=N_cut, all_P_T=all_P_T)
    print(f"Total time needed : {round(time() - t0, 2)}")
    return df


def cut_events(events):
    print("Cut nMuons [...] ")
    events = Cut.cut_nMuons(events)
    print("Cut CMUL [...] ")
    events = Cut.cut_CMUL(events)
    print("Cut zVtx [...] ")
    events = Cut.z_cut(events)

    return events


def cut_tracks(df, N_cut=4, all_P_T=True, p_T_min=0, p_T_max=8):
    df["eta"] = df.apply(lambda x: km.eta(x["Px"], x["Py"], x["Pz"]), axis=1)
    print("Cut eta [...] ")
    df = Cut.cut_eta(df)

    df["P"] = df.apply(lambda x: Cut.p_fc(math.sqrt(x["Px"] ** 2 + x["Py"] ** 2 + x["Pz"] ** 2), x['thetaAbs']), axis=1)

    df["DCA"] = df.apply(lambda x: math.sqrt(x["xDCA"] ** 2 + x["yDCA"] ** 2 + x["zDCA"] ** 2), axis=1)
    df['pDCA'] = df.P * df.DCA

    df['s_pxDCA'] = df.apply(lambda x: Cut.sigma_pxDCA(x['P'], x['thetaAbs'], N=N_cut), axis=1)

    print("Cut pDCA [...] ")
    df = Cut.cut_pDCA(df, N_cut)
    de.max_muons_pairs(df)

    if not all_P_T:
        df['p_T'] = df.apply(lambda x: km.p_T(x["Px"], x["Py"]))
        df = df[(df.p_T > p_T_min) & (df.p_T < p_T_max)]
        df = Cut.more_than_one_muon(df)

    return df


def M_inv(df):
    args = ['E', 'Px', 'Py', 'Pz', 'Charge']
    t0 = time()
    m_inv = []
    for idx, data in tqdm(df[args].groupby(level=0)):
        if len(data) == 2:
            if data.Charge.sum() == 0:
                y = km.y(data.E.sum(), data.Pz.sum())
                if Cut.y_cut(y):
                    tab = data.to_numpy()[:, 0:-1]
                    E1, E2 = tab[:, 0]
                    P1, P2 = tab[0, 1:], tab[1, 1:]
                    m_inv.append(km.inv_mass(E1, E2, P1, P2))
        else:
            tab = data.to_numpy()
            for c in combinations(np.arange(len(data)), 2):  # we try every combination of di-muons
                if tab[c, -1].sum() == 0:  # opposite charges
                    # r1, r2 = c
                    y = km.y(tab[c, 0].sum(), tab[c, 3].sum())
                    if Cut.y_cut(y):
                        E1, E2 = tab[c, 0]  # extraction of the energy
                        P1, P2 = tab[c, 1:-1]  # extraction of the impulsion
                        m_inv.append(km.inv_mass(E1, E2, P1, P2))

    print(f'Execution time : {round(time() - t0, 2)}')
    print(f'Total number of di-muons event : {len(m_inv)}')

    return m_inv


def plot_M_inv(M_inv):
    bins = np.linspace(1.5, 5, 36) # width = 0.1 GeV
    plt.figure(figsize=(12,8))
    plt.xlim(1.5,5)
    plt.semilogy()
    plt.xlabel("$m_{\mu\mu}$ (GeV)", fontweight='bold', fontsize=12)
    h = plt.hist(M_inv, bins=bins, range=[1.5,5], histtype='step', align='mid')


DF = all_filters(data_folder, run_number)
plot_M_inv(M_inv(DF))


