"""
This file gathers the filtering procedures. It is made to be called in the main file for one run.
To use it, use the command : all_filters(data_folder, rub)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from time import time

import ProjectPackage.DataExtraction as de
import ProjectPackage.Kinematic as km
from ProjectPackage import Cut

'''data_folder = 'D:/Data_muons/dimuonData_LHC18m'
run_number = 291944'''


def all_filters_muons(events, N_cut=5) -> tuple:
    """
    Given a run, it extracts the data and applies the filters on the events and then on muons tracks. It returns a
    DataFrame with also the information on di-muons pairs.
    :param events: awkward array containing the events of a run
    :param N_cut: number of sigma for the pDCA cut
    :return:
    """
    t0 = time()

    # Cuts on the events
    events = cut_events(events)

    # Muons dataframe
    df = de.muon_df(events)

    # cut on the tracks
    df = cut_tracks(df, N_cut=N_cut)

    print(f"\nTotal time needed : {round(time() - t0, 2)} s.")
    return df


def all_filters_di_muons(df_di_muons, y_range=(-2.5, -4.5), all_P_T=True, p_T_range=(0, 8)):

    # cut on the di-muons pairs (rapidity and transversal impulsion)
    df_di_muons = cut_di_muons(df_di_muons, y_range=y_range, all_P_T=all_P_T, p_T_range=p_T_range)

    return df_di_muons


def cut_events(events):
    """
    Cuts on the events :
        - N muons
        - CMUL
        - z coordinate of vertex
    :param events:
    :return: events filtered
    """

    print("\nCut nMuons [...] \n ")
    events = Cut.cut_nMuons(events)

    print("\nCut CMUL [...] \n ")
    events = Cut.cut_CMUL(events)

    print("\nCut zVtx [...] \n ")
    events = Cut.z_cut(events)

    return events


def cut_tracks(df, N_cut=5):
    """

    :param df:
    :param N_cut: nu
    :return:
    """
    # Computing the pseudo-rapidity of each track and cut
    df["eta"] = df.apply(lambda x: km.eta(x["Px"], x["Py"], x["Pz"]), axis=1)
    print("\nCut eta [...] ")
    df = Cut.cut_eta(df)

    # cut on the pDCA
    df["P"] = df.apply(lambda x: Cut.p_fc(math.sqrt(x["Px"] ** 2 + x["Py"] ** 2 + x["Pz"] ** 2), x['thetaAbs']), axis=1)
    df["DCA"] = df.apply(lambda x: math.sqrt(x["xDCA"] ** 2 + x["yDCA"] ** 2 + x["zDCA"] ** 2), axis=1)
    df['pDCA'] = df.P * df.DCA
    df['s_pxDCA'] = df.apply(lambda x: Cut.sigma_pxDCA(x['P'], x['thetaAbs'], N=N_cut), axis=1)

    print("\nCut pDCA [...] \n ")
    df = Cut.cut_pDCA(df, N_cut)

    de.max_muons_pairs(df)

    return df


def cut_di_muons(df_muons, y_range=(-2.5, -4.), all_P_T=True, p_T_range=(0, 8)):
    """

    :param df_muons:
    :param all_P_T: if True, no cut on the transverse impulsion of the di-muons pair is applied
    :param y_range: range on the rapidity
    :param p_T_range: range of transverse impulsion for the di-muon pair
    :return: df with the di-muons pairs
    """
    df_muons['y'] = df_muons.apply(lambda x: km.y(x['E'], x['P1'][-1] + x['P2'][-1]), axis=1)
    y_max, y_min = y_range

    if not all_P_T:
        p_min, p_max = p_T_range
        df_muons['p_T'] = df_muons.apply(lambda x: km.p_T_df(x['P1'], x['P2']), axis=1)
        df_muons = df_muons[(df_muons['p_T'] > p_min) & (df_muons['p_T'] < p_max)]

    df_muons_f = df_muons[(df_muons['y'] > y_min) & (df_muons['y'] < y_max)]

    print(f"This cut rejects {round((1 - df_muons_f.shape[0] / df_muons.shape[0]) * 100, 2)} % of the statistics.")

    print(f'\nNumber of di-muons pairs : {df_muons.shape[0]}')

    return df_muons_f


def plot_M_inv(M_inv_list, run_number):
    bins = np.linspace(1.5, 5, 36)  # width = 0.1 GeV
    plt.figure(figsize=(12, 8))
    plt.xlim(1.5, 5)
    plt.semilogy()
    plt.xlabel(r"$m_{\mu\mu}$ (GeV)", fontweight='bold', fontsize=12)
    h = plt.hist(M_inv_list, bins=bins, range=[1.5, 5], histtype='step', align='mid')

    plt.title(f'Run number {run_number}', fontsize=15, fontweight='bold')
    plt.ylabel('Counts per 0.1 GeV', fontsize=12, fontweight='bold')

    return h







'''args = ['E', 'Px', 'Py', 'Pz', 'Charge']
P_cols = ['Px', 'Py', 'Pz']'''

'''d = dict(Counter(DF.index.get_level_values(0)))
idx = {k: [c for c in combinations([i for i in range(v)], 2)] for k, v in d.items()}

DF[args].iloc[0].to_numpy()

for id, data in DF.groupby(level=0)['Px']:
    pass

events = de.read_root_file(data_folder, run=run_number)'''

# df_di_muons = di_muons_dataframe(DF)



