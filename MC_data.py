import numpy as np
import pandas as pd
import awkward as ak

import Filter
import ProjectPackage.Kinematic as km
import ProjectPackage.DataExtraction as de
from ProjectPackage import Cut

from tqdm import tqdm
from time import time
from itertools import combinations


def MC_analysis(data_folder, run_number, save_csv=False, path=''):
    """

    :param data_folder:
    :param run_number:
    :return:
    """
    p_t_ranges = [(i, i + 1) for i in range(6)] + [(6, 8)]

    all_numbers = {}

    gen_events = de.read_root_file(data_folder, run_number, branch='genTree')
    events = de.read_root_file(data_folder, run_number)

    df_gen = ak.to_pandas(gen_events['Muon'])
    df_events = ak.to_pandas(events['Muon'])
    df_gen_dm = di_muons_dataframe_MC(df_gen)

    df_JPsi = de.MC_muons_from_JPsi(gen_events, df_events)
    df_MC_di_muons = de.di_muons_dataframe(df_JPsi)

    for p_range in p_t_ranges:
        all_numbers[p_range] = N_gen(df_gen_dm, p_range), N_rec(df_MC_di_muons, p_range)

    if save_csv:
        pd.DataFrame(all_numbers.values(), columns=['N_gen', 'N_rec']).to_csv(f'{path}{run_number}/MC_{run_number}_genrec.csv')

    return all_numbers


def N_gen(df_dm_gen, p_t_range: tuple) -> int:
    """
    Computes the number of JPsi generated with a transverse impulsion in the range p_t_range
    :param df_dm_gen: dataframe with the generated events
    :param p_t_range: tuple (p_t_min, p_t_max)
    :return:
    """
    # id_JPsi = 443
    p_min, p_max = p_t_range

    df_dm_gen['p_T'] = df_dm_gen.apply(lambda x: km.p_T_df(x['P1'], x['P2']), axis=1)

    N_g = df_dm_gen[(df_dm_gen['p_T'] > p_min) & (df_dm_gen['p_T'] < p_max)].shape[0]

    return N_g


def N_rec(df_MC_di_muons, p_t_range: tuple) -> int:
    df_MC_di_muons_f = Cut.cut_di_muons(df_MC_di_muons, all_P_T=False, p_T_range=p_t_range)  # JPsi in the p_t range

    N_r = df_MC_di_muons_f.shape[0]

    return N_r


def di_muons_dataframe_MC(df):
    """
    Create a table from the dataframe with all di-muons pairs possible from the generated events which created a JPsi
    :param df: dataframe of the generated events
    :return: dataframe where the columns are P1, P2, E1, E2, E
    """
    id_JPsi = 443
    df = df[df['GenMotherPDGCode'] == id_JPsi]

    args = ['GenE', 'GenPx', 'GenPy', 'GenPz']
    df_di_muons = pd.DataFrame()

    t0 = time()
    P1_list, P2_list, E1_list, E2_list = [], [], [], []
    index = []
    for idx, data in tqdm(df[args].groupby(level=0)):
        tab = data.to_numpy()
        for c in combinations(np.arange(len(data)), 2):  # we try every combination of di-muons
            E1, E2 = tab[c, 0]  # extraction of the energy
            P1, P2 = tab[c, 1:]  # extraction of the impulsion

            P1_list.append(P1)
            P2_list.append(P2)
            E1_list.append(E1)
            E2_list.append(E2)
            index.append((idx, c))

    df_di_muons["P1"] = P1_list
    df_di_muons["P2"] = P2_list
    df_di_muons["E1"] = E1_list
    df_di_muons["E2"] = E2_list
    df_di_muons["E"] = df_di_muons["E1"] + df_di_muons["E2"]

    index = pd.MultiIndex.from_tuples(index, names=['Event id', 'Muon id'])
    df_di_muons.index = index

    print(f'Execution time : {round(time() - t0, 2)}')

    return df_di_muons


def M_inv_hist_MC(data_folder, run_number):

    gen_events = de.read_root_file(data_folder, run_number, branch='genTree')
    events = de.read_root_file(data_folder, run_number)

    events, idx = Filter.cut_events(events, MC=True)  # filtering on the events
    df_ev_filtered = ak.to_pandas(events['Muon'])
    df_ev_filtered.index = df_ev_filtered.index.set_levels(idx, level=0)

    df_JPsi = de.MC_muons_from_JPsi(gen_events, df_ev_filtered)  # all muons from a JPsi event
    df_JPsi = Cut.more_than_one_muon(df_JPsi)

    df_JPsi_filtered = Filter.cut_tracks(df_JPsi)

    df_MC_di_muons_filtered = de.di_muons_dataframe(df_JPsi_filtered)

    all_hist_MC = Filter.hist_M_inv_PT(df_MC_di_muons_filtered)

    return all_hist_MC


'''data_folder = 'D:/Data_muons/dimuonData_LHC18mMC'
run_number = 290350
H = M_inv_hist_MC(data_folder, run_number)'''



