import os
from itertools import combinations

import uproot
import numpy as np
import awkward as ak
import pandas as pd
import urllib
import pickle

from time import time
from scipy.special import binom
from tqdm import tqdm

"""
Functions written to extract the data from the root files and convert it into awkward array or pandas dataframe
"""


def read_root_file(data_folder, run=290222, runs_list=None, entry_stop=None, branch="eventsTree"):
    """
    This function reads the data file (.root) and converts it into a awkward Array. If multiple runs are specified,
    it returns a dictionary with every run.
    If the data file is not in the data folder, it download the file from the CERN Cloud.

    :param data_folder: path of the folder containing the folders of the runs
    :param runs_list: if True, returns a dictionary of arrays, one array per run, keys are runs' number
    :param run: run number
    :param entry_stop: max number of entries
    :param branch: branch of the tree of the root file wanted
    :return: array or dict of arrays
    """

    file_name = "AnalysisResults.root"
    t0 = time()
    if runs_list:
        # list_dict list_runs = os.listdir(data_folder)
        list_dict = {}
        for f in runs_list:
            file_dir = data_folder + "/" + f + "/"
            file = uproot.open(file_dir + file_name)
            events = file[branch]
            list_dict[f] = events.arrays(how="zip", entry_stop=entry_stop)
        print(f"Extraction took {round(time() - t0, 3)} s")
        return list_dict

    else:
        file_dir = data_folder + "/" + str(run) + "/"

        if str(run) not in os.listdir(data_folder):  # if the run file is not in the data folder
            ev = get_from_cloud(run, folder=data_folder + '/')
            return ev

        file = uproot.open(file_dir + file_name)
        events = file[branch]
        ev = events.arrays(how="zip", entry_stop=entry_stop)

        print(f"Size of the data file : {round(os.path.getsize(file_dir + file_name) / 1e6, 2)} Mo.")
        print(f"Extraction took {round(time() - t0, 1)} s.")
        print(f"Number of events : {len(ev)}.")
        return ev


def get_from_cloud(run, folder=''):
    """
    Download the data of the run from the CERN Cloud
    :param folder:
    :param run: run number
    :return:
    """
    url = f'https://cernbox.cern.ch/index.php/s/r7VFXonK39smzKP/download?path={run}/AnalysisResults.root '
    file_name = f'run{run}.data.root'
    if folder:
        os.mkdir(f'{folder}/{run}')
    urllib.request.urlretrieve(url, f'{folder}/{run}/AnalysisResults.root')
    file = uproot.open(file_name)
    events = file["eventsTree"]
    ev = events.arrays(how="zip")
    return ev


def muon_df(events: ak.Array, save_to_csv=False, path=None) -> pd.DataFrame:
    """
    Merge all the data on muons from an array object in one dataframe
    :param events: awkward array object
    :param save_to_csv: if True, the dataframe returned is saved into a csv file
    :param path: path for the csv file
    :return: Pandas dataframe with all the muons of the run
    """
    cols = ['E', 'Px', 'Py', 'Pz', 'Charge', 'thetaAbs', 'xDCA', 'yDCA', 'zDCA', 'matchedTrgThreshold']
    df = ak.to_pandas(events["Muon"])[cols]

    if save_to_csv:
        df.to_csv(path)

    print(f'Number of tracks : {len(df)}.')
    # max_muons_pairs(df)

    return df


def di_muons_dataframe(df):
    """
    Create a second table from the dataframe with all di-muons pairs possible (muons of opposite charges)
    :param df: dataframe with all the muons
    :return: dataframe where the columns are P1, P2, E1, E2, E
    """
    args = ['E', 'Px', 'Py', 'Pz', 'Charge']
    df_di_muons = pd.DataFrame()

    t0 = time()
    P1_list, P2_list, E1_list, E2_list = [], [], [], []
    index = []
    for idx, data in tqdm(df[args].groupby(level=0)):
        tab = data.to_numpy()
        for c in combinations(np.arange(len(data)), 2):  # we try every combination of di-muons
            E1, E2 = tab[c, 0]  # extraction of the energy
            P1, P2 = tab[c, 1:-1]  # extraction of the impulsion
            if tab[c, -1].sum() == 0:  # opposite charges
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


def max_muons_pairs(df):
    """
    Compute the total number of di-muons pairs candidates (possibly with the same charge)
    It gives and idea of the time complexity.
    :param df:
    :return: None
    """
    n_pairs = 0
    for _, data in df.groupby(level=0):
        n_pairs += binom(len(data), 2)

    print(f'Max number of possible muons pairs : {int(n_pairs)}')

    return None


def MC_muons_from_JPsi(gen_events: ak.Array, df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the muons data for all muons detected from a JPsi
    :param gen_events: awkward array with the generated events
    :param df_events: awkward array with the muons detected
    :return: dataframe
    """

    id_JPsi = 443

    # df_gen = ak.to_pandas(gen_events['Muon'])

    # we now have to add the information on the mother particle for each tracks
    arr = gen_events['Muon']['GenMotherPDGCode'][:, 0]
    PDGC_list = [arr[i] for i in df_events.index.get_level_values(0)]

    df_events['GenMotherPDGCode'] = PDGC_list
    df_events = df_events[df_events['GenMotherPDGCode'] == id_JPsi]

    return df_events


def read_dict_hist(filename):
    with open(filename, 'rb') as f:
        dict_hist = pickle.load(f)
    return dict_hist


def save_dict_hist(filename, dict_hist):
    with open(filename, 'wb') as f:
        pickle.dump(dict_hist, f)

    return None


def string_to_list(S):
    c_S = S.replace('[', '')
    c_S = c_S.replace(']', '')
    values = c_S.split()

    P = [float(v) for v in values]
    return P


def load_di_muon_from_csv(run_number, folder='Save/'):

    df_dm_loaded = pd.read_csv(f'{folder}{run_number}/{run_number}_dimuons.csv', index_col='Event id')
    df_dm_loaded['P1'] = df_dm_loaded['P1'].apply(string_to_list)
    df_dm_loaded['P2'] = df_dm_loaded['P2'].apply(string_to_list)

    return df_dm_loaded

