import os
from itertools import combinations

import uproot
import numpy as np
import awkward as ak
import pandas as pd

from time import time
from scipy.special import binom
from tqdm import tqdm

"""
Functions written to extract the data from the root files and convert it into awkward array or pandas dataframe
"""


def from_root_to_event(data_folder, all_runs=False, run=290222):
    """
    not sure i it is uesful
    Return an awkward event object from the root file
    :param data_folder: folder containing the folders of the runs
    :param all_runs: if True, returns a dictionary of event object, one array per run, keys are runs' number
    :param run: id of the run wanted among all runs
    :return: event object or dict of event object
    """

    file_name = "AnalysisResults.root"

    if all_runs:
        list_runs = os.listdir(data_folder)
        list_dict = {}
        for f in list_runs:
            file_dir = data_folder + "/" + f + "/"
            file = uproot.open(file_dir + file_name)
            events = file["eventsTree"]
            list_dict[f] = events

    else:
        file_dir = data_folder + "/" + str(run) + "/"
        file = uproot.open(file_dir + file_name)
        events = file["eventsTree"]
        return events


def read_root_file(data_folder, all_runs=False, run=290222, entry_stop=None):
    """
    Return an awkward array type from the root file
    :param entry_stop:
    :param data_folder: folder containing the folders of the runs
    :param all_runs: if True, returns a dictionary of arrays, one array per run, keys are runs' number
    :param run: run number of the run wanted among all runs
    :return: array or dict of arrays
    """

    # !curl "https://cernbox.cern.ch/index.php/s/r7VFXonK39smzKP/download?path=290223/AnalysisResults.root" >
    # run290223.data.root

    file_name = "AnalysisResults.root"
    t0 = time()
    if all_runs:
        list_runs = os.listdir(data_folder)
        list_dict = {}
        for f in list_runs:
            file_dir = data_folder + "/" + f + "/"
            file = uproot.open(file_dir + file_name)
            events = file["eventsTree"]
            list_dict[f] = events.arrays(how="zip", entry_stop=entry_stop)
        print(f"Extraction took {round(time() - t0, 3)} s")
        return list_dict

    else:
        file_dir = data_folder + "/" + str(run) + "/"
        if file_dir not in os.listdir(data_folder):
            pass
        file = uproot.open(file_dir + file_name)
        events = file["eventsTree"]
        ev = events.arrays(how="zip", entry_stop=entry_stop)

        print(f"Size of the data file : {round(os.path.getsize(file_dir + file_name) / 1e6, 2)} Mo.")
        print(f"Extraction took {round(time() - t0, 1)} s.")
        print(f"Number of events : {len(ev)}.")
        return ev


def muon_df(events: ak.Array, save_to_csv=False, path=None) -> pd.DataFrame:
    """
    Merge all the data on muons from an array object in one dataframe
    :param events: awkward array object
    :param save_to_csv: if True, the dataframe returned is saved into a csv file
    :param path: path for the csv file
    :return: Pandas dataframe with all the muons of the run
    """
    cols = ['E', 'Px', 'Py', 'Pz', 'Charge', 'thetaAbs', 'xDCA', 'yDCA', 'zDCA']
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

