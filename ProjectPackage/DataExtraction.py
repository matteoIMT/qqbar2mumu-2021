import os
import uproot
import numpy as np
import awkward as ak
import pandas as pd

from time import time
from scipy.special import binom

"""
Functions written to extract the data from the root files and convert it into awkward array or pandas dataframe
"""


def from_root_to_event(data_folder, all_runs=False, run=290222):
    """
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
    max_muons_pairs(df)

    return df


def max_muons_pairs(df):
    n_pairs = 0
    for _, data in df.groupby(level=0):
        n_pairs += binom(len(data), 2)

    print(f'Max number of possible muons pairs : {int(n_pairs)}')

    return None

