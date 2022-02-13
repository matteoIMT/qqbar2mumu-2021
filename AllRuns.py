import pickle
import os
import shutil
import urllib
import sys

import MC_data
import ProjectPackage.DataExtraction as de
import Filter


# __________________________________________________________________________________________________________

# this class is made in order to disable the print function locally

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# __________________________________________________________________________________________________________

def download_all_runs(data_folder, runs_list):
    """
    Script to download the files containing the data of the runs from the CERN's cloud to the specified folder.
    Does not work for the MC data (the MC data can be load in on once)
    :param data_folder: folder where to store the data
    :param runs_list: the list containing the runs' number to download
    """
    for i, run in enumerate(runs_list):
        print(f'Run : {run}')

        if str(run) not in os.listdir(data_folder):
            url = f'https://cernbox.cern.ch/index.php/s/r7VFXonK39smzKP/download?path={run}/AnalysisResults.root'

            try:
                os.mkdir(f'{data_folder}/{run}')
                urllib.request.urlretrieve(url, f'{data_folder}/{run}/AnalysisResults.root')

            except FileNotFoundError:
                print('This file is empty')

            else:
                print('File downloaded. \n')

            finally:
                continue
        else:
            print('File already downloaded. \n')

        if i % 10 == 0:
            print(f'{len(runs_list) - i} runs remaining. \n')

    return None


def load_compute_and_save(folder_saving, data_folder, runs_list):
    for i, run in enumerate(runs_list):
        print(f'\nRun : {run}')
        if str(run) not in os.listdir(folder_saving):

            print('Reading root file [...] \n')
            events = de.read_root_file(data_folder, run)

            os.mkdir(f'{folder_saving}{run}')  # create a directory for that run
            print('\n Filtering tracks [...] \n')
            with HiddenPrints():
                df = Filter.all_filters_muons(events)
            df.to_csv(f'{folder_saving}{run}/{run}_tracks.csv', index=False)

            print('\n Computing muons pairs [...] \n')
            with HiddenPrints():
                df_dm = de.di_muons_dataframe(df)
            df_dm.to_csv(f'{folder_saving}{run}/{run}_dimuons.csv', sep=',')

            print('Computing histograms [...] \n')
            with HiddenPrints():
                all_hist = Filter.hist_M_inv_PT(df_dm)
            de.save_dict_hist(f'{folder_saving}{run}/{run}_histograms.pkl', all_hist)

        else:
            print('Folder already created. \n')

        if i % 10==0:
            print(f'{len(runs_list) - i} runs remainging \n')


def all_hist_all_runs(folder_saving, runs_list):
    p_t_ranges = [(i, i + 1) for i in range(6)] + [(6, 8)]

    all_h = {}
    for run in runs_list:
        print(f'\nRun : {run}')
        if str(run) in os.listdir(folder_saving) and len(os.listdir(f'{folder_saving}/{run}')) == 3:

            f = f'{folder_saving}{run}/{run}_histograms.pkl'
            h = de.read_dict_hist(f)
            if not all_h:
                all_h = h.copy()

            for r in p_t_ranges:
                all_h[r] = (all_h[r][0] + h[r][0], h[r][1])

    return all_h


def load_compute_and_save_MC(folder_saving, data_folder, runs_list):
    for run in runs_list:
        print(f'\nRun : {run}')
        if str(run) not in os.listdir(folder_saving):
            os.mkdir(f'{folder_saving}{run}')  # create a directory for that run
            print('Computing numbers [...] \n')
            with HiddenPrints():
                d = MC_data.MC_analysis(data_folder, run, save_csv=True, path=folder_saving)

            print('Computing histograms [...] \n')
            with HiddenPrints():
                h = MC_data.M_inv_hist_MC(data_folder, run)
                de.save_dict_hist(f'{folder_saving}{run}/{run}_histograms.pkl', h)
        else:
            print('Folder already created. \n')

    return None


def CMUL_events_for_all_runs(data_folder, run_list, d=None):
    if d is None:
        d = {}
        run_remaining = run_list
    else:
        run_remaining = [r for r in run_list if r not in d.keys()]

    for i, run in enumerate(run_remaining):
        print(f'Run : {run} \n')
        ev = de.read_root_file(data_folder, run=run)

        ev = ev[ev.isCMUL == True]
        n = len(ev)
        d[run] = n

        print(f'{n} \n')

        if i % 10 == 0:
            print(f'{len(run_remaining) - i} runs remainging \n')

    with open('All_CMUL.pkl', 'wb') as f:
        pickle.dump(d, f)

    return d


DataFolder = 'D:/Data_muons/dimuonData_LHC18m'
SavingFolder = 'Save/'

DataFolderMC = 'D:/Data_muons/dimuonData_LHC18mMC'
SavingFolderMC = 'Save_MC/'

runs_MC = os.listdir(DataFolderMC)
runs = os.listdir(DataFolder)

'''all_runs_available = os.listdir(DataFolder)

runs = all_runs_available[60:70]
load_compute_and_save(Saving_folder, DataFolder, all_runs_available)

load_compute_and_save(SavingFolder, DataFolder, runs)

'''

for f in os.listdir(SavingFolderMC):
    if f.startswith('MC'):
        run_number = f.split('_')[1]
        shutil.move(SavingFolderMC + f, f'{SavingFolderMC}{run_number}')
