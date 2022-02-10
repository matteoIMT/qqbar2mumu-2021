import ProjectPackage.DataExtraction as de
import os, sys
import matplotlib.pyplot as plt
import Filter

data_folder = 'D:/Data_muons/dimuonData_LHC18m'
folder_saving = 'Save/'


# this class is made in order to disable the print function locally

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# runs_available = os.listdir(data_folder)

all_runs_available = os.listdir(folder_saving)
runs_list = all_runs_available[0:5]

# def all_hist_all_runs(folder_saving, runs_list):

'''p_t_ranges = [(i, i + 1) for i in range(6)] + [(6, 8)]

all_h = {}
for run in runs_list:
    print(f'\nRun : {run}')
    if str(run) in os.listdir(folder_saving) and len(os.listdir(f'{folder_saving}/{run}')) == 3:

        f = f'{folder_saving}{run}/{run}290469_histograms.pkl'
        h = de.read_dict_hist(f)
        if not all_h:
            all_h = h.copy()

        for r in p_t_ranges:
            all_h[r] = (all_h[r][0] + h[r][0], h[r][1])
'''

def complete_missing_hist(folder_saving):
    all_runs_available = os.listdir(folder_saving)

    runs_missing_hist = [int(r) for r in all_runs_available if len(os.listdir(f'{folder_saving}{r}')) == 2]

    for run in runs_missing_hist:
        print(f'\nRun : {run}')

        df_dm = de.load_di_muon_from_csv(run, folder=folder_saving)

        d = Filter.hist_M_inv_PT(df_dm)

        de.save_dict_hist(f'{folder_saving}{run}/{run}_histograms.pkl', d)


