import pandas as pd


def a(data_folder, NumOfRun):
    """

    :param data_folder: folder where the csv files are stored
    :param NumOfRun: run number
    :return: scaler (a in N_MB = a * N_muons)
    """

    data = pd.read_csv(f"{data_folder}counters.offline.csv")
    data2 = pd.read_csv(f"{data_folder}counters.online.csv")
    i, j = 0, 0
    b, b2 = 0, 0
    while i < len(data["run"]):
        if data["run"][i] == NumOfRun:
            b = i
        i += 1

    while j < len(data2["run"]):
        if data2["run"][j] == NumOfRun:
            b2 = j
        j += 1

    '''    for i in range(len(data["run"])):
        if data["run"][i] == NumOfRun:
            b = i

    for j in range(len(data2["run"])):
        if data2["run"][j] == NumOfRun:
            b2 = j'''

    scaler = (data2["cint7l0b"][b2] / data2["cmul7l0b"][b2] + (data["cint7all"][b] / data["cint7all&0msl"][b]) * (
                data["cmsl7all"][b] / data["cmsl7all&0mul"][b]) + (data["cint7ps"][b] / data["cint7all&0msl"][b]) * (
                          data["cmsl7ps"][b] / data["cmsl7all&0mul"][b])) / 3

    return scaler

