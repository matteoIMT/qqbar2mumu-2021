import awkward as ak
import numpy as np
import pandas as pd
import math

from tqdm import tqdm
from itertools import combinations


def z_cut(events: ak.Array, z_lim=10) -> ak.Array:
    """
    Remove the events where the z coordinate of the vertex is beyond z_lim (10 cm)
    :param events: ak.Array containing the events
    :param z_lim:
    :return: filtered array
    """
    ev_filtered = events[np.abs(events.zVtx) < z_lim]
    print(f"This cut rejects {round((1 - len(ev_filtered) / len(events)) * 100, 2)} % of the statistics")
    return ev_filtered


def cut_nMuons(events: ak.Array, nMuons=2):
    """
    Remove the events where the number of muons is less than nMuons
    :param events: ak.Array containing the events
    :param nMuons: minimum number of muons reconstructed by event
    :return: filtered array
    """
    ev_filtered = events[events.nMuons >= nMuons]
    print(f"This cut rejects {round((1 - len(ev_filtered) / len(events)) * 100, 2)} % of the statistics")
    return ev_filtered


def cut_CMUL(events: ak.Array):
    """

    :param events:
    :return:
    """
    ev_filtered = events[events.isCMUL == True]
    print(f"This cut rejects {round((1 - len(ev_filtered) / len(events)) * 100, 2)} % of the statistics")
    return ev_filtered


def radial_coord(x, y, z):
    """
    Compute the distance to the origin
    """
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


# filtering on the dataframe

def cut_eta(df, eta_min=-4.5, eta_max=-2.5):
    """

    :param df:
    :param eta_min:
    :param eta_max:
    :return:
    """
    # df["eta"] = df.apply(lambda x: km.eta(x["Px"], x["Py"], x["Pz"]), axis=1)
    df_f = df[(df["eta"] < -2.5) & (df["eta"] > -4)]
    # We only keep the tracks with at least two muons, so we remove rows with only one entry
    df_f = df_f[df_f.index.get_level_values(0).duplicated(keep=False)]
    print(f"This cut rejects {round((1 - df_f.shape[0] / df.shape[0]) * 100, 2)} % of the statistics")
    return df_f


def p_fc(P, thetaAbs: float):
    """
    Return the momentum at the first chamber touched of the spectrometer i.e without correction of the absorption
    :param P: 3D vector (numpy array)
    :param thetaAbs: angle in degrees
    :return: 3D vector (numpy array)
    """
    corr = -3 if thetaAbs < 3 else -2.4  # average correction due to MSCs

    return P + corr


def sigma_abs(thetaAbs):
    return 80. if thetaAbs < 3 else 54.


def sigma_p(P, thetaAbs, N=1, delta_p=0.0004):
    if N > 10 or N < 0:
        print("Wrong value of N")
    '''a = N * delta_p * P
    den = 1 - (a / (1 + a))
    return sigma_abs(thetaAbs) / den'''

    return sigma_abs(thetaAbs) * (1 + N * delta_p * P)


def sigma_theta(P, delta_theta=0.0005):
    return 535 * delta_theta * P


def sigma_pxDCA(P: float, thetaAbs: float, N=1) -> float:
    return math.sqrt(sigma_p(P, thetaAbs, N=N) ** 2 + sigma_theta(P) ** 2)


def DCA(x, y, z):
    return np.sqrt(x ** 2 + y ** 2, +z ** 2)


def cut_pDCA(df, N_s=4):
    df_f = df[(N_s * df.s_pxDCA - df.pDCA) > 0]
    # We only keep the tracks with at least two muons, so we remove rows with only one entry
    df_f = df_f[df_f.index.get_level_values(0).duplicated(keep=False)]

    print(f"This cut rejects {round((1 - df_f.shape[0] / df.shape[0]) * 100, 2)} % of the statistics")

    return df_f


def more_than_one_muon(df):
    df_f = df[df.index.get_level_values(0).duplicated(keep=False)]
    return df_f


def y_cut(y: float):
    return True if -4.5 < y < -2.5 else False


def df_muons_pairs(df: pd.DataFrame, save):
    """
    Could be optimized
    :param df: dataframe with all tracks
    :param save: if True, the dataframe is saved as a csv file
    :return: new data frame containing all pairs of di-muons in the domain of acceptance
    """
    df_pairs = pd.DataFrame()
    index = []
    for idx, data in tqdm(df[['E', 'Pz', 'Charge']].groupby(level=0)):
        sub_index = [i for i in range(data.shape[0])]
        for id_pair, c in enumerate(combinations(sub_index, 2)):
            index.append((idx, id_pair))
            S = pd.Series({'pairs': c})
            S = S.append(data.iloc[list(c)].sum())
            df_pairs = df_pairs.append(S, ignore_index=True)

    index = pd.MultiIndex.from_tuples(index, names=["entry", "subentry"])
    df_pairs.index = index

    df_pairs = df_pairs[df_pairs["Charge"] == 0]
    df_pairs = df_pairs[(df_pairs["y"] < -2.5) & (df_pairs["y"] > -4)]

    if save:
        df_pairs.to_csv()

    return df_pairs
