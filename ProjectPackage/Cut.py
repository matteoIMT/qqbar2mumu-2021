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

def cut_eta(df, eta_min=-4, eta_max=-2.5):
    """
    Cut on the pseudo-rapidity in the range (eta_min , eta_max)
    :param df:
    :param eta_min:
    :param eta_max:
    :return: filtered df
    """
    # df["eta"] = df.apply(lambda x: km.eta(x["Px"], x["Py"], x["Pz"]), axis=1)
    df_f = df[(df["eta"] < eta_max) & (df["eta"] > eta_min)]
    # We only keep the tracks with at least two muons, so we remove rows with only one entry
    df_f = more_than_one_muon(df_f)
    print(f"This cut rejects {round((1 - df_f.shape[0] / df.shape[0]) * 100, 2)} % of the statistics")
    return df_f


def p_fc(P, thetaAbs: float):
    """
    Return the momentum at the first chamber touched of the spectrometer i.e without correction of the absorptions
    :param P: norm of the impulsion
    :param thetaAbs: angle in degrees
    :return: 3D vector (numpy array
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
    df_f = more_than_one_muon(df_f)

    print(f"This cut rejects {round((1 - df_f.shape[0] / df.shape[0]) * 100, 2)} % of the statistics")

    return df_f


def cut_trigger(df):
    """

    :param df:
    :return:
    """
    df_f = df[df["matchedTrgThreshold"] == 2]
    df_f = more_than_one_muon(df_f)
    print(f"This cut rejects {round((1 - df_f.shape[0] / df.shape[0]) * 100, 2)} % of the statistics")

    return df_f


def more_than_one_muon(df):
    """
    Removed the events with only one muon (we only care about di-muons)
    :param df: dataframe of the muons
    :return: df with at least two tracks
    """
    df_f = df[df.index.get_level_values(0).duplicated(keep=False)]
    return df_f
