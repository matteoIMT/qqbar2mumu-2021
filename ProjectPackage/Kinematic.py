import uproot
import awkward as ak
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

"""
This library contains all the useful functions for the project. To use them, type import Project_library as pl
"""


def mag(px: float, py: float, pz: float):
    """Returns the norm of the 3-vector (px,py,pz)."""
    return math.sqrt(px * px + py * py + pz * pz)


def costheta(px: float, py: float, pz: float):
    """Returns the cos(theta) of the 3 vector (px,py,pz)."""
    p_tot = mag(px, py, pz)
    return 1.0 if p_tot == 0.0 else pz / p_tot


def eta(px: float, py: float, pz: float):
    """Returns the pseudo-rapidity of the 3 vector (px,py,pz)."""
    ct = costheta(px, py, pz)
    if pz == 0:
        return 0
    if pz > 0:
        return 10E10
    if ct * ct < 1:
        return -0.5 * math.log((1.0 - ct) / (1.0 + ct))

    else:
        return -10E20


def y(E, pz, c=1):
    """
    Return the rapidity y
    :param E: Energy
    :param pz:
    :param c: celerity of light
    :return:
    """
    return 0.5 * math.log((E + pz * c) / (E - pz * c))


def cos_theta(X, Y):
    """
    Return the cosine of the angle between two 3D vectors
    """
    unit_vector_1 = X / np.linalg.norm(X)
    unit_vector_2 = Y / np.linalg.norm(Y)

    return np.dot(unit_vector_1, unit_vector_2)


def inv_mass(E1: float, E2: float, P1, P2):
    """Returns the invariant mass from (p1+p2)^2"""
    ct12 = cos_theta(P1, P2)
    return math.sqrt(2 * E1 * E2 * (1 - ct12))


def p_T(Px, Py):
    """
    Compute the transverse impulsion
    :param Px:
    :param Py:
    :return:
    """
    return math.sqrt(Px**2 + Py**2)


def p_T_df(P1, P2):
    """
    Compute the transverse impulsion from the two di-muons produced
    :param P1:
    :param P2:
    :return:
    """
    Px, Py = P1[0] + P2[0], P1[1] + P2[1]
    return math.sqrt(Px**2 + Py**2)


