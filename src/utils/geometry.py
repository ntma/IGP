import numpy as np


def lowes_ratio_test(d1, d2, r_thrs):
    """
    Lowes ratio threshold
    :param d1: distance1
    :param d2: distance1
    :param r_thrs: threshold
    :return: Bool(d1 < r_thrs * d2)
    """
    return d1 < r_thrs * d2

def euclidean_distance(v1, v2):
    """
    Euclidean distance by numpy (non-squared)
    :param v1: vector1
    :param v2: vector2
    :return: euclidean distance
    """
    return np.linalg.norm(v1 - v2)
