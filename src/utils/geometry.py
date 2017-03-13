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
    diff = v1 - v2

    return np.dot(diff, diff)


def delta_view_dir(v1, v2):
    """
    Computes the angle between two vectors
    :param v1: vector 1
    :param v2: vector 2
    :return: angle (radians)
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def degrees2radians(a):
    """
    Degrees to radians
    :param a: angle in degrees
    :return: angle in radians
    """
    return a * (2.0 * np.pi) / 360.0


def radians2degrees(a):
    """
    Radians to degrees
    :param a: angle in radians
    :return: angle in degrees
    """
    return a * 360.0 / (2.0 * np.pi)


def bundler_extract_viewdir(R):
    """
    Gets the view direction of a bundler rotation matrix
    :param R: 3x3 rotation matrix
    :return: direction vector
    """
    return - np.array([R[2][0], R[2][1], R[2][2]])


def bundler_extract_position(R, t):
    """
    Extracts the position matrix from bundler R,t
    :param R: 3x3 rotation matrix
    :param t: 3x1 translation matrix
    :return: 3x1 position matrix
    """
    return -np.matmul(R.transpose(), t)


def compute_SVD(A):
    """
    Computes SVD for A
    :param A: Matrix A
    :return: w, u, vt
    """
    return np.linalg.svd(A)


def project_point(P, point):
    """
    Projects a 3D point into 2D using P projection
    :param P: 3x4 projection matrix
    :param point: points vector
    :return: position vector
    """

    p2 = np.append(point, [1.0]).reshape((4, 1))

    projpoint = np.dot(P, p2)

    projpoint[0][0] /= projpoint[2][0]
    projpoint[1][0] /= projpoint[2][0]

    return projpoint.reshape(3)[:-1]
