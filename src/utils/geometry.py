import numpy as np
from numpy import dot, array
from numpy.linalg import svd
from math import sqrt

from src.c_package.pyc_geometry import pyc_euclidean128, pyc_euclidean2d, pyc_euclidean3d, pyc_project_point


def lowes_ratio_test(d1, d2, r_thrs):
    """
    Lowes ratio threshold
    :param d1: distance1
    :param d2: distance1
    :param r_thrs: threshold
    :return: Bool(d1 < r_thrs * d2)
    """
    return d1 < r_thrs * d2


def py_euclidean_distance(v1, v2):
    """
    Euclidean distance by numpy (non-squared)
    :param v1: vector1
    :param v2: vector2
    :return: euclidean distance
    """
    diff = v1 - v2

    return dot(diff, diff)


def c_euclidean_distance_128(v1, v2):
    """
    Squared euclidean distance for 128 vectors (written in C)
    :param v1: vector128
    :param v2: vector128
    :return: squared euclidean distance
    """
    return pyc_euclidean128(v1, v2)


def c_euclidean_distance_2d(v1, v2):
    """
    Squared euclidean distance for 2D vectors (written in C)
    :param v1: vector2D
    :param v2: vector2D
    :return: squared euclidean distance
    """
    return pyc_euclidean2d(v1, v2)


def c_euclidean_distance_3d(v1, v2):
    """
    Squared euclidean distance for 3D vectors (written in C)
    :param v1: vector3D
    :param v2: vector3D
    :return: squared euclidean distance
    """
    return pyc_euclidean3d(v1, v2)


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
    return svd(A)


def py_project_point(P, point):
    """
    Projects a 3D point into 2D using P projection
    :param P: 3x4 projection matrix
    :param point: points vector
    :return: position vector
    """

    num1  = P[0][0] * (point[0]) + P[0][1] * (point[1]) + P[0][2] * (point[2]) + P[0][3]
    num2  = P[1][0] * (point[0]) + P[1][1] * (point[1]) + P[1][2] * (point[2]) + P[1][3]
    denom = P[2][0] * (point[0]) + P[2][1] * (point[1]) + P[2][2] * (point[2]) + P[2][3]

    return array([num1 / denom, num2 / denom])


def c_project_point(P, point):
    """
    Projects a 3D point into 2D using P projection (written in C)
    :param P: 3x4 projection matrix
    :param point: points vector
    :return: position vector
    """

    p = pyc_project_point(P, point)

    return array([p[0], p[1]])


def norm2d(v):
    """
    Length for 2D vectors
    :param v: 2D vector
    :return: length 
    """
    return sqrt(v[0] * v[0] + v[1] * v[1])


def norm3d(v):
    """
    Length for 3D vectors
    :param v: 3D vector
    :return: length
    """
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
