import numpy as np
import array

from psycopg2 import Binary


def nparray2string(a):
    """
    Converts a numpy array to string
    :param a: numpy array
    :return: string
    """
    return "(" + str(a)[1:-1] + ")"


def nparray2valuesstring(a):
    """
    Converts a list of values into a string of values (postgres style)
    :param a: input list
    :return: string (i.e. (1st elem), (2nd elem), etc...)
    """
    output = ""
    for v in a:
        output += "(" + str(v) + "),"
    output = output[:-1]

    return output


def nparray2pgarray(a):
    """
    Converts a numpy array to postgres array string
    :param a: numpy array
    :return: pg array string
    """
    return "ARRAY%s" % str(a.tolist())


def bytestring2nparray(a, type):
    """
    Converts binary string to numpy array
    :param b: binary string
    :return: numpy array
    """

    if type:
        return np.fromstring(a, dtype=np.uint8) / 512.0
    else:
        return np.fromstring(a, dtype=np.uint8)


def npint2pgbyte(a):
    """
    Converts a 128float array to uchar (escaped bytes)
    :param nparray: 128float
    :return: binary string
    """

    l = a.astype(dtype=np.int32).tolist()

    b = array.array('B', l).tostring()

    binstring = str(Binary(b))[1:-8]
    binstring = binstring.replace("''", "\'")

    return binstring


def npfloat2pgbyte(a):
    """
    Converts a 128float array to uchar (escaped bytes)
    :param nparray: 128float
    :return: binary string
    """

    a = np.floor(a * 512.0 + 0.5)

    l = a.astype(dtype=int).tolist()

    b = array.array('B', l).tostring()

    binstring = str(Binary(b))[1:-8]
    binstring = binstring.replace("''", "\'")

    return binstring


def nparr2bytearr(a):
    return array.array('B', np.floor(a * 512.0 + 0.5).astype(dtype=np.int32).tolist())


def bytearr2nparr(a):
    return np.array(a.tolist(), dtype=np.float) / 512.0
