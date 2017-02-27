import numpy as np
import struct


def read_flickr_vocabulary(filepath, n_centroids):
    """
    Reader for the Holidays, flickr visual vocabulary

    :param filepath: file path
    :param n_centroids: number of vocabulary centroids
    :return: numpy array with the vocabulary
    """
    try:
        f = open(filepath, "rb")
    except IOError:
        print "Could not open " + str(filepath)
        return None, None

    words = []

    for i in range(0, n_centroids):
        b = f.read(4)

        desc_size = struct.unpack('i', b)[0]

        desc_buffer = f.read(4 * desc_size)

        desc128f_list = struct.unpack('128f', desc_buffer[:])
        desc128f_array = np.array(desc128f_list, dtype=float)

        # Values do not seem normilized
        desc128f_array /= np.linalg.norm(desc128f_array)

        words.append(desc128f_array)

    f.close()

    words = np.array(words)

    return words


def read_sift_file(filepath):

    """
    Reader for bundler SIFT format

    :param filepath: file path
    :return: kpts, descs - keypoints and descriptors read
    """

    try:
        f = open(filepath, "r")
    except IOError:
        print "Could not open " + str(filepath)
        return None, None

    header = f.readline()[:-1].split(' ')

    n_pts = int(header[0])
    desc_size = int(header[1])

    kpts = [[0.0, 0.0] for i in range(n_pts)]
    descs = []

    for i in range(n_pts):
        line = f.readline()[:-1].split(' ')

        x = float(line[0])
        y = float(line[1])
        #s = float(line[2])
        #o = float(line[3])

        line = f.readline()[1:-1].split(' ')
        desc = np.array(line, dtype=float)

        for j in range(6):
            line = f.readline()[1:-1].split(' ')
            g = np.array(line, dtype=float)
            desc = np.concatenate((desc, g))

        desc /= 512.0

        kpts[i][0] = x
        kpts[i][1] = y
        #kpts[i][2] = s
        #kpts[i][3] = o

        descs.append(desc)

    descs = np.array(descs)

    return kpts, descs


def read_query_list(filepath, n_queries):
    """
    Reader for the query file paths
    :param filepath: file path
    :param n_queries: number of queries to expect
    :return: dictionary[filename] = ground truth 3x1 numpy array
    """

    try:
        f = open(filepath, "r")
    except IOError:
        print "Could not open " + str(filepath)
        return None

    query_list = dict()

    for i in range(n_queries):
        line = f.readline()[:-1].split(' ')

        filename = line[0]

        C1 = float(line[-3])
        C2 = float(line[-2])
        C3 = float(line[-1])

        C = np.array([[C1], [C2], [C3]])

        query_list[filename] = C

    f.close()

    return query_list

