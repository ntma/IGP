import numpy as np


def read_generic_vocabulary_100K(filepath, norm_flag=False):

    try:
        f = open(filepath, "r")
    except IOError:
        print "Could not open " + str(filepath)
        return None

    words = np.zeros((100000, 128), dtype=float)

    for i, line in enumerate(f):
        words[i] = np.asarray(line.split(), dtype=float)
    f.close()

    if norm_flag:
        words /= 512.0

    return words


def read_sift_file(filepath, normalize=True):

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

    if normalize:
        ttype = float
    else:
        ttype = np.uint8

    header = f.readline()[:-1].split(' ')

    n_pts = int(header[0])
    desc_size = int(header[1])

    kpts = np.empty((n_pts, 2), dtype=float)
    descs = np.empty((n_pts, 128), dtype=ttype)

    for i in range(n_pts):
        line = f.readline()[:-1].split(' ')

        x = float(line[0])
        y = float(line[1])
        #s = float(line[2])
        #o = float(line[3])

        desc = []
        for j in xrange(7):
            desc += f.readline().split()

        descs[i] = np.asarray(desc, dtype=ttype)

        kpts[i][0] = x
        kpts[i][1] = y

    if normalize:
        descs /= 512.0

    return kpts, descs
