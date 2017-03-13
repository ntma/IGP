import numpy as np


def read_generic_vocabulary_100K(filepath, norm_flag=False):

    try:
        f = open(filepath, "r")
    except IOError:
        print "Could not open " + str(filepath)
        return None

    words = []

    for i in range(100000):
        line = f.readline()[:-1].split()

        word = np.array(line, dtype=float)

        if norm_flag:
            word /= np.linalg.norm(word)

        words.append(word)

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

    kpts = np.empty((n_pts, 2), dtype=float)
    descs = []

    for i in range(n_pts):
        line = f.readline()[:-1].split(' ')

        x = float(line[0])
        y = float(line[1])
        #s = float(line[2])
        #o = float(line[3])

        line = f.readline()[:-1]
        if line[0] == ' ':
            line = line[1:]

        line = line.split(' ')

        desc = line

        for j in range(6):
            line = f.readline()[:-1]
            if line[0] == ' ':
                line = line[1:]
            line = line.split(' ')

            desc += line

        kpts[i][0] = x
        kpts[i][1] = y
        #kpts[i][2] = s
        #kpts[i][3] = o

        descs.append(desc)

    descs = np.array(descs, dtype=float)

    descs /= 512.0

    return kpts, descs
