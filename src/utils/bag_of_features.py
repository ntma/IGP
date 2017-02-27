import time
from pyflann import *

################################################
# Class to hold two vocabularies (fine/coarse) #
################################################
class BagOfFeatures:
    def __init__(self):
        # KD-Tree
        self.fine_bof = FLANN()

        # Vocabulary Tree Nister2006
        self.coarse_bof = FLANN()

        # Runtime search knn
        self.rtknn = FLANN()

        self.params_fine = None
        self.params_coarse = None

        self.overall_time = 0.0

    def create_fine_kdtree(self, words):
        """
        Creates a fine kd-tree index
        :param words: vocabulary centroids
        :return: params outputed by FLANN
        """

        kdstart = time.time()
        self.params_fine = self.fine_bof.build_index(words, algorithm=1, trees=1, random_seed=0, log_level="info")
        kdend = time.time() - kdstart

        print "Build time: " + str(kdend)

        return self.params_fine

    def save_fine_index(self, filename):
        """
        Stores the fine index into a file
        :param filename:
        :return:
        """
        self.fine_bof.save_index(filename)

        return 0

    def load_fine_index(self, filename, words):
        """
        Loads the fine index from a file
        :param filename:
        :param words: vocabulary centroids associated to index
        :return:
        """
        self.fine_bof.load_index(filename, words)
        return 0

    def create_coarse_kdtree(self, words):
        """
        Creates the coarse vocabulary index
        :param words: vocabulary centroids
        :return: params outputed by FLANN
        """

        # params: 5=hkm branching, iterations, centers_init, cb_index
        hkmstart = time.time()
        self.params_coarse = self.coarse_bof.build_index(words, algorithm=2, trees=1, branching=10, leaf_max_size=100, max_iterations=100, log_level="info")
        hkmend = time.time() - hkmstart

        print "Build time: " + str(hkmend)
        print "Build params: "

        return self.params_coarse

    def save_coarse_index(self, filename):
        """
        Stored the coarse index into a file
        :param filename:
        :return:
        """
        self.coarse_bof.save_index(filename)

        return 0

    def load_coarse_index(self, filename, words):
        """
        Loads the coarse index from a file
        :param filename:
        :param words: vocabulary centroids
        :return:
        """
        self.coarse_bof.load_index(filename, words)
        return 0

    def create_clusters(self, words, branch_size=11, num_branches=10):
        """
        Clustering using k-means
        :param words: input visual features
        :param branch_size:
        :param num_branches: k branches
        :return: params outputed by FLANN
        """
        vv = FLANN()

        # Create words
        return vv.hierarchical_kmeans(words, branch_size, num_branches, max_iterations=1000, dtype=None, log_level="info")

    def search_fine(self, qdata, nn, chks):
        """
        Searches for nearest neighbors in the fine vocabulary
        :param qdata: query features
        :param nn: number of NN
        :param chks: number of leaves to check
        :return: (indexes, distances)
        """

        result, dists = self.fine_bof.nn_index(qdata, num_neighbors=nn, checks=chks, cores=1)

        return result, dists

    def search_coarse(self, qdata, nn, chks):
        """
        Searches for nearest neighbors in the coarse vocabulary
        :param qdata: query features
        :param nn: number of NN
        :param chks: number of leaves to check
        :return: (indexes, distances)
        """

        kdstart = time.time()
        result, dists = self.coarse_bof.nn_index(qdata, num_neighbors=nn, checks=chks, cores=0)
        kdend = time.time() - kdstart

        print "Search time: " + str(kdend)

        return result, dists

    def search_runtime_nn(self, qdesc, dbdescs, nn):
        """
        Linear search for the NN. Built in runtime.
        :param qdesc: query features
        :param dbdescs: db features
        :return: (indexes, distances)
        """

        t_start = time.time()
        res, dist = self.rtknn.nn(dbdescs, qdesc, nn, algorithm='linear', cores=1)
        self.overall_time += time.time() - t_start

        return res, dist

    def get_overall_time(self):
        """
        Returns the overall time spent by this module
        :return: time in floating seconds
        """
        return self.overall_time
