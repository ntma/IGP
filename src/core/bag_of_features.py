from pyflann import *


################################################
# Class to hold two vocabularies (fine/coarse) #
################################################
class BagOfFeatures:
    def __init__(self):
        # KD-Tree
        self.fine_bof = FLANN()

        # For kmeans indexing
        self.coarse_bof = FLANN()

    def create_fine_kdtree(self, words):
        """
        Creates a fine kd-tree index
        :param words: vocabulary centroids
        :return: params outputed by FLANN
        """

        return self.fine_bof.build_index(words, algorithm=1, trees=1, random_seed=0, log_level="info")

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

    def create_clusters(self, words, k=10):
        """
        Builds a kmeans index over input words
        :param words: input visual features
        :param k: branching factor
        :return: params outputed by FLANN
        """
        # Create words
        return self.coarse_bof.build_index(words, algorithm=2, branching=k)

    def search_fine(self, qdata, nn, chks, nc=1):
        """
        Searches for nearest neighbors in the fine vocabulary
        :param qdata: query features
        :param nn: number of NN
        :param chks: number of leaves to check
        :param nc: Number of cores to use (0=auto choose)
        :return: (indexes, distances)
        """

        result, dists = self.fine_bof.nn_index(qdata, num_neighbors=nn, checks=chks, cores=nc)

        return result, dists

    def get_parents_at_level_L(self, L):
        """
        Gets the parent id's at level L
        :param L: level to search
        :return: max_level_ids, cluster_ids
        """
        return self.coarse_bof.get_parents_at_level_L_double(L)
