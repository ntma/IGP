# Import the base class for this extended module
import pgres_wrapper as pgw

# Import array converters TODO: remove this in future releases
from src.utils.converters import nparray2valuesstring, nparray2string


######################
# PostgreSQL Queries #
######################
class PGQueries(pgw.PGWrapper):

    def __init__(self):
        super(self.__class__, self).__init__()

    ###################
    # Runtime queries #
    ###################

    def sort_quantized_words(self, words_ids):
        """
        Sorts initial set of fine words by their associated processing cost.
        Also builds the right structure to insert into the priority queue.
        :param words_ids: quantized word ids
        :return: tuple list (search_cost, tiebreaker, word_id, descriptor_id, matched_3d, mode)
        """

        self.execute_query("""WITH input_wids(wid, in_order) AS (
                                SELECT wid, row_number() over() in_order
                                FROM (VALUES %s) as t(wid)
                             ) SELECT get_searchcost_by_wid(wid) as search_cost, 0::int, wid, in_order - 1, 0::int, 0::int
                             FROM input_wids
                             ORDER BY search_cost ASC;""" % words_ids)

        return self.fetch_all()

    def search_two_nn(self, descriptor, word_id):
        """
        Computes nearest neighbors in descriptor space
        :param descriptor: query descriptor
        :param word_id:
        :return:
        """

        self.execute_query_params("""SELECT pt_id, distance
                                     FROM get_2nn_linear(%s, %s);""", (descriptor, word_id))

        return self.fetch_all()

    def search_3d_nn(self, pt_id, n_neighbors):
        """
        Computes the nearest neighbors in 3d space
        :param pt_id: 3d point id
        :param n_neighbors: number of nearest neighbors to find
        :return: tuple list (point_id, list coarse words)
        """

        self.execute_query_params("""SELECT *
                                     FROM get_3d_nn_filtered(%s, %s)""", (pt_id, n_neighbors))

        return self.fetch_all()

    def get_3d_descriptors_from_id(self, pt3d_id, coarse_words_list):
        """
        Get the descriptors associated to a 3D point from its id and
        related coarse words
        :param pt3d_id: 3D point id
        :param coarse_words_list: coarse words list
        :return: list of uchar descriptors
        """

        self.execute_query_params("""SELECT get_3d_descriptor(%s, %s);""", (pt3d_id, coarse_words_list))

        return self.fetch_all()

    def filter_matches_by_visibility(self, matched_3d_ids, min_n_set):
        """
        Filters a computed set of correspondences by camera visibility to
        the 3D points
        :param matched_3d_ids: matched 3D points ids
        :param min_n_set: required minimum number of points in a set
        :return: list of point id sets
        """

        self.execute_query("""SELECT pt_list
                              FROM (
                                SELECT array_agg(pt3d_id) as pt_list, cluster_id, count(cluster_id) as c
                                FROM visibility_graph
                                WHERE pt3d_id IN %s
                                GROUP BY cluster_id
                                ORDER BY c DESC) x
                              WHERE x.c > %s;""" % (nparray2string(matched_3d_ids), min_n_set))

        return self.fetch_all()

    def get_xyz_from_ids(self, computed_set):
        """
        Get the XYZ position from 3D points ids
        :param computed_set: list of 3D points ids
        :return: list of tuples (x,y,z)
        """

        self.execute_query("""WITH inpoints AS (
                                SELECT id, row_number() over() rn
                                FROM (VALUES %s) as t(id)
                            ) SELECT x, y, z
                            FROM inpoints, points3d
                            WHERE points3d.id = inpoints.id
                            ORDER BY rn ASC;""" % nparray2valuesstring(computed_set))

        return self.fetch_all()
