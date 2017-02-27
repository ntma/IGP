# Python imports
import logging as lg
import numpy as np
import time
from collections import defaultdict

# IGP imports
from utils.bag_of_features import BagOfFeatures
from utils.pgres_wrapper import PGWrapper
from utils.priority_queue import PriorityQueue
from utils.geometry import lowes_ratio_test
from utils.io_datasets import read_flickr_vocabulary, read_sift_file

# 3rdparty imports
from thirdparty.sprt_ransac_6ptdlt import SPRTRANSACDLT

################################################################
### Ugly bulk queries that need to be optimized and isolated ###
################################################################
pgquery_fine_matches = "SELECT s_cost, 0::int, wid, en::int, 0::int, 0::int " \
                       "FROM(SELECT wid, s_cost, row_number() over() en " \
                       "FROM vv_fine_costs WHERE wid IN %s ORDER BY s_cost ASC) sub " \
                       "WHERE s_cost > 0;"

pgquery_3d_nn = "WITH nn3d(id) AS (" \
                    "SELECT id, wid_coarse " \
                    "FROM ilist, " \
                        "get_3d_nn_clustersets('%s'::geometry, %i) " \
                    "WHERE id = ilist.pt3d_id AND cluster_id IN (SELECT get_vizclusters_from_3dpt(%i))" \
                ") SELECT * FROM nn3d " \
                "GROUP BY id, wid_coarse;"


pgquery_3dxyz = "SELECT x, y, z " \
                "FROM pcl " \
                "WHERE id IN %s;"

pgquery_filter_matches = "SELECT pt_list " \
                         "FROM " \
                         "(SELECT array_agg(pt3d_id) as pt_list, cluster_id, count(cluster_id) as c " \
                         "FROM viz_graph " \
                         "WHERE pt3d_id IN %s " \
                         "GROUP BY cluster_id " \
                         "ORDER BY c DESC) x " \
                         "WHERE x.c > %i;"
##############
# References #
##############

# [1] Sattler, T., Leibe, B., & Kobbelt, L. (2012).
#     Improving image-based localization by active correspondence search.
#     Lecture Notes in Computer Science,
#     7572 LNCS(PART 1),752-765.
#     http://doi.org/10.1007/978-3-642-33718-5_54
# [2] Sattler, T., Leibe, B., & Kobbelt, L. (2016).
#     Efficient & Effective Prioritized Matching for Large-Scale Image-Based Localization.
#     Ieee Transactions on Pattern Analysis and Machine Intelligence, X(1).
#     http://doi.org/10.1109/TPAMI.2016.2611662
# [3] https://www.graphics.rwth-aachen.de/software/image-localization

##############################################
# Class to hold the lookup to coarse matches #
##############################################
class CoarseMatch:
    __slots__ = ['q_ids', 'cost']

    def __init__(self, q):
        self.q_ids = [q]
        self.cost = 1

    def add_element(self, q):
        self.q_ids.append(q)
        self.cost += 1



###########################################################
# Class to pose estimate photographs based on resarch [2] #
###########################################################
class IGP:
    def __init__(self):
        # Support classes required to pose estimate new photographs
        self.pgman = PGWrapper()          # Middle man. to Postgres
        self.pq = PriorityQueue()     # Priority Queue
        self.bof = BagOfFeatures()    # Bag of Features
        self.p6pdlt = SPRTRANSACDLT() # SPRT-RANSAC with p6p DLT (Adapted from ACG-Localizer [1] src available in [3])

        # EEILR parameters [1]
        self.Nt = 100             # Number of points to search (VPS+AS)
        self.N3D = 200            # Number of 3D nearest neighbors
        self.R = 0.2              # Minimum threshold for ransac
        self.min_inliers = 6      # Minimum inliers required to consider a pose
        self.r_thrs_2D3D = 0.7    # Threshold for 2D-3D Lowes ratio test
        self.r_thrs_3D2D = 0.6    # Threshold for 3D-2D Lowes ration test
        self.L_checks = 10        # Number of leafs to check in fine vocabulary

        self.lookup_table = None  # Lookup table for the coarse vocabulary

    def initialize_ilr(self):
        """
        Initiales the ILR module.
        :return:
        """

        lg.info("[Init] Initializing required structures...")

        # Connect to PostgreSQL
        self.pgman.connect_pg("pg_key.csv")

        ########################
        # Pipeline preparation #
        ########################

        lg.info("[Init] Loading 100K vocabulary")

        # Load KD-Tree Fine
        fine_words = read_flickr_vocabulary("Datasets/vocabularies/clust_flickr60_k100000.fvecs", 100000)

        lg.info("[Init] Loading fine vocabulary indexes")

        self.bof.load_fine_index("Datasets/vocabularies/fine_index.flann", fine_words)

        lg.info("[Init] Loading coarse vocabulary indexes")

        # Load KD-Tree coarse
        coarse_words = np.load("Datasets/vocabularies/coarse_words.npz")["arr_0"]

        self.bof.load_coarse_index("Datasets/vocabularies/coarse_index.flann", coarse_words)

        lg.info("[Init] Creating the fine -> coarse lookup table")

        # Lookup table
        # TODO: Save load lookup table
        self.lookup_table, lookup_dist = self.bof.search_coarse(fine_words, 1, 32)

        lg.info("[Init] Initializing the priority queue")

        # Initialize priority queue
        self.pq = PriorityQueue()

        lg.info("[Init] ...done.")

    def quantize_query_descriptors(self, descs):
        """
        Bulk transform query descriptors into fine/coarse words
        :param descs: query descriptors to quantize
        :return: triplet containing the fine words, coarse words and coarse costs
        """

        # Quantize descriptors with fine vocabulary
        res, _ = self.bof.search_fine(np.array(descs), 1, self.L_checks)

        # For each fine word, associate a coarse index
        t_coarse_start = time.time()

        coarse_words = defaultdict(list)

        fine_idx_string = ""
        for i, fine_idx in enumerate(res):
            fine_idx = res[i]

            lu_idx = self.lookup_table[fine_idx]

            if lu_idx in coarse_words:
                coarse_words[lu_idx].add_element(i)
            else:
                coarse_words[lu_idx] = CoarseMatch(i)

            fine_idx_string += str(fine_idx) + ","

        fine_idx_string = "(" + fine_idx_string[:-1] + ")"
        t_coarse_end = time.time() - t_coarse_start

        lg.info("[Coarse Lookup] " + str(t_coarse_end) + " seconds")

        # Sort result by search costs
        t_sort_cost_start = time.time()

        # TODO: What to do when the word has only one 3D point? Can't test lowes thrs
        self.pgman.execute_query(pgquery_fine_matches % fine_idx_string)

        t_sort_cost_end = time.time() - t_sort_cost_start
        lg.info("[Sort Costs] " + str(t_sort_cost_end) + " seconds")

        return self.pgman.fetch_all(), coarse_words#, coarse_costs

    def query_photograph(self, photo_path):
        """
        Main routine to query photographs.
        :param photo_path: photograph SIFT file path
        :return: None if pose failed, 3x1 array if success
        """

        global pgquery_fine_matches
        global pgquery_3d_nn
        global pgquery_3dxyz

        # Computed position matrix
        Cmat = None

        lg.info("[Query] Starting query for image: %s" % photo_path)

        #####################
        # Read query photos #
        #####################

        kpts, descs = read_sift_file(photo_path)

        ############################
        # Pipeline execution (VPS) #
        ############################

        # To measure elapse time
        t_start = time.time()

        # Number of matches found
        Nselc = 0

        # Matches found
        matches = dict()

        lg.info("[Query] Quantizing query descritpros into fine/coarse words")

        # Compute initial set of matches to analyse
        fine_matches, coarse_matches = self.quantize_query_descriptors(descs)

        # Initialize the priority heap with the initial set
        self.pq.set_queue(fine_matches)

        # Count time from here
        lg.info("[VPSAS] Starting VPS/AS for %i matches" % self.Nt)

        # For each point in priority queue
        while self.pq.pqueue:

            # Get next in priority
            next_point = self.pq.get_head()

            # Get the properties of the first element in priority
            s_cost = next_point[0]   # Cost to process
            #tieb = next_point[1]      tiebreaker for priority
            wid = next_point[2]      # Related word_id (fine if mode=VPS, coarse if mode=AS)
            pt2d_idx = next_point[3] # 2D point
            pt3d_idx = next_point[4] # 3D point
            mode = next_point[5]     # mode=0 -> VPS, mode=1 -> AS

            # Alternate between VPS and AS by their search costs
            if mode == 0:
                #########################################
                # Process Vocabulary Prioritized Search #
                #########################################

                # Compute the 2-nn within a visual word in linear time
                self.pgman.execute_query("SELECT (get_2nn_linear(%s, %i)).*;" % (self.pgman.float128tostring(descs[pt2d_idx]), wid))#, self.r_thrs_2D3D))

                nn_3d = self.pgman.fetch_all()

                # Assume always accept
                accept_match = True

                # Get the first NN
                nn_1st_3d = [nn_3d[0][0], nn_3d[0][1]]

                # If there are at least two points, compute Lowes ratio
                if len(nn_3d) > 1:
                    # Get the second NN
                    nn_2nd_3d = [nn_3d[1][0], nn_3d[1][1]]

                    # Accept 1-NN if ||d-d1||2 < r.||d-d2||2
                    accept_match = lowes_ratio_test(nn_1st_3d[1], nn_2nd_3d[1], self.r_thrs_2D3D)
                # else, TODO: accept the only point due to distinctiveness?

                # If match is accepted
                if accept_match:
                    # Store 3D->2D idx, distance, mode
                    matches[nn_1st_3d[0]] = (pt2d_idx, nn_1st_3d[1], 0)

                    # Increment number of selected matches
                    Nselc += 1

                    lg.debug("   Time %.4f | Mode: VPS | N.Matches: %i" % (round(time.time() - t_start, 4), Nselc))

                    # Early break if enough matches
                    if Nselc >= self.Nt:
                        break

                    #########################
                    # Prepare Active Search #
                    #########################

                    # Get 3D coordinates
                    self.pgman.execute_query("SELECT get_3d_position(%i);" % nn_1st_3d[0])

                    fetched_pos3d = self.pgman.fetch_all()

                    # Get NN in 3D space
                    self.pgman.execute_query(pgquery_3d_nn % (fetched_pos3d[0][0], self.N3D, nn_1st_3d[0]))

                    # List(pt3d_id, wid_coarse)
                    spatial_nn_3d = self.pgman.fetch_all()

                    # Get 3D-2D matches from coarse lookup
                    # TODO: Avoid processing self 3D
                    for spatial_cantidate in spatial_nn_3d[1:]:
                        c_id = spatial_cantidate[0]
                        c_wid_coarse = spatial_cantidate[1]

                        # If the coarse word exists in query features
                        if c_wid_coarse in coarse_matches:
                            # Get the related cost
                            c_cost = coarse_matches[c_wid_coarse].cost#[1]

                            # Add to priority queue (cost, word_id, pt2did, pt3did, mode=AS)
                            self.pq.add_element(c_cost, c_wid_coarse, pt2d_idx, c_id, 1)

            else:
                #########################
                # Process Active Search #
                #########################

                # Get the descriptor of the 3D point
                self.pgman.execute_query("SELECT get_3d_descriptor(%i, %i);" % (pt3d_idx, wid))

                coarse_descriptors = self.pgman.fetch_all()

                # Convert the 3D descriptors to float128
                # TODO: Need to account on having multiple 3D descriptors on a single coarse word
                db_3d_descriptor = self.pgman.binarystring2float(coarse_descriptors[0][0])

                # Get descriptors from the coarse_matches id's (list)
                # Numpy multi index is actual faster than manual list build??
                q_2d_descriptors = descs[coarse_matches[wid].q_ids]#[0]

                # TODO: If the number of 2d descriptors is 1, how do we Lowes ratio threshold?
                if len(q_2d_descriptors) == 1:
                    lg.debug("[AS] Only one 2D descriptor...ignoring")
                    continue

                # knn 3D-2D
                nn_2d, nn_dist = self.bof.search_runtime_nn(db_3d_descriptor, q_2d_descriptors, 2)

                # Check if is a multiple assignment p.7 [1]
                is_repeated = False
                if pt3d_idx in matches:
                    # if distance if bigger, ignore
                    if nn_dist[0][0] >= matches[pt3d_idx][1]:
                        lg.debug("[AS] Higher distance %f <-> %f" % (nn_dist[0][0], matches[pt3d_idx][1]))
                        continue
                    # if distance is lower, but came from mode VPS, ignore
                    elif matches[pt3d_idx][2] == 0:
                        lg.debug("[AS] Match came from VPS...ignoring")
                        continue
                    # If it is repeated, needs to be replaced but does not increase n.matches
                    else:
                        is_repeated = True

                # Lowes tests ratio
                c_d1 = [coarse_matches[wid].q_ids[nn_2d[0][0]], nn_dist[0][0]]
                c_d2 = [coarse_matches[wid].q_ids[nn_2d[0][1]], nn_dist[0][1]]

                # Accept 1-NN if ||d-d1||2 < r.||d-d2||2
                accept_pt2d = lowes_ratio_test(c_d1[1], c_d2[1], self.r_thrs_3D2D)

                # If accepted
                if accept_pt2d:
                    if not is_repeated:
                        # Increment for each successful match
                        Nselc += 1
                    else:
                        lg.debug("[AS] Repeated match...replacing for older")

                    # Add match (3D->2D idx, distance, mode=AS)
                    matches[pt3d_idx] = (c_d1[0], c_d1[1], 1)

                    lg.debug("   Time: %.4f | Mode:  AS | N.Matches: %i" % (round(time.time() - t_start, 4), Nselc))

                    # If number of matches equals our Nt
                    if Nselc >= self.Nt:
                        break

        lg.info("[Viz] Clustering matches by visibility")
        # Get all matched 3D point ids
        values_3d = "(" + str(matches.keys())[1:-1] + ")"

        # Cluster matches by visibility graph
        # Filters set lower than min_inliers required to pose
        self.pgman.execute_query(pgquery_filter_matches % (values_3d, self.min_inliers))

        # Get the computed sets
        computed_sets = self.pgman.fetch_all()

        lg.info("[Pose] Starting pose estimating hypothesis sets")

        # For each hypothesised set, try pose estimation
        for hypothesis in computed_sets:

            pt3d_ids = "(" + str(hypothesis[0])[1:-1] + ")"

            # Query the 3D xyz positions
            self.pgman.execute_query(pgquery_3dxyz % pt3d_ids)

            # Get the 3D xyz coordinates
            matches_3d = self.pgman.fetch_all()

            # Get the 2D xy coodinates
            matches_2d = []
            for m in hypothesis[0]:
                matches_2d.append(kpts[matches[m][0]])

            matches_2d = np.array(matches_2d)
            matches_3d = np.array(matches_3d)

            Cmat, n_inliers = self.p6pdlt.sprt_ransac_p6pdlt(matches_3d, matches_2d, self.R, self.min_inliers)

            #lg.info("[OTPo] Took " + str(t_compute_pose) + " on this round")

            if Cmat is not None and n_inliers >= self.min_inliers:
                lg.debug("[Pose] Success with " + str(n_inliers) + " inliers")
                break
            else:
                lg.debug("[Pose] Rejected with " + str(n_inliers) + " inliers")

        elapsed_time = time.time() - t_start

        lg.info("[OTPG] %s" % round(self.pgman.get_overall_time(), 4))
        lg.info("[OTVV] %s" % round(self.bof.get_overall_time(), 4))
        lg.info("[OTPQ] %s" % round(self.pq.get_overall_time(), 4))

        lg.info("   Total time was: " + str(elapsed_time) + " seconds")
        lg.info(" ------------------------ ")

        return Cmat



