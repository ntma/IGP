# Python imports
import logging as lg
import time
from collections import defaultdict

# IGP imports
from utils.converters import *
from utils.bag_of_features import BagOfFeatures
from utils.pgres_wrapper import PGWrapper
from utils.priority_queue import PriorityQueue
from utils.geometry import lowes_ratio_test, euclidean_distance
from utils.io_datasets import read_generic_vocabulary_100K, read_sift_file

# 3rdparty imports
from thirdparty.sprt_ransac_6ptdlt import SPRTRANSACDLT

################################################################
### Ugly bulk queries that need to be optimized and isolated ###
################################################################
pgquery_fine_matches = "WITH input_wids(wid, in_order) AS (" \
                            "SELECT wid, row_number() over() in_order " \
                            "FROM (VALUES %s) as t(wid)" \
                       ") SELECT search_cost, 0::int, inverted_costs.wid, in_order - 1, 0::int, 0::int " \
                       "FROM inverted_costs, input_wids " \
                       "WHERE input_wids.wid = inverted_costs.wid " \
                            "AND search_cost > 0 " \
                       "ORDER BY search_cost ASC;"

pgquery_3d_nn = "WITH nn3d(id, w) AS (" \
                    "SELECT id, wid_coarse_lvl3 " \
                    "FROM inverted_list, " \
                        "get_3d_nn_clustersets('%s'::geometry, %i) " \
                    "WHERE id = inverted_list.pt3d_id AND cluster_id IN (SELECT get_vizclusters_from_3dpt(%i))" \
                ") SELECT id, array_agg(w) FROM nn3d " \
                "WHERE id != %i " \
                "GROUP BY id;"


pgquery_3dxyz = "WITH inpoints AS (" \
                    "SELECT id, row_number() over() rn " \
                    "FROM (VALUES %s) as t(id)" \
                ")SELECT x,y,z " \
                "FROM inpoints, points3d " \
                "WHERE points3d.id = inpoints.id " \
                "ORDER BY rn ASC;"

pgquery_filter_matches = "SELECT pt_list " \
                         "FROM " \
                            "(SELECT array_agg(pt3d_id) as pt_list, cluster_id, count(cluster_id) as c " \
                            "FROM visibility_graph " \
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

###################################################
# Class to hold the two closest nearest neighbors #
# Adapted from [3] ACG-Localizer c++ source code  #
###################################################
class NN:
    __slots__ = ['nn_1_id', 'nn_2_id', 'nn_1_dist', 'nn_2_dist']

    def __init__(self):
        self.nn_1_id = -1  # 1st nn
        self.nn_2_id = -1  # 2nd nn

        self.nn_1_dist = 0  # 1st nn distance
        self.nn_2_dist = 0  # 2nd nn distance

    # Add a nearest neighbor
    def add_nn(self, idx, dist):
        # If we have yet to set the 1st nn
        if self.nn_1_id == -1:
            self.nn_1_id = idx
            self.nn_1_dist = dist
        else:
            # If the 1st nn is already set
            if self.nn_1_dist > dist:
                # We want two nn with different ids
                if self.nn_1_id != idx:
                    self.nn_2_id = self.nn_1_id
                    self.nn_2_dist = self.nn_1_dist

                self.nn_1_id = idx
                self.nn_1_dist = dist
            # If the 2nd nn isn't set or the new nn is closer
            elif (dist < self.nn_2_dist or self.nn_2_id == -1) and (idx != self.nn_1_id):
                self.nn_2_id = idx
                self.nn_2_dist = dist

    # Returns true if the two nn are set
    def validate(self):
        if self.nn_1_id == -1 or self.nn_2_id == -1:
            return False
        else:
            return True


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
        self.min_solution = 6     # Minimum number of inlier to sucess pose  TODO: In [2] this param. is set to 12
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
        fine_words = read_generic_vocabulary_100K("Datasets/vocabularies/markt_paris_gpu_sift_100k.cluster", True)

        lg.info("[Init] Loading fine vocabulary indexes")

        # Load the KD-Tree index
        self.bof.load_fine_index("Datasets/vocabularies/fine_index.flann", fine_words)

        lg.info("[Init] Loading coarse vocabulary indexes")

        # Load the parents at level 3
        self.lookup_table = np.load("Datasets/vocabularies/coarse_level3.npz")["arr_0"]

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
        quantized_descriptors, _ = self.bof.search_fine(np.array(descs), 1, self.L_checks)

        # For each fine word, associate a coarse index
        t_coarse_start = time.time()

        # To store the activated words on level 3
        coarse_words = defaultdict(list)

        fine_idx_string = ""
        for i, fine_idx in enumerate(quantized_descriptors):
            fine_idx = quantized_descriptors[i]

            lu_idx = self.lookup_table[fine_idx]

            if lu_idx in coarse_words:
                coarse_words[lu_idx].add_element(i)
            else:
                coarse_words[lu_idx] = CoarseMatch(i)

            fine_idx_string += "(" + str(fine_idx) + "),"

        fine_idx_string = fine_idx_string[:-1]

        t_coarse_end = time.time() - t_coarse_start

        lg.info("[Coarse Lookup] " + str(t_coarse_end) + " seconds")

        # Sort result by search costs
        t_sort_cost_start = time.time()

        self.pgman.execute_query(pgquery_fine_matches % fine_idx_string)

        t_sort_cost_end = time.time() - t_sort_cost_start
        lg.info("[Sort Costs] " + str(t_sort_cost_end) + " seconds")

        return self.pgman.fetch_all(), coarse_words

    def query_photograph(self, photo_path, w, h):
        """
        Main routine to query photographs.
        :param photo_path: photograph SIFT file path
        :return: None if pose failed, 3x1 array if success
        """

        global pgquery_fine_matches
        global pgquery_3d_nn
        global pgquery_3dxyz
        global pgquery_filter_matches

        # Computed position matrix
        Cmat = None

        lg.info("[Query] Starting query for image: %s" % photo_path)
        lg.info("        Read width: %i height: %i" % (w, h) )

        #####################
        # Read query photos #
        #####################
        kpts, descs = read_sift_file(photo_path)

        # Adjust query kpts to the center of the image
        for i, k in enumerate(kpts):
            kpts[i][0] -= (w - 1.0) / 2.0
            kpts[i][1] = (h - 1.0) / 2.0 - kpts[i][1]

        ############################
        # Pipeline execution (VPS) #
        ############################

        # To measure elapse time
        t_start = time.time()

        # Number of matches found
        Nselc = 0

        # Matches found
        matches = dict()

        lg.info("[Query] Quantizing query descriptros into fine/coarse words")

        # Compute initial set of matches to analyse
        activate_fine, activated_coarse = self.quantize_query_descriptors(descs)

        # Initialize the priority heap with the initial set
        self.pq.set_queue(activate_fine)

        # Count time from here
        lg.info("[VPSAS] Starting VPS/AS for %i matches" % self.Nt)

        # For each point in priority queue
        while self.pq.pqueue:

            # Get next in priority
            next_point = self.pq.get_head()

            # Get the properties of the first element in priority
            # s_cost = next_point[0]    Cost to process
            # tieb = next_point[1]      tiebreaker for priority
            wid = next_point[2]       # Related word_id (fine if mode=VPS, coarse if mode=AS)
            pt2d_idx = next_point[3]  # 2D point index
            pt3d_idx = next_point[4]  # 3D point index
            mode = next_point[5]      # mode=0 -> VPS, mode=1 -> AS

            # Alternate between VPS and AS by their search costs
            if mode == 0:
                #########################################
                # Process Vocabulary Prioritized Search #
                #########################################

                # Compute the 2-nn within a visual word in linear time
                self.pgman.execute_query("SELECT (get_2nn_linear(%s, %i)).*;" % (nparray2pgarray(descs[pt2d_idx]), wid))

                nn_3d = self.pgman.fetch_all()

                # If there are at least two nearest neighbors, compute Lowes ratio
                if len(nn_3d) < 2:
                    continue

                # Get the first NN
                nn_1st = [nn_3d[0][0], nn_3d[0][1]]
                nn_2nd = [nn_3d[1][0], nn_3d[1][1]]

                # Accept 1-NN if ||d-d1||2 < r.||d-d2||2
                accept_match = lowes_ratio_test(nn_1st[1], nn_2nd[1], self.r_thrs_2D3D)

                # If match is accepted
                if accept_match:
                    # Store (3D->2D idx, distance, mode)
                    matches[nn_1st[0]] = (pt2d_idx, nn_1st[1], 0)

                    # Increment number of selected matches
                    Nselc += 1

                    lg.debug("   Time %.4f | Mode: VPS | N.Matches: %i" % (round(time.time() - t_start, 4), Nselc))

                    # Early break if enough matches
                    if Nselc >= self.Nt:
                        break

                    #########################
                    # Prepare Active Search #
                    #########################

                    # Get 3D coordinates from a point 3D index
                    self.pgman.execute_query("SELECT get_3d_position(%i);" % nn_1st[0])

                    fetched_pos3d = self.pgman.fetch_all()

                    # Get self.N3D nearest neighbors in the 3D space
                    self.pgman.execute_query(pgquery_3d_nn % (fetched_pos3d[0][0], self.N3D, nn_1st[0], nn_1st[0]))

                    # List(pt3d_id, [wid_coarse_list])
                    spatial_nn_3d = self.pgman.fetch_all()

                    # Get 3D-2D matches from coarse lookup
                    for spatial_cantidate in spatial_nn_3d:
                        candidate_id = spatial_cantidate[0]  # Get the id of a nn
                        candidate_coarse_list = spatial_cantidate[1]  # Get the coarse words associated to this id

                        c_cost = 0

                        # If the coarse word was activated by the image query features
                        # we add the associated cost
                        for c_wid_coarse in candidate_coarse_list:
                            # If the coarse word exists in query features
                            if c_wid_coarse in activated_coarse:
                                # Get the related cost
                                c_cost += activated_coarse[c_wid_coarse].cost#[1]

                        # Add to priority queue (cost, word_id, pt2did, pt3did, mode=AS)
                        if c_cost > 0:
                            self.pq.add_element(c_cost, candidate_coarse_list, pt2d_idx, candidate_id, 1)
                        # else query descriptors not present in this word

                # End of Vocabulary Prioritized Search

            else:
                #########################
                # Process Active Search #
                #########################

                # Get the descriptor of the 3D point
                self.pgman.execute_query("SELECT get_3d_descriptor(%i, %s);" % (pt3d_idx, "ARRAY" + str(wid)))

                db_descriptors = self.pgman.fetch_all()

                # Get descriptors from the activated coarse word id's (list)
                query_descriptors = []
                query_descriptors_idxes = []
                for val in wid:
                    if val not in activated_coarse:
                        continue

                    query_descriptors += descs[activated_coarse[val].q_ids].tolist()

                    query_descriptors_idxes += activated_coarse[val].q_ids

                # Linear compute two nearest neighbors (3D-2D)
                nearest_neighbors = NN()

                # For all the 3D descriptors associated to this 3D point
                for uchar128 in db_descriptors:
                    # Convert the bytea descriptor to float
                    float128 = bytestring2nparray(uchar128[0], True)

                    # For all the 2D candidate descriptors
                    for j, q_desc in enumerate(query_descriptors):
                        # Compute the squared euclidean distance
                        distance = euclidean_distance(float128, q_desc)

                        # Add to nearest neighbors
                        nearest_neighbors.add_nn(query_descriptors_idxes[j], distance)

                # If we got less than two nearest neighbors, we skip this point
                if not nearest_neighbors.validate():
                    continue

                # Check if is a multiple assignment p.7 [1]
                is_repeated = False

                # If the current point was previously processed
                if pt3d_idx in matches:
                    # if the new distance is bigger, ignore
                    if nearest_neighbors.nn_1_dist >= matches[pt3d_idx][1]:
                        lg.debug("[AS] Higher distance %f <-> %f" % (nearest_neighbors.nn_1_dist, matches[pt3d_idx][1]))
                        continue
                    # if the new distance is lower, but came from mode VPS, ignore
                    elif matches[pt3d_idx][2] == 0:
                        lg.debug("[AS] Match came from VPS...ignoring")
                        continue
                    # If it is repeated, needs to be replaced but does not increase n.matches
                    else:
                        is_repeated = True

                # Lowes tests ratio
                # Accept 1-NN if ||d-d1||2 < r.||d-d2||2
                accept_match = lowes_ratio_test(nearest_neighbors.nn_1_dist, nearest_neighbors.nn_2_dist, self.r_thrs_3D2D)

                # If accepted
                if accept_match:
                    if not is_repeated:
                        # Increment successful match
                        Nselc += 1
                    else:
                        lg.debug("[AS] Repeated match...replacing for older")

                    # Add match (3D->2D idx, distance, mode=AS)
                    matches[pt3d_idx] = (nearest_neighbors.nn_1_id, nearest_neighbors.nn_1_dist, 1)

                    lg.debug("   Time: %.4f | Mode:  AS | N.Matches: %i" % (round(time.time() - t_start, 4), Nselc))

                    # If number of matches equals our Nt
                    if Nselc >= self.Nt:
                        break

        lg.info("[Viz] Clustering matches by visibility")

        # Get all matched 3D point ids
        values_3d = nparray2string(matches.keys())

        # Cluster matches by visibility graph
        # Filters set lower than min_inliers required to pose
        self.pgman.execute_query(pgquery_filter_matches % (values_3d, self.min_solution))

        # Get the computed sets
        computed_sets = self.pgman.fetch_all()

        lg.info("[Pose] Starting pose estimating hypothesis sets")

        # For each hypothesised set, try pose estimation
        for hypothesis in computed_sets:
            # Query the 3D xyz positions
            self.pgman.execute_query(pgquery_3dxyz % nparray2valuesstring(hypothesis[0]))

            # Get the 3D xyz coordinates
            matches_3d = self.pgman.fetch_all()

            # Get the 2D xy coordinates
            matches_2d = [kpts[matches[m][0]] for m in hypothesis[0]]

            matches_2d = np.array(matches_2d)
            matches_3d = np.array(matches_3d)

            # Number of correspondences for this hypothesis
            nb_corr = len(matches_3d)

            # Compute the pose for this hypothesis
            # Returns the query camera center if success
            Cmat, n_inliers = self.p6pdlt.compute_pose(matches_3d, matches_2d, nb_corr, np.max((self.R, float(self.min_solution) / float(nb_corr))))

            #lg.info("[OTPo] Took " + str(t_compute_pose) + " on this round")

            # If the number of inliers validated by this pose is higher than our threshold
            # we assume that the pose is a success
            if Cmat is not None and n_inliers >= self.min_solution:
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
