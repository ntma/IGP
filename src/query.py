# Python imports
import logging as lg
import time
import numpy as np

# Core modules
from core.bag_of_features import BagOfFeatures
from core.pgres_queries import PGQueries
from core.priority_queue import PriorityQueue
from thirdparty.sprt_ransac_6ptdlt import SPRTRANSACDLT

# Helper modules
from utils.converters import bytestring2nparray
from utils.geometry import lowes_ratio_test, euclidean_distance
from utils.io_datasets import read_generic_vocabulary_100K, read_sift_file


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
class CoarseMatch(object):
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
class NN(object):
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
        self.pgqueries = PGQueries()       # Middle man between thie module and the PostgreSQL wrapper
        self.pq = PriorityQueue()          # Priority Queue module
        self.bof = BagOfFeatures()         # Bag of Features module
        self.p6pdlt = SPRTRANSACDLT()      # SPRT-RANSAC with p6p DLT (Adapted from ACG-Localizer [1] src available in [3])

        # EEILR parameters [1]
        self.Nt = 100             # Number of points to search (VPS+AS)
        self.N3D = 200            # Number of 3D nearest neighbors
        self.R = 0.2              # Minimum threshold for ransac
        self.min_inliers = 6      # Minimum inliers required to consider a pose
        self.min_solution = 12    # Minimum number of inlier to sucess pose
        self.r_thrs_2D3D = 0.7    # Threshold for 2D-3D Lowes ratio test
        self.r_thrs_3D2D = 0.6    # Threshold for 3D-2D Lowes ration test
        self.L_checks = 10        # Number of leafs to check in fine vocabulary

        self.lookup_table = None  # Lookup table for the coarse vocabulary

    def initialize_ilr(self, pgkey_filepath, bof_directory):
        """
        :param pgkey_filepath: Path to the csv containing the connection key to postgres
        :param bof_directory: Bag of features directory
        Initiales the ILR module. Opens a connection to postgres and loads the requires visual
        vocabularies.
        :return:
        """

        lg.info("[Init] Initializing required structures...")

        # Connect to PostgreSQL
        self.pgqueries.connect_pg(pgkey_filepath)

        ########################
        # Pipeline preparation #
        ########################

        lg.info("   Loading 100K vocabulary")

        # Load KD-Tree Fine
        fine_words = read_generic_vocabulary_100K(bof_directory + "markt_paris_gpu_sift_100k.cluster", True)

        lg.info("   Loading fine vocabulary indexes")

        # Load the KD-Tree index
        self.bof.load_fine_index(bof_directory + "fine_index.flann", fine_words)

        lg.info("   Loading coarse vocabulary indexes")

        # Load the parents at level 3
        self.lookup_table = np.load(bof_directory + "coarse_level3.npz")["arr_0"]

        lg.info("   Initializing the priority queue")

        # Initialize priority queue
        self.pq = PriorityQueue()

        lg.info("   ...done.")

    def quantize_query_descriptors(self, query_descriptors):
        """
        Bulk transform query descriptors into fine/coarse words
        :param query_descriptors: query descriptors to quantize
        :return: triplet containing the fine words, coarse words and coarse costs
        """

        # Quantize descriptors with fine vocabulary
        quantized_descriptors, _ = self.bof.search_fine(query_descriptors, 1, self.L_checks)

        # To store the activated words on level 3
        coarse_words = dict()

        # For each quantized descriptor, we need to lookup for their parents at level 3
        fine_idx_string = ""
        for i, fine_idx in enumerate(quantized_descriptors):
            lu_idx = self.lookup_table[fine_idx]

            # if word already inserted, we append the id
            if lu_idx in coarse_words:
                coarse_words[lu_idx].add_element(i)
            else:
                coarse_words[lu_idx] = CoarseMatch(i)

            fine_idx_string += "(" + str(fine_idx) + "),"

        fine_idx_string = fine_idx_string[:-1]

        # Build a prioritized queue based on the quantized words.
        # We only do this for the fine words, since the coarse are
        # only a lookup table to the fine words for the active search.
        fine_words = self.pgqueries.sort_quantized_words(fine_idx_string)

        return fine_words, coarse_words

    def find_correspondences(self, query_descriptors, activated_fine, activated_coarse):
        """
        Given the query descriptors and the activated fine and coarse words,
        computes correpondences to database points
        :param query_descriptors: Query descriptors
        :param activated_fine: Activated fine words
        :param activated_coarse: Activated coarse words
        :return:
        """

        ############################
        # Pipeline execution (VPS) #
        ############################

        # Initialize the priority heap with the initial set
        self.pq.set_queue(activated_fine)

        # Number of matches found
        n_selected = 0

        # Matches found
        matches = dict()

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
                nn_3d = self.pgqueries.search_two_nn(query_descriptors[pt2d_idx].tolist(), wid)

                # If there are at least two nearest neighbors, compute Lowes ratio
                if len(nn_3d) < 2:
                    continue

                # Get the two closest nearest neighbors
                nn_1st = [nn_3d[0][0], nn_3d[0][1]]
                nn_2nd = [nn_3d[1][0], nn_3d[1][1]]

                # Accept 1-NN if ||d-d1||2 < r.||d-d2||2
                accept_match = lowes_ratio_test(nn_1st[1], nn_2nd[1], self.r_thrs_2D3D)

                # If match is accepted
                if accept_match:
                    # Store (3D->2D idx, distance, mode)
                    matches[nn_1st[0]] = (pt2d_idx, nn_1st[1], 0)

                    # Increment number of selected matches
                    n_selected += 1

                    lg.debug("   Mode: VPS | N.Matches: %i" % n_selected)

                    # Early break if enough matches
                    if n_selected >= self.Nt:
                        break

                    #########################
                    # Prepare Active Search #
                    #########################

                    # Get self.N3D nearest neighbors in the 3D space. List(pt3d_id, [wid_coarse_list])
                    spatial_nn_3d = self.pgqueries.search_3d_nn(nn_1st[0], self.N3D)

                    # Get 3D-2D matches from coarse lookup
                    for candidate_id, candidate_coarse_list in spatial_nn_3d:
                        #  candidate_id -> Get the id of a nn
                        #  candidate_coarse_list -> Get the coarse words associated to this id

                        c_cost = 0

                        # If the coarse word was activated by the image query features
                        # we add the associated cost
                        for c_wid_coarse in candidate_coarse_list:
                            # If the coarse word exists in query features
                            if c_wid_coarse in activated_coarse:
                                # Get the related cost
                                c_cost += activated_coarse[c_wid_coarse].cost

                        # Add to priority queue (cost, word_id, pt2did, pt3did, mode=AS)
                        if c_cost > 0:
                            self.pq.add_element(c_cost, candidate_coarse_list, pt2d_idx, candidate_id, 1)
                        # else query descriptors not present in this word

                # End of Vocabulary Prioritized Search

            else:
                #########################
                # Process Active Search #
                #########################

                # Get the descriptors for this 3D point
                db_descriptors = self.pgqueries.get_3d_descriptors_from_id(pt3d_idx, wid)

                # Get descriptors from the activated coarse word id's (list)
                q_descriptors = []
                q_descriptors_idx = []
                for val in wid:
                    if val not in activated_coarse:
                        continue

                    q_descriptors += query_descriptors[activated_coarse[val].q_ids].tolist()

                    q_descriptors_idx += activated_coarse[val].q_ids

                # Linear compute two nearest neighbors (3D-2D)
                nearest_neighbors = NN()

                # For all the 3D descriptors associated to this 3D point
                for uchar128 in db_descriptors:
                    # Convert the bytea descriptor to float
                    float128 = bytestring2nparray(uchar128[0], True)

                    # For all the 2D candidate descriptors
                    for j, q_desc in enumerate(q_descriptors):
                        # Compute the squared euclidean distance
                        distance = euclidean_distance(float128, q_desc)

                        # Add to nearest neighbors
                        nearest_neighbors.add_nn(q_descriptors_idx[j], distance)

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
                        n_selected += 1
                    else:
                        lg.debug("[AS] Repeated match...replacing for older")

                    # Add match (3D->2D idx, distance, mode=AS)
                    matches[pt3d_idx] = (nearest_neighbors.nn_1_id, nearest_neighbors.nn_1_dist, 1)

                    lg.debug("   Mode:  AS | N.Matches: %i" % n_selected)

                    # If number of matches equals our threshold Nt
                    if n_selected >= self.Nt:
                        break

        # Return number of selected matches and the matches found
        return n_selected, matches

    def hypothesise_pose(self, keypoints, matches):
        """
        Hypothesises the pose given the query kpts and correspondences to the database
        :param keypoints: query keypoints
        :param matches: correspondences
        :return: Success: (True, 3x1 position), Fail: (False, None)
        """

        pose_success = False
        Cmatrix = None

        # Cluster matches by visibility graph
        # Filters set lower than min_inliers required to pose
        computed_sets = self.pgqueries.filter_matches_by_visibility(matches.keys(), self.min_solution)

        # For each hypothesised set, try pose estimation
        for hypothesis in computed_sets:

            # Query the 3D xyz positions
            matches_3d = self.pgqueries.get_xyz_from_ids(hypothesis[0])

            # Get the 2D xy coordinates
            matches_2d = [keypoints[matches[m][0]] for m in hypothesis[0]]

            matches_2d = np.array(matches_2d)
            matches_3d = np.array(matches_3d)

            # Number of correspondences for this hypothesis
            n_correspondences = len(matches_3d)

            # Set minimum inlier ratio as in [1]
            min_inlier_ratio = np.max((self.R, float(self.min_solution) / float(n_correspondences)))

            # Compute the pose for this hypothesis
            # Returns the query camera center if success
            lg.debug("   Considering %i correspondences" % n_correspondences)

            Cmatrix, n_inliers = self.p6pdlt.compute_pose(matches_3d, matches_2d, n_correspondences, min_inlier_ratio)

            # If the number of inliers validated by this pose is higher than our threshold
            # we assume that the pose is a success
            if Cmatrix is not None and n_inliers >= self.min_solution:
                lg.debug("   Success with " + str(n_inliers) + " inliers")

                pose_success = True
                break
            else:
                lg.debug("   Rejected with " + str(n_inliers) + " inliers")

        # Return the success flag and the computed position matrix
        return pose_success, Cmatrix

    def query_photograph(self, filename, width, height):
        """
        Main routine to query photographs.
        :param filename: photograph SIFT file path
        :param width: photograph width
        :param height: photograph height
        :return: None if pose failed, 3x1 array if success
        """

        lg.info("[Query] Image: %s" % filename)
        lg.info("   Width: %i Height: %i" % (width, height))

        #####################
        # Read query photos #
        #####################
        keypoints, descriptors = read_sift_file(filename)

        # Adjust query keypoints to the center of the image
        for i, k in enumerate(keypoints):
            keypoints[i][0] -= (width - 1.0) / 2.0
            keypoints[i][1] = (height - 1.0) / 2.0 - keypoints[i][1]

        # Measure elapse time
        t_total_start = time.time()

        # Quantize words
        lg.info("   Quantizing query descriptors into fine/coarse words")

        # Compute initial set of matches to process
        activated_fine, activated_coarse = self.quantize_query_descriptors(descriptors)

        lg.info("   Starting VPS/AS for %i matches" % self.Nt)

        # Search for correspondences
        n_matches_found, matches = self.find_correspondences(descriptors, activated_fine, activated_coarse)

        lg.info("   Found %i matches...now hypothesising pose" % n_matches_found)

        # Hypothesise a pose using the correspondences found
        success, Cmatrix = self.hypothesise_pose(keypoints, matches)

        elapsed_time = time.time() - t_total_start

        lg.info("   Total time was: " + str(elapsed_time) + " seconds")
        lg.info(" ------------------------ ")

        # Returns the success flag and the computed position matrix
        return success, Cmatrix
