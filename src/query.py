# Python imports
import logging as lg
import time
import numpy as np
from collections import defaultdict
from collections import deque

# Core modules
from core.bag_of_features import BagOfFeatures
from core.pgres_queries import PGQueries
from core.priority_queue import PriorityQueue
from thirdparty.sprt_ransac_6ptdlt import SPRTRANSACDLT

# Helper modules
from utils.converters import bytestring2nparray
from utils.geometry import lowes_ratio_test, c_euclidean_distance_128
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
        self.pgqueries = PGQueries()  # Middle man between thie module and the PostgreSQL wrapper
        self.pq = PriorityQueue()  # Priority Queue module
        self.bof = BagOfFeatures()  # Bag of Features module
        self.p6pdlt = SPRTRANSACDLT()  # SPRT-RANSAC with p6p DLT (Adapted from ACG-Localizer [1] src available in [3])

        # EEILR parameters [1]
        self.Nt = 100  # Number of points to search (VPS+AS)
        self.N3D = 200  # Number of 3D nearest neighbors
        self.R = 0.2  # Minimum threshold for ransac
        self.min_inliers = 6  # Minimum inliers required to consider a pose
        self.min_solution = 12  # Minimum number of inlier to sucess pose
        self.r_thrs_2D3D = 0.49  # Threshold for 2D-3D Lowes ratio test (0.7^2 due to squared euclidean)
        self.r_thrs_3D2D = 0.36  # Threshold for 3D-2D Lowes ration test (0.6^2 due to squared euclidean)
        self.L_checks = 10  # Number of leafs to check in fine vocabulary

        self.low_dim_thrs = 5000  # Low dimensionality threshold
        self.lookup_table_coarse = None  # Lookup table for the coarse vocabulary
        self.lookup_table_fine = None  # Lookup table for the coarse vocabulary

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

        lg.info("   Loading coarse vocabulary indexes for level 2 and 3")

        # Load parents at level 2
        self.lookup_table_coarse = np.load(bof_directory + "coarse_level2.npz")["arr_0"]

        # Load parents at level 3
        self.lookup_table_fine = np.load(bof_directory + "coarse_level3.npz")["arr_0"]

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

        # We choose the parents at level 3 if # query features > 5000
        if len(query_descriptors) > self.low_dim_thrs:
            lookup_table = self.lookup_table_fine
        else:
            lookup_table = self.lookup_table_coarse

        # Quantize descriptors with fine vocabulary
        quantized_descriptors, _ = self.bof.search_fine(query_descriptors, 1, self.L_checks)

        # To store the activated words on level L
        coarse_words = dict()

        # For each quantized descriptor, we need to lookup for their parents at level L
        fine_idx_string = ""
        for i, fine_idx in enumerate(quantized_descriptors):
            lu_idx = lookup_table[fine_idx]

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

        # First a bit of setup, we set functions for high|low dimensionality
        # to avoid unnecessary loop if conditions
        if len(query_descriptors) > self.low_dim_thrs:
            search_3d_nn = self.pgqueries.search_3d_nn_lvl3
            get_3d_descriptors_from_id = self.pgqueries.get_3d_descriptors_from_id_lvl3
        else:
            search_3d_nn = self.pgqueries.search_3d_nn_lvl2
            get_3d_descriptors_from_id = self.pgqueries.get_3d_descriptors_from_id_lvl2

        # Number of matches found
        n_selected = 0

        # Matches found
        matches = dict()
        feature_in_correspondence = [-1 for x in xrange(len(query_descriptors))]

        # Initialize the priority heap with the initial set
        self.pq.set_queue(activated_fine)

        # For each point in priority queue
        while self.pq.pqueue:
            # Get next in priority
            next_point = self.pq.get_head()

            # Get the properties of the first element in priority
            # s_cost = next_point[0]    Cost to process
            # tieb = next_point[1]      tiebreaker for priority
            # wid = next_point[2]       Related word_id (fine if mode=VPS, coarse if mode=AS)
            # pt2d_idx = next_point[3]  2D point index
            # pt3d_idx = next_point[4]  3D point index
            # mode = next_point[5]      mode=0 -> VPS, mode=1 -> AS

            _, _, wid, pt2d_idx, pt3d_idx, mode = next_point

            #########################################
            # Process Vocabulary Prioritized Search #
            # Alternate between 2D-3D/3D-2D by      #
            # their search cost                     #
            #########################################
            if mode == 0:
                #################
                # Process 2D-3D #
                #################

                # Compute the 2-nn within a visual word in linear time
                nn_3d = self.pgqueries.search_two_nn(query_descriptors[pt2d_idx].tolist(), wid)

                # If there are at least two nearest neighbors, compute Lowes ratio
                if len(nn_3d) < 2:
                    continue

                # Get the two closest nearest neighbors
                nn_1st = (nn_3d[0][0], nn_3d[0][1])
                nn_2nd = (nn_3d[1][0], nn_3d[1][1])

                # Accept 1-NN if ||d-d1||2 < r.||d-d2||2
                accept_match = lowes_ratio_test(nn_1st[1], nn_2nd[1], self.r_thrs_2D3D)

                # If match is accepted
                if accept_match:
                    if nn_1st[0] not in matches:
                        # Increment number of selected matches
                        n_selected += 1
                    # If new distance if higher, we discard
                    elif matches[nn_1st[0]][1] <= nn_1st[1]:
                        continue
                    # If it is repeated and distance is lower
                    else:
                        feature_in_correspondence[matches[nn_1st[0]][0]] = -1

                    # Store (3D->2D idx, distance, mode)
                    matches[nn_1st[0]] = (pt2d_idx, nn_1st[1], 0)
                    feature_in_correspondence[pt2d_idx] = nn_1st[0]

                    lg.debug("   Mode: VPS | N.Matches: %i" % n_selected)

                    # Early break if enough matches
                    if n_selected >= self.Nt:
                        break

                    #########################
                    # Process Active Search #
                    #########################

                    # Get self.N3D nearest neighbors in the 3D space. List(pt3d_id, [wid_coarse_list])
                    spatial_nn_3d = search_3d_nn(nn_1st[0], self.N3D)

                    # Get 3D-2D matches from coarse lookup
                    for candidate_id, candidate_coarse_list in spatial_nn_3d:
                        #  candidate_id -> Get the id of a nn
                        #  candidate_coarse_list -> Get the coarse words associated to this id

                        c_cost = 0
                        activated_coarse_list = []

                        # If the coarse word was activated by the image query features
                        # we add the associated cost
                        for c_wid_coarse in candidate_coarse_list:
                            # If the coarse word exists in query features
                            if c_wid_coarse in activated_coarse:
                                # Get the related cost
                                c_cost += activated_coarse[c_wid_coarse].cost
                                activated_coarse_list.append(c_wid_coarse)

                        # Add to priority queue (cost, word_id, pt2did, pt3did, mode=AS)
                        if c_cost > 0:
                            self.pq.add_element(c_cost, activated_coarse_list, pt2d_idx, candidate_id, 1)

                            # else query descriptors not present in this word
                            # End of Active Search
                            # End of 2D-3D

            else:
                #################
                # Process 3D-2D #
                #################

                # Get the descriptors for this 3D point
                db_descriptors = get_3d_descriptors_from_id(pt3d_idx, wid)

                # Get descriptors from the activated coarse word id's (list)
                q_descriptors_idx = []
                for val in wid:
                    if val not in activated_coarse:
                        continue

                    q_descriptors_idx += activated_coarse[val].q_ids

                # Linear compute two nearest neighbors (3D-2D)
                nearest_neighbors = NN()

                # For all the 3D descriptors associated to this 3D point
                for uchar128 in db_descriptors:
                    # Convert the bytea descriptor to float
                    float128 = bytestring2nparray(uchar128[0], True)

                    # For all the 2D candidate descriptors
                    for j, q_desc_idx in enumerate(q_descriptors_idx):
                        # Compute the squared euclidean distance
                        distance = c_euclidean_distance_128(float128, query_descriptors[q_desc_idx])

                        # Add to nearest neighbors
                        nearest_neighbors.add_nn(q_desc_idx, distance)

                # If we got less than two nearest neighbors, we skip this point
                if not nearest_neighbors.validate():
                    continue

                # Lowes tests ratio
                # Accept 1-NN if ||d-d1||2 < r.||d-d2||2
                accept_match = lowes_ratio_test(nearest_neighbors.nn_1_dist, nearest_neighbors.nn_2_dist,
                                                self.r_thrs_3D2D)

                # If accepted
                if accept_match:

                    # Check if is a multiple assignment p.7 [1]
                    # If the correspondence is not set yet
                    if feature_in_correspondence[nearest_neighbors.nn_1_id] == -1:
                        n_selected += 1
                        feature_in_correspondence[nearest_neighbors.nn_1_id] = pt3d_idx

                        matches[pt3d_idx] = (nearest_neighbors.nn_1_id, nearest_neighbors.nn_1_dist, 1)

                    # If the correspondence if set and the previous match has an higher distance
                    elif matches[feature_in_correspondence[nearest_neighbors.nn_1_id]][1] > nearest_neighbors.nn_1_dist:
                        matches.pop(feature_in_correspondence[nearest_neighbors.nn_1_id])

                        feature_in_correspondence[nearest_neighbors.nn_1_id] = pt3d_idx
                        matches[pt3d_idx] = (nearest_neighbors.nn_1_id, nearest_neighbors.nn_1_dist, 1)

                    lg.debug("   Mode:  AS | N.Matches: %i" % n_selected)

                    # If number of matches equals our threshold Nt
                    if n_selected >= self.Nt:
                        break

                        # End of 3D-2D

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

        # Cluster matches by visibility graph
        # Filters set lower than min_inliers required to pose
        computed_sets = self.pgqueries.filter_matches_by_visibility(matches.keys(), self.min_solution)

        ####################################################
        # TSattler connected components from ACG-Localizer #
        ####################################################

        images_per_point = defaultdict(list)
        image_edges = defaultdict(list)
        cc_per_corr = dict((key, -1) for key in matches.keys())
        current_cc = -1
        max_cc = -1
        max_set_size = 0

        # First build the connectivity graph 3D<->cameras
        for pt_id, cam_id in computed_sets:
            images_per_point[pt_id].append(cam_id)
            image_edges[cam_id].append(pt_id)

        # For each 3D point we search all the 3D points that are seen by the same camera set
        for map_it_3D in matches.keys():

            if cc_per_corr[map_it_3D] < 0:

                # start new cc
                current_cc += 1
                size_current_cc = 0

                # Enqueue the first point
                point_queue = deque([map_it_3D])

                # breadth first search for remaining points in connected component
                while point_queue:
                    curr_point_id = point_queue.popleft()

                    # If the cc for this point is not set, we set it
                    if cc_per_corr[curr_point_id] < 0:

                        cc_per_corr[curr_point_id] = current_cc
                        size_current_cc += 1

                        # and add all points in images visible by this point
                        for it_images_point in images_per_point[curr_point_id]:

                            for p_it in image_edges[it_images_point]:

                                if cc_per_corr[p_it] < 0:
                                    point_queue.append(p_it)

                            # clear the image, we do not the this multi-edge anymore
                            image_edges[it_images_point] = []

                # If the new connected component is larger than the previous, we store it
                if size_current_cc > max_set_size:
                    max_set_size = size_current_cc
                    max_cc = current_cc

        # Now get the 3D point ids that were filtered
        hypothesis = [key for key, val in cc_per_corr.iteritems() if val == max_cc]

        # Query the 3D xyz positions by point id
        matches_3d = self.pgqueries.get_xyz_from_ids(hypothesis)

        # Get the 2D xy coordinates
        matches_2d = [keypoints[matches[m][0]] for m in hypothesis]

        matches_2d = np.array(matches_2d)
        matches_3d = np.array(matches_3d)

        # Number of correspondences for this hypothesis
        n_correspondences = len(matches_3d)

        # Set minimum inlier ratio as in [1]
        min_inlier_ratio = max(self.R, float(self.min_solution) / float(n_correspondences))

        # Compute the pose for this hypothesis
        # Returns the query camera center if success
        lg.info("   Considering %i correspondences" % n_correspondences)

        Cmatrix, n_inliers = self.p6pdlt.compute_pose(matches_3d, matches_2d, n_correspondences, min_inlier_ratio)

        # If the number of inliers validated by this pose is higher than our threshold
        # we assume that the pose is a success
        if Cmatrix is not None and n_inliers >= self.min_solution:
            lg.info("   Success with " + str(n_inliers) + " inliers")

            pose_success = True
        else:
            lg.info("   Rejected with " + str(n_inliers) + " inliers")

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

        lg.info("   Read %i features" % len(descriptors))

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
