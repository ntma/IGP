import logging as lg
import argparse
import numpy as np
import struct

from collections import defaultdict
from pyflann import *

from src.utils.geometry import bundler_extract_position, bundler_extract_viewdir
from src.utils.io_datasets import read_generic_vocabulary_100K

from src.core.bag_of_features import BagOfFeatures
from src.utils.io_pointcloud import PCLHolder

pcl_holder = PCLHolder()


def create_visibility_graph(k, out_path):

    global pcl_holder

    lg.info("[Prepare] Loading point cloud cameras data")

    n_cameras = len(pcl_holder.cameras)

    # Compute the viewing direction of each camera
    view_directions = []
    camera_positions = []
    for c in pcl_holder.cameras:
        R = c[3]
        t = c[4]

        R = R.reshape((3, 3))
        t = t.transpose()

        v_dir = bundler_extract_viewdir(R)
        C = bundler_extract_position(R, t).transpose()

        view_directions.append(v_dir)
        camera_positions.append(C)

    camera_positions = np.array(camera_positions)

    lg.info("   Clustering cameras")

    # Cluster 3D positions by NN
    flann = FLANN()

    # Compute the centroids
    # clusters, distances = flann.nn(camera_positions, camera_positions, k + 1, algorithm=0, log_level="info")
    clusters = [[] for i in xrange(n_cameras)]
    for iii in xrange(n_cameras):

        cluster, distances = flann.nn(camera_positions, np.array([camera_positions[iii]]), k + 1, algorithm=0,
                                      log_level="info")
        end_cluster = []
        n_selc = 0
        for w in cluster[0]:
            if n_selc == 10:
                break

            if w != iii:
                end_cluster.append(w)

        clusters[iii] = end_cluster

    lg.info("   Computing camera sets by delta angle")

    # Connect cameras with at least 60 viewing direction
    idx = 0

    images_covered_by_image = [[i] for i in range(n_cameras)]

    for iii, cluster in enumerate(clusters):

        c_idx = iii

        c_view_dir = view_directions[c_idx]

        n_accepted = 0
        for cam_idx in cluster:
            sim_view_dir = view_directions[cam_idx]

            if np.dot(c_view_dir, sim_view_dir) >= 0.5:  # delta_view_dir(c_view_dir, sim_view_dir) < a_thrs:

                # Compute sim
                images_covered_by_image[c_idx].append(cam_idx)
                n_accepted += 1

                if n_accepted == k:
                    break

        idx += 1

    lg.info("   Choosing the best sets")

    # Now select the largest sets

    size_set_cover = 0
    new_image_ids = [-1 for i in range(n_cameras)]
    image_covered_by = [[] for i in range(n_cameras)]

    nb_new_images_covered = [(i, len(images_covered_by_image[i])) for i in range(n_cameras)]

    while len(nb_new_images_covered) > 0:

        nb_new_images_covered.sort(key=lambda tup: tup[1], reverse=False)

        if nb_new_images_covered[-1][1] == 0:
            break

        cam_id_ = nb_new_images_covered[-1][0]

        new_image_ids[cam_id_] = size_set_cover

        # mark related images as covered
        for idx, it in enumerate(images_covered_by_image[cam_id_]):
            image_covered_by[it].append(size_set_cover)

        size_set_cover += 1

        # pop first element
        del nb_new_images_covered[-1]

        # recompute the nb of new images each image can cover
        for it_idx, it_val in enumerate(nb_new_images_covered):  # ; it != nb_new_images_covered.end(); ++it )

            new_count = 0
            for idx2, it2 in enumerate(images_covered_by_image[it_val[0]]):

                if len(image_covered_by[it2]) == 0:
                    new_count += 1

            nb_new_images_covered[it_idx] = (it_val[0], new_count)

    lg.info("   Set cover contains %i cameras out of %i" % (size_set_cover, n_cameras))

    cluster_id_covers_set = defaultdict(list)

    for cam_id, cluster in enumerate(image_covered_by):
        for cluster_id in cluster:
            if cam_id not in cluster_id_covers_set[cluster_id]:
                cluster_id_covers_set[cluster_id].append(cam_id)

    with open(out_path, "w") as f:

        for i in range(len(pcl_holder.pts3D)):
            rep = pcl_holder.reprojections[i]

            camera_set = set()

            for j in xrange(0, len(rep), 2):
                cam_id = rep[j]

                for cluster in image_covered_by[cam_id]:
                    if cluster not in camera_set:
                        camera_set.add(cluster)

            for cluster in camera_set:
                f.write("%i\t%i\n" % (i, cluster))

        f.close()

    lg.info("...done.")


def create_vocabularies(filename):

    base_path = filename.rsplit('/', 1)[0] + '/'

    np.random.seed(1)

    bof = BagOfFeatures()

    lg.info("   Loading fine vocabulary")

    # Read vocabulary
    words = read_generic_vocabulary_100K(filename, True)

    lg.info("   Creating kd-tree index")
    # Create fine vocabulary kd-tree
    params_fine = bof.create_fine_kdtree(words)

    lg.info("   ...saving.")

    bof.save_fine_index(base_path + "fine_index.flann")

    lg.info("   Creating kmean cluster index")

    # Create fine vocabulary kd-tree
    params_coarse = bof.create_clusters(words, 10)

    lg.info("   Getting parents for lvl 2 and 3")

    max_levels2, cluster_ids2 = bof.get_parents_at_level_L(2)

    lg.info("Max parents at level 2: %i" % max_levels2)

    max_levels3, cluster_ids3 = bof.get_parents_at_level_L(3)

    lg.info("Max parents at level 3: %i" % max_levels3)

    lg.info("   ...saving.")

    np.savez(base_path + "coarse_level2.npz", cluster_ids2)
    np.savez(base_path + "coarse_level3.npz", cluster_ids3)

    lg.info("...done.")

    return


def mpvw_descriptors(vocab_path, out_path):

    global pcl_holder

    base_vocabulary_path = vocab_path.rsplit('/', 1)[0] + '/'

    lg.info("   Loading vocabularies")

    bof = BagOfFeatures()

    words = read_generic_vocabulary_100K(vocab_path, True)

    # Load fine vocabulary
    bof.load_fine_index(base_vocabulary_path + "fine_index.flann", words)

    # Load lookups for level 2 and level 3
    coarse_level2 = np.load(base_vocabulary_path + "coarse_level2.npz")["arr_0"]
    coarse_level3 = np.load(base_vocabulary_path + "coarse_level3.npz")["arr_0"]

    lg.info("   Computing/Inserting mean per visual words for %i points" % len(pcl_holder.pts3D))

    with open(out_path, "wb") as f:

        curr_idx = 0
        # For each 3D point
        for i in range(0, len(pcl_holder.pts3D)):
            qdescriptors = np.asarray(pcl_holder.descriptors[curr_idx: curr_idx + int(len(pcl_holder.reprojections[i]) / 2)], dtype=float)
            qdescriptors /= 512.0

            curr_idx += int(len(pcl_holder.reprojections[i]) / 2)

            # Query fine index, 1-NN, L=50 leaf
            qq, dists = bof.search_fine(qdescriptors, 1, 10)  # 50)

            # Now int mean descriptor / vw
            seen = set()

            mpvw_data = []

            for v in qq:
                # if not processed yet
                if v not in seen:
                    # Find duplicate indexes
                    dups = np.where(qq == v)[0]

                    # Sum descriptors belonging to repeate indexes
                    m_descriptor = np.zeros(128, dtype=float)
                    for d in dups:
                        m_descriptor = np.add(m_descriptor, qdescriptors[d])

                    # Now mean the sum
                    m_descriptor /= float(len(dups))

                    """int_descriptor = np.zeros(128, dtype=np.int32)
                    for k in range(0, 128):
                        bottom = m_descriptor[k] - np.floor(m_descriptor[k])
                        top = np.ceil(m_descriptor[k]) - m_descriptor[k]
                        if bottom < top:
                            int_descriptor[k] = int(np.floor(m_descriptor[k]))
                        else:
                            int_descriptor[k] = int(np.ceil(m_descriptor[k]))
                    """
                    # Convert 128float to binary string
                    bindesc = np.asarray(np.floor(m_descriptor * 512.0 + 0.5), dtype=uint8)

                    # Store to later insert in database
                    mpvw_data.append([i, int(v), int(coarse_level2[v]), int(coarse_level3[v]), bindesc])

                    # Add too seen
                    seen.add(v)

            f.write(struct.pack("i", len(mpvw_data)))

            for mpvw in mpvw_data:
                f.write(struct.pack("i", mpvw[0]))
                f.write(struct.pack("i", mpvw[1]))
                f.write(struct.pack("i", mpvw[2]))
                f.write(struct.pack("i", mpvw[3]))
                f.write(struct.pack("B" * 128, *mpvw[4]))

            if i % 100000 == 0:
                lg.debug("   on going: %i" % i)

        f.close()

    lg.info("...done.")


def generate_csv(out_path):

    cameras_path = out_path + "cameras.csv"
    points3d_path = out_path + "points3d.csv"
    viewlist_path = out_path + "viewlist.csv"

    f = open(cameras_path, "w")

    for i, cam in enumerate(pcl_holder.cameras):
        focal = cam[0]
        k1 = cam[1]
        R = cam[3]
        t = cam[4]

        f.write("%i\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (i, focal, k1,
                                                                                  R[0], R[1], R[2],
                                                                                  R[3], R[4], R[5],
                                                                                  R[6], R[7], R[8],
                                                                                  t[0], t[1], t[2]))

    f.close()

    f = open(points3d_path, "w")

    for i, p3d in enumerate(pcl_holder.pts3D):
        f.write("%i\t%f\t%f\t%f\n" % (i, p3d[0], p3d[1], p3d[2]))
    f.close()

    f = open(viewlist_path, "w")

    for i, vl in enumerate(pcl_holder.reprojections):
        for j in xrange(0, len(vl), 2):
            f.write("%i\t%i\t%i\n" % (i, vl[j], vl[j + 1]))

    f.close()


def pre_process_dataset(bin_path, vocab_path, out_path):

    global pcl_holder

    pcl_holder.load_binary(bin_path)

    # Create visibility graph
    create_visibility_graph(10, out_path + "viz_graph.csv")

    # Create Connected components
    # TODO: for the remaining datasets, this is required

    # Create Vocabularies
    create_vocabularies(vocab_path)

    # Computing assignemnts
    mpvw_descriptors(vocab_path, out_path + "assignments.bin")

    # Create CSV's
    generate_csv(out_path)


if __name__ == "__main__":
    # Set the logging module
    lg.basicConfig(format='%(message)s', level=lg.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Set the argument options
    parser.add_argument('-p', action='store', dest='BINARY_PATH', help='Binary file outputted by parse_dataset', default='')
    parser.add_argument('-o', action='store', dest='OUTPUT_PATH', help='Output directory for CSVs', default='')
    parser.add_argument('-w', action='store', dest='VOCABULARY_PATH', help='Visual vocabulary file', default='vocabularies/markt_paris_gpu_sift_100k.cluster')

    arg_v = parser.parse_args()

    # Path to the dataset folder
    binary_path = arg_v.DATASET_PATH
    vocabulary_path = arg_v.VOCABULARY_PATH
    output_path = arg_v.OUTPUT_PATH

    # Add / if not present
    if binary_path[-1] != '/':
        binary_path += '/'

    if output_path == '':
        output_path = binary_path.rsplit('/', 1)[0] + '/'

    pre_process_dataset(binary_path, vocabulary_path, output_path)
