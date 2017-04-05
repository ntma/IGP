import argparse
import logging as lg

from src.utils.io_pointcloud import PCLHolder


def read_cameras_list(path):
    """
    Loads the SFM camera list
    :param path: Path to the camera list file
    :return: [filepaths]
    """

    cameras_list = None

    with open(path, "r") as f:
        cameras_list = [line.split(' ')[0].rstrip() for line in f]

    return cameras_list


def parse_dataset(base_path, bundle_path, cameras_list_path, out_path):
    """
    Parses a Bundler SFM dataset and outputs a binary with all the data required
    to pre-process the dataset
    :param base_path: 
    :param bundle_path: 
    :param cameras_list_path: 
    :param output_path: 
    :return: 
    """

    # Read the dataset camera list
    cameras_list = read_cameras_list(cameras_list_path)

    pcl_parser = PCLHolder()

    pcl_parser.loadPCL(bundle_path)

    pcl_parser.load_descriptors(base_path, cameras_list)

    pcl_parser.write_binary(out_path)


if __name__ == "__main__":
    # Set the logging module
    lg.basicConfig(format='%(message)s', level=lg.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Set the argument options
    parser.add_argument('-p', action='store', dest='DATASET_PATH', help='Dataset path', default='')
    parser.add_argument('-o', action='store', dest='OUTPUT_PATH', help='Binary output path', default='')

    arg_v = parser.parse_args()

    # Path to the dataset folder
    dst_path = arg_v.DATASET_PATH
    output_path = arg_v.OUTPUT_PATH

    # Add / if not present
    if dst_path[-1] != '/':
        dst_path += '/'

    if output_path == '':
        output_path = dst_path + "bundle.bin"

    # Set the required paths to pre-process the dataset
    pcl_path = dst_path + "bundle/bundle.db.out"
    cameras_list_path = dst_path + "list.db.txt"

    parse_dataset(dst_path, pcl_path, cameras_list_path, output_path)



