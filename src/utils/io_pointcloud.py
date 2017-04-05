import numpy as np
import struct

from src.utils.io_datasets import read_sift_file

float4 = np.float32
float8 = np.float64

"""
    http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6

    BUNDLER FILE STRUCTURE
    # Bundle file v0.3
    <num_cameras> <num_points>   [two integers]
    <camera1>
    <camera2>
       ...
    <cameraN>
    <point1>
    <point2>
       ...
    <pointM>

    <f> <k1> <k2>   [the focal length, followed by two radial distortion coeffs]
    <R>             [a 3x3 matrix representing the camera rotation]
    <t>             [a 3-vector describing the camera translation]

    <position>      [a 3-vector describing the 3D position of the point]
    <color>         [a 3-vector describing the RGB color of the point]
    <view list>     [a list of views the point is visible in]
"""


class PCLHolder:
    def __init__(self):
        self.cameras = []
        self.camera_paths = None

        self.pts3D = []
        self.reprojections = []
        self.descriptors = None

        self.n_points = 0
        self.n_cameras = 0
        self.n_descriptors = 0

    def loadPCL(self, filename):

        with open(filename, "r") as f:

            # Read header
            dummy = f.readline()

            # Read n_cameras, n_points
            line = f.readline()[:-1].split(' ')

            self.n_cameras = int(line[0])
            self.n_points  = int(line[1])

            # Read cameras
            for i in xrange(self.n_cameras):
                line = f.readline()[:-1].split(' ')

                focal = float8(line[0])
                k1    = float8(line[1])
                k2    = float8(line[2])

                R = np.array([0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0], dtype=float8)

                Rs = f.readline()[:-1].split(' ')
                R[0] = float8(Rs[0])
                R[1] = float8(Rs[1])
                R[2] = float8(Rs[2])

                Rs = f.readline()[:-1].split(' ')
                R[3] = float8(Rs[0])
                R[4] = float8(Rs[1])
                R[5] = float8(Rs[2])

                Rs = f.readline()[:-1].split(' ')
                R[6] = float8(Rs[0])
                R[7] = float8(Rs[1])
                R[8] = float8(Rs[2])

                ts = f.readline()[:-1].split(' ')

                t = np.array([0.0, 0.0, 0.0], dtype=float8)

                t[0] = float8(ts[0])
                t[1] = float8(ts[1])
                t[2] = float8(ts[2])

                c = (focal, k1, k2, R, t)

                self.cameras.append(c)

            self.pts3D = np.zeros((self.n_points, 3), dtype=float4)
            self.reprojections = [[] for i in range(self.n_points)]

            # Read 3D points
            for i in range(self.n_points):
                line = f.readline()[:-1].split(' ')

                self.pts3D[i][0] = float4(line[0])
                self.pts3D[i][1] = float4(line[1])
                self.pts3D[i][2] = float4(line[2])

                line = f.readline()[:-1].split(' ')

                # We ignore RGB values

                line = f.readline()[:-1].split(' ')

                sz_view_list = int(line[0])

                view_list = [0.0] * (sz_view_list * 2)

                for j in range(sz_view_list):
                    step = j*4

                    view_list[j * 2 + 0] = int(line[1 + step])
                    view_list[j * 2 + 1] = int(line[1 + step + 1])

                    self.n_descriptors += 1

                self.reprojections[i] = view_list

                if i % 100000 == 0:
                    print "On going %i " % i

            print " Read :"
            print "   %i Cameras" % self.n_cameras
            print "   %i 3D points " % self.n_points
            print "   %i Projections " % self.n_descriptors

        return True

    def load_descriptors(self, base_path, cameras_list):

        camera_to_descriptor = [[] for i in range(self.n_cameras)]

        for it, v in enumerate(self.reprojections):
            for i in xrange(0, len(v), 2):
                camera_to_descriptor[v[i]].append((v[i + 1], it))

        self.descriptors = np.zeros((self.n_descriptors, 128), dtype=np.uint8)

        desc_counter = [0 for i in xrange(len(self.pts3D))]
        desc_bound = [0 for i in xrange(len(self.pts3D))]

        sum_idx = 0
        for i, v in enumerate(self.reprojections):
            desc_bound[i] = sum_idx

            sum_idx += int(len(v) / 2)

        desc_id = 0

        for cid, camera_path in enumerate(cameras_list):
            kpts, descs = read_sift_file(base_path + camera_path[:-4] + ".key", False)

            for it in camera_to_descriptor[cid]:
                d_idx = it[0]
                pt_idx = it[1]

                self.descriptors[desc_bound[pt_idx] + desc_counter[pt_idx]] = descs[d_idx]

                desc_counter[pt_idx] += 1

            desc_id += len(camera_to_descriptor[cid])

            if cid % 1000 == 0:
                print "On going camera %i" % cid

    def load_camera_paths(self, filename):
        with open(filename, "r") as f:
            self.camera_paths = [line.split(' ')[0].rstrip() for line in f]

    def load_only_cameras(self, filename):

        f = None

        try:
            f = open(filename, "r")
        except IOError:
            print "Could not open file: " + filename
            return False

        # Read header
        dummy = f.readline()

        # Read n_cameras, n_points
        line = f.readline()[:-1].split(' ')

        n_cameras = int(line[0])
        n_points  = int(line[1])

        # Read cameras
        for i in range(n_cameras):
            line = f.readline()[:-1].split(' ')

            focal = float8(line[0])
            k1 = float8(line[1])
            k2 = float8(line[2])

            R = np.array([0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0], dtype=float8)

            Rs = (f.readline()[:-1].split(' '))
            R[0] = float8(Rs[0])
            R[1] = float8(Rs[1])
            R[2] = float8(Rs[2])

            Rs = (f.readline()[:-1].split(' '))
            R[3] = float8(Rs[0])
            R[4] = float8(Rs[1])
            R[5] = float8(Rs[2])

            Rs = (f.readline()[:-1].split(' '))
            R[6] = float8(Rs[0])
            R[7] = float8(Rs[1])
            R[8] = float8(Rs[2])

            ts = f.readline()[:-1].split(' ')

            t = np.array([0.0, 0.0, 0.0], dtype=float8)

            t[0] = float8(ts[0])
            t[1] = float8(ts[1])
            t[2] = float8(ts[2])

            c = (focal, k1, k2, R, t)

            self.cameras.append(c)
        f.close()

        return True

    def load_binary(self, filename):

        with open(filename, "rb") as f:

            print "Reading cameras"
            n_cameras = struct.unpack("i", f.read(4))[0]

            for i in xrange(n_cameras):
                focal = struct.unpack("d", f.read(8))[0]
                k1 = struct.unpack("d", f.read(8))[0]
                k2 = struct.unpack("d", f.read(8))[0]

                R = np.array([0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0], dtype=float)

                for j in xrange(9):
                    R[j] = struct.unpack("d", f.read(8))[0]

                T = np.array([0.0, 0.0, 0.0], dtype=float)

                for j in xrange(3):
                    T[j] = struct.unpack("d", f.read(8))[0]

                self.cameras.append((focal, k1, k2, R, T))

                if i % 1000 == 0:
                    print "On going %i" % i

            print "Reading points and reprojections"

            n_points = struct.unpack("i", f.read(4))[0]
            n_descriptors = struct.unpack("i", f.read(4))[0]

            self.pts3D = np.zeros((n_points, 3), dtype=float4)
            self.descriptors = np.zeros((n_descriptors, 128), dtype=np.uint8)

            d_idx = 0

            for i in xrange(n_points):
                self.pts3D[i][0] = struct.unpack("f", f.read(4))[0]
                self.pts3D[i][1] = struct.unpack("f", f.read(4))[0]
                self.pts3D[i][2] = struct.unpack("f", f.read(4))[0]

                n_reprojections = struct.unpack("i", f.read(4))[0]

                reproj = []

                for j in xrange(0, n_reprojections, 2):
                    cam_id = struct.unpack("i", f.read(4))[0]
                    feat_id = struct.unpack("i", f.read(4))[0]

                    reproj.append(cam_id)
                    reproj.append(feat_id)

                    self.descriptors[d_idx] = np.array(struct.unpack("B"*128, f.read(128)), dtype=np.uint8)

                    d_idx += 1

                self.reprojections.append(reproj)

                if i % 100000 == 0:
                    print "On going %i" % i

            print "Loaded: "
            print "   %i cameras" % n_cameras
            print "   %i points" % n_points
            print "   %i reprojections" % d_idx

            f.close()

    def write_binary(self, filename):

        with open(filename, "wb") as f:

            f.write(struct.pack("i", len(self.cameras)))

            for i in self.cameras:
                f.write(struct.pack("d", i[0]))
                f.write(struct.pack("d", i[1]))
                f.write(struct.pack("d", i[2]))
                f.write(struct.pack("d" * 9, *i[3]))
                f.write(struct.pack("d" * 3, *i[4]))

            f.write(struct.pack("i", len(self.pts3D)))
            f.write(struct.pack("i", len(self.descriptors)))

            desc_iter = 0

            for i, pt in enumerate(self.pts3D):

                f.write(struct.pack("f" * 3, *pt))

                f.write(struct.pack("i", len(self.reprojections[i])))

                for j in range(0, len(self.reprojections[i]), 2):
                    f.write(struct.pack("i", self.reprojections[i][j]))
                    f.write(struct.pack("i", self.reprojections[i][j + 1]))
                    f.write(struct.pack("B" * 128, *self.descriptors[desc_iter]))

                    desc_iter += 1

            f.close()
