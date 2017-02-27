from cv2 import SVDecomp, decomposeProjectionMatrix
import random
import numpy as np

##############
# References #
##############

# [1] https://www.graphics.rwth-aachen.de/software/image-localization
# [2] http://people.rennes.inria.fr/Eric.Marchand/pose-estimation/tutorial-pose-dlt-opencv.html

#################################################################
# ACG SPRT-RANSAC Implementation adapted from ACG-Localizer [1] #
#################################################################
class SPRTRANSACDLT:
    def __init__(self):
        self.SPRT_m_s = 1
        self.t_M = 200

        self.nb_SPRT_tests = 100

        self.epsilon_i = [0.0] * self.nb_SPRT_tests
        self.delta_i = [0.0] * self.nb_SPRT_tests
        self.A_i = [0.0] * self.nb_SPRT_tests
        self.k_i = [0.0] * self.nb_SPRT_tests
        self.h_i = [0.0] * self.nb_SPRT_tests
        self.nb_epsilon_values = 8
        self.nb_delta_values = 10

        self.max_number_of_LO_samples = 12 # TODO: keep this?
        self.nb_lo_steps = 10

        self.LOG_5_PER = -2.99573

        # By the paper efficient and effective
        self.error = np.sqrt(10)

        self.SPRT_eps_val = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.SPRT_delta_val = [0.01, 0.02, 0.07, 0.12, 0.17, 0.22, 0.27, 0.32, 0.37, 0.42]
        self.SPRT_h_i_val = np.array([ 1.0, 1.0, 1.0, 1.0015, 1.0249, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       1.6642, 1.7299, 2.1589, 3.062, 6.8894, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0007, 1.1308, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       2.3958, 2.511, 3.3098, 5.0339, 12.3835, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       1.4658, 1.4975, 1.6723, 1.9206, 2.3428, 3.2742, 7.2705, 0.0, 0.0, 0.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0001, 1.1615, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       3.2527, 3.4146, 4.5761, 7.1345, 18.1024, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       1.9981, 2.0543, 2.3846, 2.8683, 3.6984, 5.5358, 13.4329, 0.0, 0.0, 0.0,
                                       1.3779, 1.3972, 1.4959, 1.6163, 1.7805, 2.0283, 2.4583, 3.4147, 7.5328, 0.0,
                                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0001, 1.0001,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       4.2999, 4.515, 6.0786, 9.5662, 24.5884, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       2.6433, 2.7221, 3.2052, 3.932, 5.1915, 7.9902, 20.0375, 0.0, 0.0, 0.0,
                                       1.8283, 1.863, 2.0522, 2.2912, 2.6209, 3.121, 3.9915, 5.93, 14.282, 0.0,
                                       1.3367, 1.35, 1.4156, 1.4894, 1.5799, 1.6982, 1.8634, 2.1155, 2.5557, 3.5371,
                                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       5.65, 5.9326, 7.9949, 12.6205, 32.6092, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       3.4734, 3.5781, 4.2326, 5.2359, 6.9898, 10.9034, 27.7798, 0.0, 0.0, 0.0,
                                       2.404, 2.453, 2.7335, 3.0995, 3.6105, 4.3906, 5.7528, 8.7914, 21.8925, 0.0,
                                       1.7615, 1.7858, 1.9137, 2.0634, 2.2497, 2.4947, 2.8383, 3.364, 4.2829, 6.3331,
                                       1.3251, 1.3352, 1.3838, 1.4357, 1.4956, 1.5679, 1.659, 1.7796, 1.9491, 2.2087,
                                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       7.5527, 7.9306, 10.6887, 16.8849, 43.7002, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       4.6432, 4.7833, 5.6642, 7.0274, 9.4258, 14.7974, 38.0023, 0.0, 0.0, 0.0,
                                       3.2139, 3.2803, 3.6693, 4.1888, 4.9231, 6.0516, 8.0298, 12.4514, 31.5338, 0.0,
                                       2.356, 2.3911, 2.5854, 2.8216, 3.1203, 3.5167, 4.0755, 4.9328, 6.4339, 9.7866,
                                       1.7753, 1.7941, 1.8911, 2.0, 2.1278, 2.2836, 2.481, 2.7431, 3.1124, 3.6786,
                                       1.3458, 1.3541, 1.3935, 1.4344, 1.4796, 1.5315, 1.593, 1.6682, 1.7637, 1.8903,
                                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       10.8055, 11.3461, 15.2922, 24.1588, 62.5457, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       6.6429, 6.8433, 8.1047, 10.0615, 13.5147, 21.2668, 54.8035, 0.0, 0.0, 0.0,
                                       4.598, 4.6932, 5.2538, 6.0104, 7.0894, 8.758, 11.6945, 18.2739, 46.7041, 0.0,
                                       3.3708, 3.4216, 3.7094, 4.0683, 4.5298, 5.1483, 6.0258, 7.3778, 9.7513, 15.0601,
                                       2.5408, 2.5694, 2.726, 2.9098, 3.1304, 3.4028, 3.751, 4.2155, 4.8725, 5.8821,
                                       1.9284, 1.9444, 2.0277, 2.1195, 2.2239, 2.3455, 2.4908, 2.6696, 2.8972, 3.2,
                                       1.4386, 1.4461, 1.4822, 1.5192, 1.5591, 1.6033, 1.6535, 1.7119, 1.7814, 1.8666,
                                       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        self.SPRT_h_i_val.reshape((8, 8, 10)) # TODO: is this shape right?

    #################################
    # SPRT-RANSAC Support functions #
    #################################
    def sprt_compute_A(self, sig, eps):
        a = 1.0 - sig
        b = 1.0 - eps

        C = a * np.log(a / b) + sig * np.log(sig / eps)
        a = self.t_M * C / self.SPRT_m_s + 1.0

        A_0 = a
        A_1 = a + np.log(A_0)

        while abs(A_1 - A_0) > 1e-6:
            A_0 = A_1
            A_1 = a + np.log(A_0)

        return A_1

    def get_max_ransac_steps(self, inlier_ratio):
        # For 6 samples
        if inlier_ratio == 1:
            return 1

        real_ratio = inlier_ratio

        real_ratio = np.max((real_ratio, 0.029780))

        return np.ceil(self.LOG_5_PER / np.log(1.0 - real_ratio * real_ratio * real_ratio * real_ratio * real_ratio * real_ratio))

    def SPRT_get_max_sprt_ransac_steps(self, epsilon, l):
        # levenberg - marquardt variables
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        ff = 0
        fff = 0
        f_1 = 0
        A = 0
        g = 0
        x_old = 0
        x_new = 0
        h_lm = 0
        F_x_new = 0
        F_x_old = 0
        t = 0
        upsilon = 0
        mu = 0
        tmp_flt1 = 0
        tmp_flt2 = 0
        rho = 0
        k = 0
        found  = False

        # compute all the h_i
        idx_eps = 0
        idx_eps_i = 0
        idx_delta_i = 0
        # Timer
        #h_i_timer;
        #h_i_timer.Init();
        #h_i_timer.Start();

        for i in range(0, l):
            # do a table look-up to get a approximate starting value for h_i

            # first find a similar value for epsilon
            tmp_flt1 = abs(epsilon - self.SPRT_eps_val[0])
            idx_eps = 0

            for j in range(1, self.nb_epsilon_values):
                tmp_flt2 = abs(epsilon - self.SPRT_eps_val[j])

                if tmp_flt2 > tmp_flt1:
                    break
                else:
                    idx_eps += 1

            # now for epsilon_i
            tmp_flt1 = abs(self.epsilon_i[i] - self.SPRT_eps_val[0])
            idx_eps_i = 0

            for j in range(1, self.nb_epsilon_values):

                tmp_flt2 = abs(self.epsilon_i[i] - self.SPRT_eps_val[j])
                if tmp_flt2 > tmp_flt1:
                    break
                else:
                    idx_eps_i += 1

            # now for delta_i
            tmp_flt1 = abs(self.delta_i[i] - self.SPRT_delta_val[0])
            idx_delta_i = 0

            for j in range(1, self.nb_delta_values):
                tmp_flt2 = abs(self.epsilon_i[i] - self.SPRT_delta_val[j])
                if tmp_flt2 > tmp_flt1:
                    break
                else:
                    idx_delta_i += 1

            x_old = self.SPRT_h_i_val[idx_eps][idx_eps_i][idx_delta_i]

            if x_old == 0.0:
                x_old = 100.0

            x_new = x_old

            # now do levenberg - marquardt for refinement
            a = self.delta_i[i] / self.epsilon_i[i]
            b = (1.0-self.delta_i[i]) / (1.0 - self.epsilon_i[i])
            c = 1.0 - epsilon
            d = np.log(a)
            e = np.log(b)

            k = 0
            ff = np.power(a, x_old)
            fff = np.power(b, x_old)
            f_1 = epsilon * d * ff + c * e * fff
            A = f_1 * f_1
            F_x_new = F_x_old = epsilon * ff * +c * fff-1.0
            g = f_1 * F_x_old

            mu = 1e-6 * A
            upsilon = 2.0

            found = abs(g) <= 1e-6

            while not found and k < 200:
                k += 1

                h_lm = -g / (A + mu)
                found = abs(h_lm) <= 1e-6 * (abs(x_old) + 1e-6)
                x_new = x_old + h_lm
                F_x_new = epsilon * np.power(a, x_new) + c * np.power(b, x_new) - 1.0
                rho = 2.0 * (F_x_old * F_x_old - F_x_new * F_x_new) / (h_lm * (mu * h_lm - g))

                if rho > 0.0:
                    x_old = x_new
                    f_1 = epsilon * d * np.power(a, x_old) * ff + c * e * np.power(b, x_old)
                    A = f_1 * f_1
                    g = f_1 * F_x_new
                    F_x_old = F_x_new
                    found = abs(g) <= 1e-6
                    t = 2.0 * rho - 1.0
                    mu *= np.max((1.0 / 3.0, 1.0 - t * t * t ))
                    upsilon = 2.0

                else:
                    mu *= upsilon
                    upsilon *= 2.0
            # end while

            self.h_i[i] = x_old
        # end for
        #h_i_timer.Stop();

        # now we compute the probability eta(l - 1)

        eta_l_minus_1 = 0.0
        Pg = 1.0

        Pg = epsilon * epsilon * epsilon * epsilon * epsilon * epsilon

        for i in range(0, l):
            eta_l_minus_1 += np.log( 1.0 - Pg * (1.0 - np.power( self.A_i[i], -self.h_i[i] )) ) * self.k_i[i]

        numerator = self.LOG_5_PER - eta_l_minus_1

        if numerator >= 0.0:
            return 0

        return np.ceil(numerator / np.log(1.0 - Pg * (1.0 - 1.0 / self.A_i[l] )))

    def compute_residuals(self, pts3d, pts2d, P):
        residuals = np.zeros(len(pts3d))

        # Project 3D to 2D using P
        for i in range(len(pts3d)):
            h_pt2d = np.matmul(P, (np.array([pts3d[i][0], pts3d[i][1], pts3d[i][2], 1.0])).reshape(4, 1))

            newh = np.array([h_pt2d[0][0] / h_pt2d[2][0], h_pt2d[1][0] / h_pt2d[2][0]])

            # Compute delta errors
            residuals[i] = np.linalg.norm(pts2d[i] - newh)

        return residuals

    #########################
    # DLT Support functions #
    #########################
    def scaleCorrespondences(self, wX, x):
        """
        Normalizes the 3D-2D correspondences
        :param wX: 3D points
        :param x: 2D points
        :return: success, scaled3D, scaled2D, scaledImgMat, scaledWorldMat
        """
        # Center points for scaling
        center2D = np.array([0.0, 0.0])
        center3D = np.array([0.0, 0.0, 0.0])

        # Determine c.o.g (Center of Gravity?)
        n = 1.0
        for i in range(len(wX)):
            center2D = np.add(x[i] * (1.0/n), center2D * ((n-1.0) / n))
            center3D = np.add(wX[i] * (1.0 / n), center3D * ((n - 1.0) / n))

            n += 1.0

        # Move points to CoG
        n = 1.0
        scale3D = 0.0
        scale2D = 0.0

        scaledwX = []
        scaledx = []

        for i in range(len(wX)):
            pt2d = np.subtract(x[i], center2D)
            pt3d = np.subtract(wX[i], center3D)

            scale2D = scale2D * ((n - 1.0) / n) + np.linalg.norm(pt2d) * (1.0 / n)
            scale3D = scale3D * ((n - 1.0) / n) + np.linalg.norm(pt3d) * (1.0 / n)

            scaledx.append(pt2d)
            scaledwX.append(pt3d)

            n += 1.0

        scaledx = np.array(scaledx)
        scaledwX = np.array(scaledwX)

        if abs(scale2D) < 1e-12 or abs(scale3D) < 1e-12:
            return False, None, None, None, None

        scale2D = 1.41421 / scale2D # sqrt(2)
        scale3D = 1.73205 / scale3D # sqrt(3)

        for i in range(len(scaledwX)):
            scaledx[i] *= scale2D
            scaledwX[i] *= scale3D

        m_matScaleImgInv = np.zeros((3, 3))
        m_matScaleWorld = np.zeros((4, 4))

        m_matScaleImgInv[0][0] = 1.0 / scale2D
        m_matScaleImgInv[1][1] = 1.0 / scale2D
        m_matScaleImgInv[0][2] = center2D[0]
        m_matScaleImgInv[1][2] = center2D[1]
        m_matScaleImgInv[2][2] = 1.0

        m_matScaleWorld[0][0] = scale3D
        m_matScaleWorld[1][1] = scale3D
        m_matScaleWorld[2][2] = scale3D
        m_matScaleWorld[0][3] = -center3D[0] * scale3D
        m_matScaleWorld[1][3] = -center3D[1] * scale3D
        m_matScaleWorld[2][3] = -center3D[2] * scale3D

        m_matScaleWorld[3][3] = 1.0

        return True, scaledwX, scaledx, m_matScaleImgInv, m_matScaleWorld

    # 3D points, 2D points, outT, outR
    # Based on [2]
    def pose_dlt(self, unscaledwX, unscaledx):
        """
        Computes the P matrix from unscaled 3D-2D correspondences
        :param unscaledwX: unscaled 3D points
        :param unscaledx: unscaled 2D points
        :return: P matrix
        """

        # First scale coordinates (normalization to avoid large deviations when applying DLT)
        m_flag, wX, x, m_imgscale, m_worldscale = self.scaleCorrespondences(unscaledwX, unscaledx)

        if not m_flag:
            print "[ERROR] Computing wrong scaling in DLT..."

        ctw = np.zeros((3, 1))
        cRw = np.zeros((3, 3))

        npoints = len(wX)
        A = np.zeros((2 * npoints, 12))  # Mat, CV_64F, cv::Scalar(0));
        for i in range(npoints):  # Update matrix A using eq. 5
            A[2 * i][0] = wX[i][0]  # //wX[i][0] ;
            A[2 * i][1] = wX[i][1]
            A[2 * i][2] = wX[i][2]
            A[2 * i][3] = 1
            A[2 * i + 1][4] = wX[i][0]
            A[2 * i + 1][5] = wX[i][1]
            A[2 * i + 1][6] = wX[i][2]
            A[2 * i + 1][7] = 1
            A[2 * i][8] = - x[i][0] * wX[i][0]
            A[2 * i][9] = - x[i][0] * wX[i][1]
            A[2 * i][10] = - x[i][0] * wX[i][2]
            A[2 * i][11] = - x[i][0]
            A[2 * i + 1][8] = - x[i][1] * wX[i][0]
            A[2 * i + 1][9] = - x[i][1] * wX[i][1]
            A[2 * i + 1][10] = - x[i][1] * wX[i][2]
            A[2 * i + 1][11] = - x[i][1]

        w, u, vt = SVDecomp(A)  # cv2.svd.compute
        # w, u, vt = np.linalg.svd(A)

        smallestSv = w[0][0]  # double
        indexSmallestSv = 0  # unisnged int

        for i in range(1, w.shape[0]):
            if w[i][0] < smallestSv:
                smallestSv = w[i][0]
                indexSmallestSv = i

        h = vt[indexSmallestSv]  # Mat row

        if h[11] < 0:  # // tz < 0
            h *= -1

        # Normalization to ensure that ||r3|| = 1
        norm = np.sqrt(h[8] * h[8] + h[9] * h[9] + h[10] * h[10])
        h = h / norm

        for i in range(0, 3):
            ctw[i][0] = h[4 * i + 3]  # // Translation
            for j in range(0, 3):
                cRw[i][j] = h[4 * i + j]  # // Rotation

        P = np.array([[cRw[0][0], cRw[0][1], cRw[0][2], ctw[0][0]],
                      [cRw[1][0], cRw[1][1], cRw[1][2], ctw[1][0]],
                      [cRw[2][0], cRw[2][1], cRw[2][2], ctw[2][0]]])

        # Undo the rescaling
        # T.P.U
        P = np.matmul(m_imgscale, P)
        P = np.matmul(P, m_worldscale)

        return P

    def compute_pose_from_pmatrix(self, P):
        """
        Decomposes the P matrix into K, R, t, and the 3 euler angles
        :param P: Projection matrix
        :return:
        """
        K, R, t, eular_a, euler_b, euler_c, euler_d = decomposeProjectionMatrix(P)

        return np.array([t[0] / t[3], t[1] / t[3], t[2] / t[3]])

    def sprt_ransac_p6pdlt(self, pts3d, pts2d, min_inlier_ratio, number_of_samples):
        """
        SPRT-RANSAC function converted from c++ ACG-Localizer [1]
        :param pts3d: 3D points
        :param pts2d: 2D points
        :param min_inlier_ratio: min. inlier ratio to consider a pose
        :param number_of_samples:
        :return: sucess: 3x1 position, n. inliers | fail: None, n. inliers
        """
        # Size of the best inlier set
        size_inlier_set = 0

        nb_correspondences = len(pts3d) #10 # TODO:

        minimal_consensus_size = number_of_samples

        # TODO: this
        # If the number of correspondences is lower than the number of required points to dlt
        if nb_correspondences < minimal_consensus_size:
            return None, 0

        # Number of inliers found
        inlier_found = 0

        # Compute inlier ratio
        inlier_ratio = np.max((min_inlier_ratio, float(number_of_samples) / float(nb_correspondences)))

        # Max steps ransac has to take
        max_steps = self.get_max_ransac_steps(inlier_ratio)

        #
        new_inlier_found = 0

        inlier = []
        storedP = None
        #storedR = None
        #storedt = None

        self.epsilon_i[0] = inlier_ratio
        self.delta_i[0] = 0.01
        self.A_i[0] = self.sprt_compute_A(self.epsilon_i[0], self.delta_i[0])

        current_test = 0
        old_test = -1

        """elapsed_time_total = 0.0
        elapsed_time_sec = 0.0
        old_time = 0.0
        sum_time = 0.0
        old_mins = 0
        elapsed_mins = 0
        nb_LO_samples = 0 #?
        counter = 0
        time_out_timer = None
        time_out_timer.Init()
        time_out_timer.Start()"""

        eval_1 = self.delta_i[0] / self.epsilon_i[0]
        eval_0 = (1.0 - self.delta_i[0]) / (1.0 - self.epsilon_i[0])
        lambda_v = 1.0

        delta_hat = self.delta_i[0]
        old_ratio = inlier_ratio
        nb_rejected = 1.0
        bad_model = False

        taken_samples = 0
        self.k_i[0] = 0


        while True:

            while taken_samples < max_steps:# """ 0.01 """;){
                # TODO: is this required?
                """if stop_after_n_secs:
                    elapsed_time_sec = time_out_timer.GetElapsedTime()
                    time_out_timer.Stop()
                    time_out_timer.Restart()

                    if elapsed_time_sec >= 1.0:
                        elapsed_time_total += elapsed_time_sec
                        elapsed_time_sec = 0.0
                        time_out_timer.Stop()
                        time_out_timer.Start()
                        elapsed_mins = int(floor(elapsed_time_total)) / 60

                        if elapsed_mins > old_mins and elapsed_mins > 0:

                            print "[RANSAC] elapsed time so far " + str(elapsed_time_total) + " s"
                            old_mins = elapsed_mins


                    old_time = sum_time
                    sum_time = elapsed_time_sec + elapsed_time_total

                    if sum_time >= max_time:
                        print "[RANSAC] Warning: RANSAC reached time limit of " + str(max_time) + " s and was stopped"
                        break

                    if old_time == sum_time and old_time > 0.0:
                        print "[RANSAC] Warning: Time did not change value (possibly due to nummerical reasons), so we abort after " + str(old_time) + " s"
                        break
                """

                # Increment n samples
                taken_samples += 1
                self.k_i[current_test] += 1

                if taken_samples == 0:
                    taken_samples -= 1

                    print "[RANSAC] Error: The number of samples taken exceeds " + str(taken_samples) + " ( 2^64-1 ) which is the maximal number representable by a uint32_t."
                    print "[RANSAC] Error: Therefore, we stop searching for a better model here. (Maybe you want to use SCRAMSAC, if applicable, to get rid of outlier!)"
                    break


                # take a random sample from the set of correspondences
                pseudo_random_indexes = random.sample(xrange(len(pts3d)), 6)

                # add the correspondences
                x = []
                wX = []
                for pr in pseudo_random_indexes:
                    x.append(pts2d[pr])
                    wX.append(pts3d[pr])
                x = np.array(x)
                wX = np.array(wX)

                # Compute model
                P = self.pose_dlt(wX, x)

                # compute number of inlier to the hypothesis, evaluate the current SPRT test inlier_found = 0;
                inlier_found = 0
                lambda_i = 1.0
                bad_model = False

                # Compute residuals
                residuals = self.compute_residuals(wX, x, P)

                for i in range(len(wX)):
                    # If it is an inlier
                    #Compute distance between projection
                    if residuals[i] <= self.error:
                        lambda_i *= eval_1
                        inlier_found += 1
                    else:
                        lambda_i *= eval_0

                    if lambda_i > self.A_i[current_test]:
                        bad_model = True
                        break


                # Algorithm 2, if model is rejected
                if bad_model:
                    # check if we have to design a new test
                    nb_rejected += 1.0
                    delta_hat = delta_hat * (nb_rejected - 1.0) / nb_rejected + float(inlier_found) / (float(nb_correspondences) * nb_rejected)

                    if abs(delta_hat - self.delta_i[current_test]) > 0.05:
                        current_test += 1
                        self.k_i[current_test] = 0
                        self.epsilon_i[current_test] = self.epsilon_i[current_test-1]
                        self.delta_i[current_test] = delta_hat
                        self.A_i[current_test] = self.sprt_compute_A( self.epsilon_i[current_test], delta_hat )

                    continue

                # Algorithm 2, If model is accepted and biggest inliers found so far
                # compare found inliers to the biggest set of correspondences found so far
                if inlier_found > size_inlier_set:#{
                    # store hypothesis and update inlier ratio
                    storedP = P

                    # compute inlier
                    inlier = []
                    for ii in range(len(residuals)):
                        if residuals[ii] < self.error:
                            inlier.append(pseudo_random_indexes[ii])

                    #TODO: this optimization
                    # do Local Optimization(LO) - steps( if possible!)
                    """
                    nb_LO_samples = np.max((self.number_of_samples, np.min((inlier_found / 2, self.max_number_of_LO_samples))))
                    for lo_steps in range(self.nb_lo_steps):
                        #random_number_gen.generate_pseudorandom_numbers_unique( (uint32_t) 0, (uint32_t) (inlier.size() - 1), nb_LO_samples, LO_randomly_choosen_corr_indices );
                        pseudo_random_inliers = random.sample(xrange(nb_LO_samples), 6)
                        # generate hypothesis
                        # add the correspondences
                        #clear_solver();

                        owX = []
                        ox = []

                        for i in pseudo_random_inliers:
                            owX.append(pts3d[inlier[i]])
                            ox.append(pts2d[inlier[i]])

                        owX = np.array(owX)
                        ox = np.array(ox)

                        #for counter in range(0, nb_LO_samples):
                        #    add_correspondence(c1, c2, inlier[LO_randomly_choosen_corr_indices[counter]]);


                        # compute hypothesis
                        #if (!solve_system())
                        #    continue;
                        loR, lot = pose_dlt(owX, ox)

                        loP = np.array([[loR[0][0], loR[0][1], loR[0][2], lot[0][0]],
                                      [loR[1][0], loR[1][1], loR[1][2], lot[1][0]],
                                      [loR[2][0], loR[2][1], loR[2][2], lot[2][0]]])


                        # compute inlier to the hypothesis
                        new_found_inlier = 0
                        lo_residuals = compute_residuals(owX, ox, loP)

                        #for (std::vector < uint32_t >::const_iterator it = correspondence_indices.begin(); it != correspondence_indices.end(); ++it ){
                        for res in residuals:
                        #    if (evaluate_correspondece(c1, c2, *it)):
                            if res <= self.error:
                                new_found_inlier += 1
                        #}

                        # update found model if new best hypothesis
                        if new_found_inlier > inlier_found:
                            inlier_found = new_found_inlier
                            storedP = loP #store_hypothesis();
                    """

                    old_ratio = inlier_ratio
                    inlier_ratio = np.max((inlier_ratio, float(inlier_found) / float(nb_correspondences )))
                    max_steps = self.get_max_ransac_steps(inlier_ratio)
                    size_inlier_set = inlier_found

                    # design a new test if needed
                    # the check must be done to avoid designing a test for an inlier ratio below the specified minimal inlier ratio
                    if  old_ratio < inlier_ratio:
                        current_test += 1
                        self.k_i[current_test] = 0
                        self.epsilon_i[current_test] = inlier_ratio
                        self.delta_i[current_test] = delta_hat
                        self.A_i[current_test] = self.sprt_compute_A(inlier_ratio, delta_hat)
                # end Alg 2, if biggest support
            # end for


            # Max ransac steps
            # adjust the number of steps SPRT RANSAC has to take
            if old_test != current_test:
                old_test = current_test
                max_steps = self.SPRT_get_max_sprt_ransac_steps(inlier_ratio, current_test)
            else:
                break

        # end while

        Cmat = None

        if storedP is not None:
            Cmat = self.compute_pose_from_pmatrix(storedP)

        return Cmat, size_inlier_set



