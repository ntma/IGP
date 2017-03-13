from src.utils.geometry import compute_SVD, project_point
from src.utils.geometry import euclidean_distance
import random
import numpy as np
import logging as lg


##############
# References #
##############

# [1] https://www.graphics.rwth-aachen.de/software/image-localization

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

        self.number_of_samples = 6

        self.max_number_of_LO_samples = 12 # TODO: keep this?
        self.nb_lo_steps = 20

        self.LOG_5_PER = -2.99573

        # By the paper efficient and effective (squared reprojection error)
        self.reproj_error = 10

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

            x_old = self.SPRT_h_i_val[idx_eps * 8 * 10 + idx_eps_i * 10 + idx_delta_i]

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

    def evaluate_correspondence(self, pt3d, pt2d, P):

        projpoint = project_point(P, pt3d)

        return euclidean_distance(pt2d, projpoint)

    def compute_pseudo_random_indexes(self, pts3d, pts2d):
        # take a random sample from the set of correspondences
        pseudo_random_indexes = random.sample(xrange(len(pts3d)), 6)

        # add the correspondences
        x  = pts2d[pseudo_random_indexes] # numpy hack with list of indexes
        wX = pts3d[pseudo_random_indexes] # numpy hack with list of indexes

        return wX, x

    #########################
    # DLT Support functions #
    #########################
    def scale_correspondences(self, wX, x):
        """
        Normalizes the 3D-2D correspondences
        :param wX: 3D points
        :param x: 2D points
        :return: success, scaled3D, scaled2D, scaledImgMat, scaledWorldMat
        """
        # Center points for scaling
        center2D = np.zeros(2)#np.array([0.0, 0.0])
        center3D = np.zeros(3)#np.array([0.0, 0.0, 0.0])

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

        scaledwX = wX.copy()
        scaledx = x.copy()

        for i in range(len(wX)):
            scaledx[i] -= center2D
            scaledwX[i] -= center3D

            scale2D = scale2D * ((n - 1.0) / n) + np.linalg.norm(scaledx[i]) * (1.0 / n)
            scale3D = scale3D * ((n - 1.0) / n) + np.linalg.norm(scaledwX[i]) * (1.0 / n)

            n += 1.0

        if abs(scale2D) < 1e-12 or abs(scale3D) < 1e-12:
            return False, None, None, None, None

        scale2D = 1.41421 / scale2D  # sqrt(2)
        scale3D = 1.73205 / scale3D  # sqrt(3)

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

    def pose_dlt_acg(self, unscaledwX, unscaledx):
        endCorr = len(unscaledwX)

        nrows = 3 * endCorr
        ncols = 12

        mat_A = np.zeros((nrows, ncols))
        vec_point = np.zeros(4)

        m_flag, wX, x, m_imgscale, m_worldscale = self.scale_correspondences(unscaledwX, unscaledx)

        if not m_flag:
            return None

        for corr in range(0, endCorr):
            vec_point[0] = wX[corr][0]
            vec_point[1] = wX[corr][1]
            vec_point[2] = wX[corr][2]
            vec_point[3] = 1.0

            mat_A[3*corr][0] = - vec_point[0]
            mat_A[3*corr][1] = - vec_point[1]
            mat_A[3*corr][2] = - vec_point[2]
            mat_A[3*corr][3] = - vec_point[3]

            mat_A[3*corr][8] = vec_point[0] * x[corr][0]
            mat_A[3*corr][9] = vec_point[1] * x[corr][0]
            mat_A[3*corr][10] = vec_point[2] * x[corr][0]
            mat_A[3*corr][11] = vec_point[3] * x[corr][0]

            mat_A[3*corr + 1][4] = - vec_point[0]
            mat_A[3*corr + 1][5] = - vec_point[1]
            mat_A[3*corr + 1][6] = - vec_point[2]
            mat_A[3*corr + 1][7] = - vec_point[3]

            mat_A[3*corr + 1][8] = vec_point[0] * x[corr][1]
            mat_A[3*corr + 1][9] = vec_point[1] * x[corr][1]
            mat_A[3*corr + 1][10] = vec_point[2] * x[corr][1]
            mat_A[3*corr + 1][11] = vec_point[3] * x[corr][1]

            mat_A[3*corr + 2][0] = - vec_point[0] * x[corr][1]
            mat_A[3*corr + 2][1] = - vec_point[1] * x[corr][1]
            mat_A[3*corr + 2][2] = - vec_point[2] * x[corr][1]
            mat_A[3*corr + 2][3] = - vec_point[3] * x[corr][1]

            mat_A[3*corr + 2][4] = vec_point[0] * x[corr][0]
            mat_A[3*corr + 2][5] = vec_point[1] * x[corr][0]
            mat_A[3*corr + 2][6] = vec_point[2] * x[corr][0]
            mat_A[3*corr + 2][7] = vec_point[3] * x[corr][0]

        _, _, mat_VT = compute_SVD(mat_A)

        m_projectionMatrix = np.zeros((3, 4))
        for row in range(3):
            for col in range(4):
                m_projectionMatrix[row][col] = mat_VT[11][4 * row + col]

        m_projectionMatrix = np.matmul(m_imgscale, m_projectionMatrix)
        m_projectionMatrix = np.matmul(m_projectionMatrix, m_worldscale)

        return m_projectionMatrix

    def compute_pose_from_pmatrix(self, m_projectionMatrix):
        """
        Decomposes the P matrix into K, R, t, and the 3 euler angles
        :param P: Projection matrix
        :return:
        """

        _, _, mat_VT = compute_SVD(m_projectionMatrix)

        position = np.zeros(3)

        for i in range(0, 3):
            position[i] = mat_VT[3][i] / mat_VT[3][3]

        # get the viewing direction of the camera, see Hartley & Zisserman, 2nd ed., pages 160 - 161
        # compute the determinant of the 3x3 part of the projection matrix
        # det = m_projectionMatrix[0][0] * m_projectionMatrix[1][1] * m_projectionMatrix[2][2] +
        #       m_projectionMatrix[0][1] * m_projectionMatrix[1][2] * m_projectionMatrix[2][0] +
        #       m_projectionMatrix[0][2] * m_projectionMatrix[1][0] * m_projectionMatrix[2][1] -
        #       m_projectionMatrix[0][2] * m_projectionMatrix[1][1] * m_projectionMatrix[2][0] -
        #       m_projectionMatrix[0][1] * m_projectionMatrix[1][0] * m_projectionMatrix[2][2] -
        #       m_projectionMatrix[0][0] * m_projectionMatrix[1][2] * m_projectionMatrix[2][1]

        # remember that the camera in reconstructions computed by Bundler looks
        # down the negative z-axis instead of the positive z-axis.
        # So we have to multiply the orientation with -1.0
        # for i in range(0, 3):
        #    orientation[i] = - m_projectionMatrix[2][i] * det

        return position

    def sprt_ransac_p6pdlt(self, pts3d, pts2d, nb_correspondences, min_inlier_ratio):
        """
        SPRT-RANSAC function converted from c++ ACG-Localizer [1]
        :param pts3d: 3D points
        :param pts2d: 2D points
        :param nb_correspondences: Number of correspondences
        :param min_inlier_ratio: min. inlier ratio to consider a pose
        :return: sucess: P matrix, n. inliers | fail: None, n. inliers
        """

        if nb_correspondences < self.number_of_samples:
            return None, 0

        # Set the inlier ratio
        inlier_ratio = np.max((min_inlier_ratio, float(self.number_of_samples) / float(nb_correspondences)))


        # Max steps ransac has to take
        max_steps = self.get_max_ransac_steps(inlier_ratio)

        # Size of the best inlier set
        size_inlier_set = self.number_of_samples - 1

        new_inlier_found = 0

        # Initialize the 0-th SPRT test
        self.epsilon_i[0] = inlier_ratio
        self.delta_i[0] = 0.01
        self.A_i[0] = self.sprt_compute_A(self.epsilon_i[0], self.delta_i[0])

        current_test = 0
        old_test = -1


        eval_1 = self.delta_i[0] / self.epsilon_i[0]
        eval_0 = (1.0 - self.delta_i[0]) / (1.0 - self.epsilon_i[0])
        lambda_v = 1.0

        delta_hat = self.delta_i[0]
        old_ratio = inlier_ratio
        nb_rejected = 1.0
        bad_model = False

        taken_samples = 0
        self.k_i[0] = 0

        #inlier = []
        storedP = None

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

                    lg.debug("[RANSAC] Error: The number of samples taken exceeds " + str(taken_samples) + " ( 2^64-1 ) which is the maximal number representable by a uint32_t.")
                    lg.debug("[RANSAC] Error: Therefore, we stop searching for a better model here. (Maybe you want to use SCRAMSAC, if applicable, to get rid of outlier!)")
                    break

                # Random sampling
                wX, x = self.compute_pseudo_random_indexes(pts3d, pts2d)

                # Compute model
                P = self.pose_dlt_acg(wX, x)

                if P is None:
                    continue

                # compute number of inlier to the hypothesis, evaluate the current SPRT test
                inlier_found = 0
                lambda_v = 1.0
                bad_model = False

                # Compute residuals
                for i in range(len(pts3d)):
                    residual = self.evaluate_correspondence(pts3d[i], pts2d[i], P)

                    # If it is an inlier
                    # Compute distance between projection
                    if residual <= self.reproj_error:
                        lambda_v *= eval_1
                        inlier_found += 1
                    else:
                        lambda_v *= eval_0

                    if lambda_v > self.A_i[current_test]:
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
                    for ii in range(len(pts3d)):
                        if self.evaluate_correspondence(pts3d[ii], pts2d[ii], P) <= self.reproj_error:
                            inlier.append(ii)

                    # do Local Optimization(LO) - steps( if possible!)
                    nb_LO_samples = np.max((self.number_of_samples, np.min((inlier_found / 2, self.max_number_of_LO_samples))))
                    for lo_steps in range(self.nb_lo_steps):

                        pseudo_random_inliers = random.sample(xrange(len(inlier)), nb_LO_samples)
                        # generate hypothesis
                        # add the correspondences

                        owX = []
                        ox = []

                        for i in pseudo_random_inliers:
                            owX.append(pts3d[inlier[i]])
                            ox.append(pts2d[inlier[i]])

                        owX = np.array(owX)
                        ox = np.array(ox)

                        loP = self.pose_dlt_acg(owX, ox)

                        if loP is None:
                            continue

                        # compute inlier to the hypothesis
                        new_found_inlier = 0

                        for i in range(len(pts3d)):
                            lo_residual = self.evaluate_correspondence(pts3d[i], pts2d[i], loP)

                            if lo_residual <= self.reproj_error:
                                new_found_inlier += 1

                        # update found model if new best hypothesis
                        if new_found_inlier > inlier_found:
                            inlier_found = new_found_inlier
                            storedP = loP

                    old_ratio = inlier_ratio
                    inlier_ratio = max((inlier_ratio, float(inlier_found) / float(nb_correspondences )))
                    max_steps = self.get_max_ransac_steps(inlier_ratio)
                    size_inlier_set = inlier_found

                    # Design a new test if needed.
                    # The check must be done to avoid designing a test for an inlier
                    # ratio below the specified minimal inlier ratio.
                    if old_ratio < inlier_ratio:
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

        lg.debug("[RANSAC] SPRT-LO-RANSAC took " + str(taken_samples) + " samples using " + str(current_test + 1) + " SPRTs, found " + str(size_inlier_set) + " inlier ( " +str(inlier_ratio) + " % ) ")

        # end while
        return storedP, size_inlier_set

    def compute_pose(self, pts3d, pts2d, nb_correspondences, min_inlier_ratio):
        """
        Computes and validates the hypothesised P matrix using the found correspondences
        :param pts3d: 3D correspondences
        :param pts2d: 2D correspondences
        :param nb_correspondences: number of correspondences
        :param min_inlier_ratio: min. inlier ratio
        :return: Success: P, n. inliers | Fail: None, n. inliers
        """

        # Hypothesise P matrix using the found correspondences
        P, n_inliers = self.sprt_ransac_p6pdlt(pts3d, pts2d, nb_correspondences, min_inlier_ratio)

        # If P was successful, then we compute the actual number of inliers
        if P is not None and 6 <= n_inliers:

            n_inliers = 0

            for i in range(len(pts3d)):

                residual = self.evaluate_correspondence(pts3d[i], pts2d[i], P)

                if residual <= self.reproj_error:
                    n_inliers += 1

            return self.compute_pose_from_pmatrix(P), n_inliers
        else:
            return None, n_inliers
