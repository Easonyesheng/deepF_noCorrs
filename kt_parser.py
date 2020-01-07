"""
parse a calib_cam_to_cam.txt.
'''
Use this code to calculate Fundamental Matrix 
'''
"""
import os
import cv2
import numpy as np

class KittiParamParser(object):
    def __init__(self, calib_path):
        self.path = calib_path
        self.calib = self.parse()

    def parse(self):
        d = {}
        with open(self.path) as f:
            for l in f:
                if l.startswith("calib_time"):
                    d["calib_time"] = l[l.index("calib_time")+1:]
                else:
                    [k,v] = l.split(":")
                    k,v = k.strip(), v.strip()
                    #get the numbers out
                    v = [float(x) for x in v.strip().split(" ")]
                    v = np.array(v)
                    if len(v) == 9:
                        v = v.reshape((3,3))
                    elif len(v) == 3:
                        v = v.reshape((3,1))
                    d[k] = v
        return d

    def get_F(self, f_cam, t_cam):
        """Returns fundamental matrix from f_cam to t_cam, according to
        the following formula:
        F = K2^(-T)*R*[t]x*K1^(-1)"""
        # #assemble the ingredients
        # K1, K2 = self.calib['K_0{}'.format(f_cam)], self.calib['K_0{}'.format(t_cam)]
        # R1, R2 = self.calib['R_0{}'.format(f_cam)], self.calib['R_0{}'.format(t_cam)]
        # # R1, R2 = self.calib['R_rect_0{}'.format(f_cam)], self.calib['R_rect_0{}'.format(t_cam)]
        # t1, t2 = self.calib['T_0{}'.format(f_cam)], self.calib['T_0{}'.format(t_cam)]

        # print(f"K1: {K1}, K2: {K2}, R1: {R1}, R2: {R2}, t1: {t1}, t2: {t2}")


        # R = np.dot(R2, np.linalg.inv(R1))
        # t = t2 - t1
        
        # T = np.array([
        #     [0,     -t[2], t[1]],
        #     [t[2],  0,     -t[0]],
        #     [-t[1], t[0],  0]
        # ])
        # #compute
        # F = np.dot(np.linalg.inv(K2.T), np.dot(T, np.dot(R, np.linalg.inv(K1))))
        # F /= F[2,2]
        # assert np.linalg.matrix_rank(F) == 2
        """
        Calculate the F by 
        F = [e']P'P^+
        where e' = P'C
        where PC = 0
        """
        P, P_ = self.calib['P_rect_0{}'.format(f_cam)], self.calib['P_rect_0{}'.format(t_cam)]
        P = P.reshape(3,4)
        P_ = P_.reshape(3,4)
        # print('P: ',P)
        P_c = P[:,:3]
        zero = P[:,3:]
        zero = -1*zero
        c = np.linalg.solve(P_c,zero)
        C = np.ones([4,1])
        C[:3,:] = c
        e_ = np.dot(P_,C)
        e_M = np.array([
            [0, -e_[2,0], e_[1,0]],
            [e_[2,0], 0, -e_[0,0]],
            [-e_[1,0], e_[0,0], 0]
        ])
        P = np.matrix(P)
        P_wn = np.linalg.pinv(P)
        F = np.dot(np.dot(e_M, P_),P_wn)
        F = F/F[2,2]

        return F

    def undistort(self, img, cam_id):
        D = self.calib['D_00']
        K = self.calib['K_00']
        return cv2.undistort(img, K, D)

if __name__ == "__main__":
    p = KittiParamParser('./data/kitti/calib_cam_to_cam.txt')
    print(p.get_F('0','1'))