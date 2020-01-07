"""
Use Predicted F to draw Epipolar lines
@Eason
"""

import cv2
import numpy as np 

'''
input : Gray Image & lines & dots
'''

keywords = 'Single_epW-1'

def drawlines(imgl, imgr, lines, pts1, pts2 ):
    pts1 = pts1[20:60]
    pts2 = pts2[20:60]
    # imgl = img1.copy()
    # imgr = img2.copy()
    r, c, h = imgl.shape

    i = 0
    for r, pt1, pt2 in zip(lines,pts1,pts2):
        i+=1
        color = np.random.randint(0,255,3).tolist()
        x0,y0 = map(int, [0,-r[2]/r[1]])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])

        

        imgl = cv2.line(imgl, (x0,y0), (x1,y1), color ,1)
        imgl = cv2.circle(imgl, tuple(pt1),5,color,-1)
        imgr = cv2.circle(imgr, tuple(pt2),5,color,-1)

    return imgl, imgr 



if __name__ == "__main__":

    Image_path = 'D:\\F_Estimation\\deepF_noCorrs\\data\\Pred\\Pred1_12_07'
    left_name = '\\image_00\data\\000000000'
    right_name = '\\image_01\data\\000000000'
    F_Path = 'D:\\F_Estimation\\deepF_noCorrs\\Pred_Result\\'+keywords+'\\'
    save_path = 'D:\\F_Estimation\\deepF_noCorrs\\Pred_Result\\'+keywords+'\\'

    for j in range(7):

        img_left = cv2.imread(Image_path+left_name+str(j)+'.png')
        img_right = cv2.imread(Image_path+right_name+str(j)+'.png')

        F = np.loadtxt(F_Path+str(j)+'.txt')
        F = F.reshape(3,3)
        F = F/F[2,2]
        print('F:',F)

        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img_left, None)
        kp2, des2 = sift.detectAndCompute(img_right, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        pts1 = []
        pts2 = []

        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F_r, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]

        # print(len(pts1))

        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
        
        lines1 = lines1.reshape(-1,3)
        # np.savetxt(save_path+'\line1_RANSAC.txt',lines1)
        # lines1 *= [1,-1,1]
        # print('Lines1: ',lines1)

        img5, img6 = drawlines(img_left,img_right,lines1,pts1,pts2)

        cv2.imwrite(save_path+'\Left_'+keywords+str(j)+'.jpg',img5)

        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 2, F)
        
        lines2 = lines2.reshape(-1,3)
        # np.savetxt(save_path+'\line2_RANSAC.txt',lines2)
        
        img3, img4 = drawlines(img_right,img_left,lines2,pts2,pts1)

        cv2.imwrite(save_path+'\Right_'+keywords+str(j)+'.jpg',img3)



