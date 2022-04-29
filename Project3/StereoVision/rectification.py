import numpy as np
import cv2

def LeastSquares(x_1_ls, x_2_ls):
    
    lis = list()

    #forming the X matrix
    X = x_1_ls
    Y = np.reshape(x_2_ls, (x_2_ls.shape[0], 1))

    #computing B matrix 
    X_total = np.dot(X.T, X)
    X_total_inv = np.linalg.inv(X_total)
    Y_total = np.dot(X.T, Y)
    B_mat = np.dot(X_total_inv, Y_total)

    #computing the y coordinates and forming a list to return
    new_y = np.dot(X, B_mat)
    for i in new_y:
        for a in i:
            lis.append(a)

    return B_mat

def Rectification(F_mat,points1,points2):
    
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    # epipoles of left and right images
    U, sigma, VT = np.linalg.svd(F_mat)
    V = VT.T
    s = np.where(sigma < 0.00001)
    
    e_left = V[:,s[0][0]]
    e_right = U[:,s[0][0]]
    
    e_left = np.reshape(e_left,(e_left.shape[0],1))
    e_right = np.reshape(e_right,(e_right.shape[0],1))
    
    T1 = np.array([[1,0,-(640/2)],[0,1,-(480/2)],[0,0,1]])
    ef = np.dot(T1,e_right)
    ef = ef[:,:]/ef[-1,:]
    
    len = ((ef[0][0])**(2)+(ef[1][0])**(2))**(1/2)
    
    if ef[0][0] >= 0:
        
        alpha = 1
    else:
        
        alpha = -1
        
    T2 = np.array([[(alpha*ef[0][0])/len, (alpha*ef[1][0])/len, 0],
                    [-(alpha*ef[1][0])/len, (alpha*ef[0][0])/len, 0],[0, 0, 1]])
    ef = np.dot(T2,ef)
    
    T3 = np.array([[1, 0, 0],[0, 1, 0],[((-1)/ef[0][0]), 0, 1]])
    ef = np.dot(T3,ef)
    
    PHI2 = np.dot(np.dot(np.linalg.inv(T1),T3),np.dot(T2,T1))

    h_ones = np.array([1,1,1])
    h_ones = np.reshape(h_ones,(1,3))
    
    z = np.array([[0,-e_left[2][0],e_left[1][0]],[e_left[2][0],0,
                                -e_left[0][0]],[-e_left[1][0],e_left[0][0],0]])
    
    M = np.dot(z,F_mat) + np.dot(e_left,h_ones)

    Homography = np.dot(PHI2,M)
    
    ones = np.ones((points1.shape[0],1))
    points_1 = np.concatenate((points1,ones), axis = 1)
    points_2 = np.concatenate((points2,ones), axis = 1)
    
    x_1 = np.dot(Homography,points_1.T)
    x_1 = x_1[:,:]/x_1[2,:]
    x_1 = x_1.T
    
    x_2 = np.dot(PHI2,points_2.T)
    x_2 = x_2[:,:]/x_2[2,:]
    x_2 = x_2.T
    
    x_2_dash = np.reshape(x_2[:,0], (x_2.shape[0],1))
    
    L_S = LeastSquares(x_1,x_2_dash)
    
    d_1 = np.array([[L_S[0][0],L_S[1][0],L_S[2][0]],[0,1,0],[0,0,1]])
    
    PHI1 = np.dot(np.dot(d_1,PHI2),M)
    
    return PHI1,PHI2


'''
#  Built in comparison
def to_rectify(F_mat,points1,points2):
    _, PHI1, PHI2 = cv2.stereoRectifyUncalibrated(np.float32(points1), np.float32(points2), F_mat, imgSize=(680,420))
    return PHI1,PHI2
'''