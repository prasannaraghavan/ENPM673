import cv2 as cv2
import numpy as np

K = np.array([[1346.100595,0,932.1633975],
              [0,1355.933136,654.8986796],
                    [0,0,1]             ])

def projectionMatrix(h,K):  
    h1 = h[:,0]   
    h2 = h[:,1]
    h3 = h[:,2]

    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K),h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K),h2)))
    b_t = lamda * np.matmul(np.linalg.inv(K),h)
    det = np.linalg.det(b_t)
    if det > 0:
        b = b_t
    else:              
        b = -1 * b_t  
        
    row1 = b[:, 0]
    row2 = b[:, 1] 
    row3 = np.cross(row1, row2)
    
    t = b[:, 2]
    Rt = np.column_stack((row1, row2, row3, t))

    P = np.matmul(K,Rt)  
    return P


def cubeproj(frame,H):
     cube_position = np.array([[0,0,0,1],[0,160,0,1],[160,160,0,1],
                     [160,0,0,1],[160,0,-80,1],[0,0,-80,1],[0,160,-80,1],
                     [160,160,-80,1]])
     
     P = projectionMatrix(H, K)
     new_cube = P @ cube_position.T
     new_cube = np.int0(new_cube[:2]/new_cube[2])
     x, y  = new_cube
     
     cv2.line(frame,(int(x[0]),int(y[0])),(int(x[5]),int(y[5])), (0,0,255), 2)
     cv2.line(frame,(int(x[1]),int(y[1])),(int(x[6]),int(y[6])), (0,0,255), 2)
     cv2.line(frame,(int(x[2]),int(y[2])),(int(x[7]),int(y[7])), (0,0,255), 2)
     cv2.line(frame,(int(x[3]),int(y[3])),(int(x[4]),int(y[4])), (0,0,255), 2)
    
     cv2.line(frame,(int(x[0]),int(y[0])),(int(x[1]),int(y[1])), (255,0,0), 2)
     cv2.line(frame,(int(x[1]),int(y[1])),(int(x[2]),int(y[2])), (255,0,0), 2)
     cv2.line(frame,(int(x[2]),int(y[2])),(int(x[3]),int(y[3])), (255,0,0), 2)
     cv2.line(frame,(int(x[3]),int(y[3])),(int(x[0]),int(y[0])), (255,0,0), 2)
     
     cv2.line(frame,(int(x[4]),int(y[4])),(int(x[5]),int(y[5])), (0,255,0), 2)
     cv2.line(frame,(int(x[4]),int(y[4])),(int(x[7]),int(y[7])), (0,255,0), 2)
     cv2.line(frame,(int(x[5]),int(y[5])),(int(x[6]),int(y[6])), (0,255,0), 2)
     cv2.line(frame,(int(x[6]),int(y[6])),(int(x[7]),int(y[7])), (0,255,0), 2)
     return frame