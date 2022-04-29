import cv2 as cv
import numpy as np

def order(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def Homography(p1, p2):
    A = []
    p1 = order(p1)
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        m, n = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -m * x, -m * y, -m])
        A.append([0, 0, 0, x, y, 1, -n * x, -n * y, -n])
    A = np.array(A)
    u, s, v= np.linalg.svd(A)
    H_matrix = v[-1, :]
    H_matrix = H_matrix.reshape(3,3)
    H_matrix_normalized = H_matrix/H_matrix[2,2]
    
    return H_matrix_normalized