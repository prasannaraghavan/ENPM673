import cv2 as cv2
import numpy as np




def warpPerspective(img, H, size):
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        result = np.zeros([size[1], size[0], 3], np.uint8)
    else:
        result = np.zeros([size[1], size[0]], np.uint8)
    x, y = np.indices((w, h))
    x,y = x.flatten(),y.flatten()
    
    img_coords = np.vstack((x,y,[1]*x.size))
    
    new_coords = H @ img_coords
    
    new_coords = new_coords/(new_coords[2]) #+ 1e-6)
    new_x, new_y, _ = np.int0(np.round(new_coords))
    
    new_x[np.where(new_x < 0)] = 0
    new_y[np.where(new_y < 0)] = 0
    new_x[np.where(new_x > size[0] - 1)] = size[0] - 1
    new_y[np.where(new_y > size[1] - 1)] = size[1] - 1
    
    result[new_y,new_x] = img[y,x]
    
    return result