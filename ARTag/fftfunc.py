import cv2 as cv
import numpy as np
# from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
import math

def fftimg(frame):
    gray = cv.cvtColor(frame , cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(7,7),cv.BORDER_DEFAULT)
    ret,threshed = cv.threshold(blur,150,200,cv.THRESH_BINARY)
    
    dft = cv.dft(np.float32(threshed), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    mag_spec = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    
    rows, cols = threshed.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 80
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0 
    
    fshift = dft_shift * mask
    maskmag = 2000 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    fishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(fishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back, gray,threshed, mag_spec, maskmag
