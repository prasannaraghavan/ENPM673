import cv2 
import glob
import numpy as np
import matplotlib.pyplot as plt

def histmat(image, bins):
    
    histogram = np.zeros(bins)

    for pixel in image:
        histogram[pixel] += 1
    return histogram


def arrsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

def Normalize(cs):
    
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    cs = nj / N
    return cs

def hist(image):

    image = np.array(image)
    
    flat = image.flatten()

    hist = histmat(flat, 256)
    
    cs = arrsum(hist)

    cs = Normalize(cs)

    cs = cs.astype('uint8')
    himg = np.zeros((image.shape))

    himg = alpha * cs[flat]

    himg = np.reshape(himg, image.shape)

    himg = himg.astype('uint8')

    return himg

def adhist(img):
    img = img.copy()
    h, w = img.shape
    bh, bw = h//8, w//8
   
    for i in range(8):
        for j in range(8):
            img[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = hist(img[i*bh:(i+1)*bh, j*bw:(j+1)*bw])
            img = np.array(img, dtype=np.uint8)
    return img

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, (1224, 370))
path = glob.glob(r"P:\Dox Files\Maryland\Courses\ENPM 673\Project\P2\Q1\dataset\*")
alpha = 1
for file in path:
    image = cv2.imread(file)
    width = image.shape[1]
    height = image.shape[0]

    r = image[:,:,2]
    g = image[:,:,1]
    b = image[:,:,0]

    histr = hist(r)
    histg = hist(g)
    histb = hist(b)

    histi = np.zeros((histr.shape[0],histr.shape[1],3), np.uint8)

    histi[:,:,2] = histr
    histi[:,:,1] = histg
    histi[:,:,0] = histb

    adhr = adhist(r)
    adhg = adhist(g)
    adhb = adhist(b)

    adhi = np.zeros((adhr.shape[0],adhr.shape[1],3), np.uint8)

    adhi[:,:,2] = adhr
    adhi[:,:,1] = adhg
    adhi[:,:,0] = adhb

    vertical = np.concatenate((histi, adhi), axis=0)
    cv2.imshow('', vertical )
    out.write(vertical)
    #cv2.imshow('Histogram',histi)
    #cv2.imshow('Adaptive Histogram',adhi)
    
    if cv2.waitKey(0) & 0xFF == ord('d'):
        break
out.release()  
cv2.destroyAllWindows()
