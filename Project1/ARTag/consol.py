import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from fftfunc import *
from homographyfunc import *
from warpfunc import *
from cubefunc import *

video = cv2.VideoCapture("newinput.avi")
testudo = cv2.imread("testudo.png")   
testudo = cv2.resize(testudo, (160,160))

width  = int(video.get(3))
height = int(video.get(4))
frameSize = (width, height)
out = cv2.VideoWriter('OutputVid.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)

while video.isOpened():
    isTrue, frame = video.read()
    if frame is not None:
        img_back,gray,threshed, magnitude_spectrum,fshift_mask_mag = fftimg(frame)
        # Good features
        corners = cv2.goodFeaturesToTrack(img_back,20,0.01,10)       
        corners= (np.int0(corners[:,0]))            
        x,y = corners.T
        i=y.argmax()
        j=x.argmin()
        k=y.argmin()
        l=x.argmax()
        
        c0, c1, c2, c3 = corners[(i,j,k,l),] 
        cn = (c0+c1+c2+c3)/4 #tag center
        dt = math.dist(cn, c0)
        tag_corners = []
        for pt in corners:
            
            if 0.25*dt < math.dist(pt, cn) < 0.58*dt :
                tag_corners.append(pt)
                #print(pt)
        
        
        for x, y in tag_corners:
    
            cv2.circle(frame, (x, y), 2, [0,0,255], -1)
        
        tag_corners = np.array(tag_corners)
        
        m,n = tag_corners.T
       
        p0 = m.argmax()
        p1 = n.argmin()
        p2 = m.argmin()
        p3 = n.argmax() 
        p0, p1, p2, p3 = tag_corners[(p0,p1,p2,p3),]
        Icorner_points= np.array([p0,p1,p2,p3])
    

        cv2.putText(frame, str(p0), p0, cv2.FONT_HERSHEY_SIMPLEX, 1,((209, 80, 0, 255)),1 )
        cv2.putText(frame, str(p1), p1, cv2.FONT_HERSHEY_SIMPLEX, 1,((209, 80, 0, 255)),1 )
        cv2.putText(frame, str(p2), p2, cv2.FONT_HERSHEY_SIMPLEX, 1,((209, 80, 0, 255)),1 )
        cv2.putText(frame, str(p3), p3, cv2.FONT_HERSHEY_SIMPLEX, 1,((209, 80, 0, 255)),1 )
        #cv2.imshow('corners', frame)
    
        tagv = frame.copy() 
        dstpnt = np.array([[0,0],[150,0],[150,200],[0,200]])
        taghomo = Homography(Icorner_points,dstpnt)
        warped = warpPerspective(tagv,taghomo,(150,200))
        #cv2.imshow("Warped Testudo",warped)
        
        
        w,h = testudo.shape[:2]
        p1 =np.float32([[0,0],[w,0],[w,h],[0,h]])
        p2=np.float32(Icorner_points)
        h1, w1 = gray.shape 
        H = Homography(p1,p2)  
        warped = warpPerspective(testudo,H,(w1,h1))
        p2=np.int32(p2)
        cv2.fillConvexPoly(frame,p2,(0,0,0))
        frame = frame + warped
        #cv2.imshow("testudowarp",frame)
        frame = cubeproj(frame,H)
    
        #cv2.imshow("cube proj",frame)
        out.write(frame)
        if cv2.waitKey(5) & 0xFF==ord("k"):
            break
    else:
        break
out.release()
video.release()
plt.show()
cv2.destroyAllWindows()