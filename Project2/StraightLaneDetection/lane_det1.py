import numpy as np
import cv2

polygon = np.array([[(350,355),(590,355),(950,540),(40,540)]])
p1, p2, p3, p4 = tuple(polygon[0,0,:]), tuple(polygon[0,1,:]), tuple(polygon[0,2,:]), tuple(polygon[0,3,:])
window = np.array([[(0,0),(480,0),(0,600),(480,600)]])

def ROI(image):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def ChangePerspective(image):
    matrix = cv2.getPerspectiveTransform(np.float32([[p1,p2,p4,p3]]),np.float32(window))
    result = cv2.warpPerspective(canny,matrix,(480,600),flags = cv2.INTER_LANCZOS4)
    return result

def LaneDivider(image):
    histogram = np.sum(image, axis=0)

    midpoint = int(histogram.shape[0] / 2)
    leftlanepixel_initial = np.sum(histogram[:midpoint])
    rightlanepixel_initial = np.sum(histogram[midpoint:])

    if leftlanepixel_initial > rightlanepixel_initial:
        leftcolor = (0, 255, 0)
        rightcolor = (0,0,255)
    else:
        leftcolor = (0, 0, 255)
        rightcolor = (0, 255, 0)
    return leftcolor, rightcolor

def LaneLines(frame,canny):
    leftline = cv2.HoughLinesP(canny[:,0:mpt], 2, np.pi / 180, 100, np.array([]), minLineLength=70, maxLineGap=150)
    rightline = cv2.HoughLinesP(canny[:,mpt:], 2, np.pi / 180, 100, np.array([]), minLineLength=7, maxLineGap=50)

    line_image = np.zeros_like(frame)
    if leftline is not None:
        for line in leftline:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame,(x1,y1),(x2,y2),leftcolor,10)
    if rightline is not None:
        for line in rightline:
            x1,y1,x2,y2 = line[0]
            cv2.line(frame,(x1+mpt,y1),(x2+mpt,y2),rightcolor,10)
    weightedimage = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return weightedimage

vidcap = cv2.VideoCapture(r'P:\Dox Files\Maryland\Courses\ENPM 673\Project\P2\Q2\whiteline.mp4')

while True:
    IsTrue, frame = vidcap.read()
    #frame = cv2.flip(frame, 1)
    mpt = int(frame.shape[1]/2)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #L_roi = ROI(gray)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    polyframe = ROI(thresh)
    canny = cv2.Canny(polyframe,10,150)
    cv2.imshow('Canny Lines', canny)
    warped = ChangePerspective(polyframe)
    #cv2.imshow('Canny Lines', warped)
    leftcolor, rightcolor = LaneDivider(warped)
    wimage = LaneLines(frame,canny)
    cv2.imshow('Lane Line Image',wimage)
    if cv2.waitKey(25) & 0xFF == ord('d'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
