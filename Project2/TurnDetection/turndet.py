import cv2 as cv
import numpy as np


def warp(img):
    
    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(img, M, (500, 300), flags=cv.INTER_NEAREST) 

    return warped

def unwarp(img):
    
    
    img_size = (img.shape[1], img.shape[0])
    Minv = cv.getPerspectiveTransform(dst, src)
    unwarped = cv.warpPerspective(img, Minv, img_size, flags=cv.INTER_NEAREST)
    
    return unwarped

def Process(img):
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    h_channel_hls = hls[:,:,0]
    l_channel_hls = hls[:,:,1]
    s_channel_hls = hls[:,:,2]

    lab = cv.cvtColor(img, cv.COLOR_RGB2Lab)
    l_channel_lab = lab[:,:,0]
    a_channel_lab = lab[:,:,1]
    b_channel_lab = lab[:,:,2]
    img_edge = cv.Canny(s_channel_hls, 100, 200)

    ret,thresh = cv.threshold(img_edge,120,255,cv.THRESH_BINARY)

    return thresh

def average():
    count = 0
    fit0 = 0
    fit1 = 0
    fit2 = 0
    for fit in left_fit_avg:
        count+=1
        fit0 = fit0 + fit[0]
        fit1 = fit1 + fit[1]
        fit2 = fit2 + fit[2]
    
    denominator = count 
    fit0 = fit0/denominator
    fit1 = fit1/denominator
    fit2 = fit2/denominator

    left_fit = np.array([fit0,fit1,fit2])

    count = 0
    fit0 = 0
    fit1 = 0
    fit2 = 0
    for fit in right_fit_avg:
        count+=1
        fit0 = fit0 + fit[0]
        fit1 = fit1 + fit[1]
        fit2 = fit2 + fit[2]
    
    denominator = count
    fit0 = fit0/denominator
    fit1 = fit1/denominator
    fit2 = fit2/denominator

    right_fit = np.array([fit0,fit1,fit2])

    return left_fit,right_fit

def turn_predict(image_center,right_lane_pos, left_lane_pos):

    lane_center = left_lane_pos + (right_lane_pos - left_lane_pos)/2
    
    if (lane_center - image_center < -8):
        return ("Turning left")
    elif (lane_center - image_center < 5):
        return ("straight")
    else:
        return ("Turning right")

def SlidingWindow(warped_img):

    histogram = np.sum(warped_img, axis=0)

    out_img = np.dstack((warped_img,warped_img,warped_img))*255

    midpoint = int(histogram.shape[0]/2)
    leftlanepixel_initial = np.argmax(histogram[:midpoint])
    rightlanepixel_initial = np.argmax(histogram[midpoint:]) + midpoint

    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftlanepixel_current = leftlanepixel_initial
    rightlanepixel_current = rightlanepixel_initial

    image_center = int(warped_img.shape[1]/2)
    prediction = turn_predict(image_center,rightlanepixel_initial, leftlanepixel_initial)

    left_lane_idxs = []
    right_lane_idxs = []

    window_height = int(warped_img.shape[0]/num_windows)

    for window in range(num_windows):

        win_y_down = warped_img.shape[0] - (window+1)*window_height
        win_y_up = warped_img.shape[0] - window*window_height
        win_x_left_down = leftlanepixel_current - window_width
        win_x_right_down = rightlanepixel_current - window_width
        win_x_left_up = leftlanepixel_current + window_width
        win_x_right_up = rightlanepixel_current + window_width
        
        cv.rectangle(out_img,(win_x_left_down,win_y_down),(win_x_left_up,win_y_up),(0,255,0), 1)
        cv.rectangle(out_img,(win_x_right_down,win_y_down),(win_x_right_up,win_y_up),(0,255,0), 1)

        good_left_idxs = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_left_down) & (nonzerox < win_x_left_up)).nonzero()[0]
        good_right_idxs = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_right_down) & (nonzerox < win_x_right_up)).nonzero()[0]

        left_lane_idxs.append(good_left_idxs)
        right_lane_idxs.append(good_right_idxs)

        if len(good_left_idxs) > minpix:
            leftlanepixel_current = int(np.mean(nonzerox[good_left_idxs]))
        if len(good_right_idxs) > minpix:        
            rightlanepixel_current = int(np.mean(nonzerox[good_right_idxs]))

    left_lane_idxs = np.concatenate(left_lane_idxs)
    right_lane_idxs = np.concatenate(right_lane_idxs)
    left_pixels_x = nonzerox[left_lane_idxs]
    left_pixels_y = nonzeroy[left_lane_idxs]
    right_pixels_x = nonzerox[right_lane_idxs]
    right_pixels_y = nonzeroy[right_lane_idxs]

    if left_pixels_x.size == 0 or right_pixels_x.size == 0 or left_pixels_y.size == 0 or right_pixels_y.size == 0:


        ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
        pts = np.hstack((left_line_pts, right_line_pts))
        pts = np.array(pts, dtype=np.int32)
        color_blend = np.zeros_like(img).astype(np.uint8)
        cv.fillPoly(color_blend, pts, (0,255, 0))
        Unwarped_img = unwarp(color_blend)
        result = cv.addWeighted(img, 1, Unwarped_img, 0.5, 0)

        return result,out_img,left_fit,right_fit,prediction
    
    out_img[nonzeroy[right_lane_idxs], nonzerox[right_lane_idxs]] = [255, 0, 0]
    out_img[nonzeroy[left_lane_idxs], nonzerox[left_lane_idxs]] = [0, 0, 255]
    left_fit = np.polyfit(left_pixels_y, left_pixels_x, 2)
    right_fit = np.polyfit(right_pixels_y, right_pixels_x, 2)
    left_fit_avg.append(left_fit)
    right_fit_avg.append(right_fit)
    ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    pts = np.hstack((left_line_pts, right_line_pts))
    pts = np.array(pts, dtype=np.int32)
    color_blend = np.zeros_like(img).astype(np.uint8)
    cv.fillPoly(color_blend, pts, (0,255, 0))
    Unwarped_img = unwarp(color_blend)
    result = cv.addWeighted(img, 1, Unwarped_img, 0.5, 0)
    return result,out_img,left_fit,right_fit,prediction

def radius_curvature(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty*ymtr_per_pixel, left_fitx*xmtr_per_pixel, 2)
    right_fit_cr = np.polyfit(ploty*ymtr_per_pixel, right_fitx*xmtr_per_pixel, 2)
    left_rad = ((1 + (2*left_fit_cr[0]*y_eval*ymtr_per_pixel + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_rad = ((1 + (2*right_fit_cr[0]*y_eval*ymtr_per_pixel + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return (left_rad, right_rad)

def curvatures(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel):
    (left_curvature, right_curvature) = radius_curvature(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
    out_img = np.copy(img)
    avg_rad = round(np.mean([left_curvature, right_curvature]),0)
    cv.putText(out_img, 'Average lane curvature: {:.2f} m'.format(avg_rad), 
                (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
    cv.putText(out_img, 'left lane curvature: {:.2f} m'.format(left_curvature), 
                (50, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv.putText(out_img, 'right lane curvature: {:.2f} m'.format(right_curvature), 
                (50, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    return out_img

def dist_from_center(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    
    ymax = img.shape[0]*ymtr_per_pixel
    center = img.shape[1] / 2
    lineLeft = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
    lineRight = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]
    mid = lineLeft + (lineRight - lineLeft)/2
    dist = (mid - center) * xmtr_per_pixel
    if dist >= 0. :
        message = 'Vehicle location: {:.2f} m right'.format(dist)
    else:
        message = 'Vehicle location: {:.2f} m left'.format(abs(dist))

    return message




    



w, h = 1280, 720

dst = np.array([(500, 300), (0, 300), (0, 0), (500, 0)],np.float32)
src = np.array([(1100, 660), (200, 680), (600, 450), (730, 445)],np.float32)

num_windows = 10
window_width = 50
minpix = 25


xmtr_per_pixel=3/1280
ymtr_per_pixel=30/720

left_fit_avg = []
right_fit_avg = []

cap = cv.VideoCapture(r'P:\Dox Files\Maryland\Courses\ENPM 673\Project\P2\Q3\challenge.mp4')
while cap.isOpened():
    ret, img = cap.read()
    if not ret: break

    thresh = Process(img)

    warped = warp(thresh)
    
    result,out_img,left_fit,right_fit,prediction = SlidingWindow(warped)

    result = curvatures(result, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)

    message = dist_from_center(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)

    cv.putText(result, message, (50, 150),cv.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255),1, cv.LINE_AA)
    cv.putText(result, prediction, (50, 180),cv.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255),1, cv.LINE_AA)

    #cv.imshow('',warped_img)
    #cv.imshow('',out_img)
    cv.imshow('',result)
    
    if cv.waitKey(100) & 0xFF == ord('d'): 
        break
cap.release()
cv.destroyAllWindows()
