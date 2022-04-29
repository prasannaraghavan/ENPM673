import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
from fundamentalmatrix import *
from rectification import *
#from pose import EssentialMatrix


dataset = int(input("Enter the dataset number: "))
while dataset<1 or dataset>3:
    dataset = int(input("Enter a valid dataset number between 1 and 3: "))

if dataset == 1:
    K1=np.array([[1742.11, 0, 804.90],[0, 1742.11, 541.22],[0, 0, 1]])
    K2=K1
    baseline=221.76
    width=1920
    height=1080
    ndisp=100
    vmin=29
    vmax=61
    f = K1[0,0]
    img_1 = cv2.imread('./Dataset 1/im0.png')
    img_2 = cv2.imread('./Dataset 1/im1.png')
elif dataset == 2:
    K1=np.array([[1758.23, 0, 977.42],[ 0, 1758.23, 552.15],[ 0, 0, 1]])
    K2=K1
    baseline=88.39
    width=1920
    height=1080
    ndisp=220
    vmin=55
    vmax=195
    f = K1[0,0]
    img_1 = cv2.imread('./Dataset 2/im0.png')
    img_2 = cv2.imread('./Dataset 2/im1.png')
elif dataset == 3:
    K1= np.array([[1729.05, 0, -364.24],[ 0, 1729.05, 552.22],[ 0, 0, 1]])
    K2= K2
    baseline=537.75
    width=1920
    height=1080
    ndisp=180
    vmin=25
    vmax=150
    f = K1[0,0]
    img_1 = cv2.imread('./Dataset 3/im0.png')
    img_2 = cv2.imread('./Dataset 3/im1.png')
    

def EssentialMatrix(Fmatrix):
    
    e_mat = np.dot(K2.T,Fmatrix)
    e_mat = np.dot(e_mat,K1)
    #solving for E using SVD
    Ue, sigma_e, Ve = np.linalg.svd(e_mat)
    sigma_final = np.zeros((3,3))
    
    for i in range(3):
        sigma_final[i,i] = 1
    sigma_final[-1,-1] = 0

    E_mat = np.dot(Ue,sigma_final)
    E_mat = np.dot(E_mat,Ve)
    
    return E_mat


def PoseInfo(e_mat):

    Ue, sigma, Ve = np.linalg.svd(e_mat)
    d = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    Rot_1 = np.dot(Ue, d)
    Rot_1 = np.dot(Rot_1, Ve)
    c1 = Ue[:, 2]
    if (np.linalg.det(Rot_1) < 0):
        Rot_1 = -Rot_1
        c1 = -c1
        
    Rot_2 = Rot_1
    c2 = -Ue[:, 2]
    if (np.linalg.det(Rot_2) < 0):
        Rot_2 = -Rot_2
        c2 = -c2
        
    Rot_3 = np.dot(Ue, d.T)
    Rot_3 = np.dot(Rot_1, Ve)
    c3 = Ue[:, 2]
    if (np.linalg.det(Rot_3) < 0):
        Rot_3 = -Rot_3
        c3 = -c3
        
    Rot_4 = Rot_3
    c4 = -Ue[:, 2]
    if (np.linalg.det(Rot_4) < 0):
        Rot_4 = -Rot_4
        c4 = -c4
    
    c1 = c1.reshape((3,1))
    c2 = c2.reshape((3,1))
    c3 = c3.reshape((3,1))
    c4 = c4.reshape((3,1))
    
    rot_final = [Rot_1,Rot_2,Rot_3,Rot_4]
    c_final = [c1,c2,c3,c4]
    
    return rot_final,c_final

def threeDpoints(r2,c2,p1,p2):
    
    c1 = np.array([[0],[0],[0]])
    
    r1 = np.identity(3)
    
    r1_c1 = -np.dot(r1,c1)
    r2_c2 = -np.dot(r2,c2)

    j1 = np.concatenate((r1, r1_c1), axis = 1)
    j2 = np.concatenate((r2, r2_c2), axis = 1)

    P1 = np.dot(K1,j1)
    P2 = np.dot(K2,j2)

    l_sol = []
    
    for i in range(len(p1)):
        
        x_1 = np.array(p1[i])
        x_2 = np.array(p2[i])
        
        x_1 = np.reshape(x_1,(2,1))
        q = np.array([1])
        q = np.reshape(q,(1,1))
        
        x_1 = np.concatenate((x_1,q), axis = 0)
        x_2 = np.reshape(x_2,(2,1))
        x_2 = np.concatenate((x_2,q), axis = 0)
  
        x_1_skew = np.array([[0,-x_1[2][0],x_1[1][0]],[x_1[2][0], 0, -x_1[0][0]],[-x_1[1][0], x_1[0][0], 0]])
        x_2_skew = np.array([[0,-x_2[2][0],x_2[1][0]],[x_2[2][0], 0, -x_2[0][0]],[-x_2[1][0], x_2[0][0], 0]])
        
        A1 = np.dot(x_1_skew, P1)
        A2 = np.dot(x_2_skew, P2)
        #calculating A and solving using SVD
        A = np.zeros((6,4))
        for i in range(6):
            if i<=2:
                A[i,:] = A1[i,:]
            else:
                A[i,:] = A2[i-3,:]
                
        U, sigma, VT = np.linalg.svd(A)
        VT = VT[3]
        VT = VT/VT[-1]
        l_sol.append(VT)
        
    l_sol = np.array(l_sol) 
    
    return l_sol    

def Triangulation(R_lis,C_lis,p1,p2):
    
    clis = list()
    for i in range(4):
        x = threeDpoints(R_lis[i],C_lis[i],p1,p2)
        n = 0
        for j in range(x.shape[0]):
            cord = x[j,:].reshape(-1,1)
            if np.dot(R_lis[i][2], (cord[0:3] - C_lis[i])) > 0 and cord[2]>0:
                n += 1
        clis.append(n)
        ind = clis.index(max(clis))
        if C_lis[ind][2]>0:
            C_lis[ind] = -C_lis[ind]
    return R_lis[ind], C_lis[ind]


def SumAbsoluteDifference(pixel_vals_1, pixel_vals_2):

    if pixel_vals_1.shape != pixel_vals_2.shape:
        return -1
    return np.sum(abs(pixel_vals_1 - pixel_vals_2))


def BlockCompare(y, x, left_local, right, window_size=5):
    
    #get search range for the right image
    x_min = max(0, x - local_window)
    x_max = min(right.shape[1], x + local_window)
    min_sad = None
    index_min = None
    first = True
    
    for x in range(x_min, x_max):
        right_local = right[y: y+window_size,x: x+window_size]
        sad = SumAbsoluteDifference(left_local, right_local)
        if first:
            min_sad = sad
            index_min = (y, x)
            first = False
        else:
            if sad < min_sad:
                min_sad = sad
                index_min = (y, x)

    return index_min

def Disparity(image_left,image_right):
    
    left = np.asarray(image_left)
    right = np.asarray(image_right)
    
    left = left.astype(int)
    right = right.astype(int)
    
    if left.shape != right.shape:
        print("Image Shapes do not match!!")
      
    h, w , g = left.shape
    
    disparity = np.zeros((h, w))
    #going over each pixel
    for y in range(window, h-window):
        for x in range(window, w-window):
            left_local = left[y:y + window, x:x + window]
            index_min = BlockCompare(y, x, left_local, right, window_size = window)
            disparity[y, x] = abs(index_min[1] - x)
    #print(disparity)
    
    #plt.imshow(disparity, cmap='hot', interpolation='bilinear')
    plt.imshow(disparity, cmap='hot', interpolation='bicubic')
    plt.title('Disparity Plot Heat')
    plt.savefig('disparity_image_heat.png')
    plt.show()
    
    #plt.imshow(disparity, cmap='gray', interpolation='bilinear')
    plt.imshow(disparity, cmap='gray', interpolation='bicubic')
    plt.title('Disparity Plot Gray')
    plt.savefig('disparity_image_gray.png')
    plt.show()
    
    return disparity

#function to draw the epipolar lines on the given images
def EpiLines(drawline_im1,drawline_im2,lines,pts1,pts2):
   
    sh = drawline_im1.shape
    r = sh[0]
    c = sh[1]
    
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        pt1 = [int(pt1[0]),int(pt1[1])]
        pt2 = [int(pt2[0]),int(pt2[1])]
        
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        drawline_im1 = cv2.line(drawline_im1, (x0,y0), (x1,y1), color,1)
        drawline_im1 = cv2.circle(drawline_im1,tuple(pt1),2,color,-1)
        drawline_im2 = cv2.circle(drawline_im2,tuple(pt2),2,color,-1)
        
    return drawline_im1,drawline_im2

#extracting the matching features using SIFT
def feature_matching(): 
    
    img_1_sift = img_1.copy()
    img_2_sift = img_2.copy()
    
    img_1_sift = cv2.resize(img_1_sift,(700,500),fx=0,fy=0,interpolation=cv2.INTER_AREA)
    img_2_sift = cv2.resize(img_2_sift,(700,500),fx=0,fy=0,interpolation=cv2.INTER_AREA)
    #img_1_sift = cv2.cvtColor(img_1_sift, cv2.COLOR_BGR2RGB)
    #img_2_sift = cv2.cvtColor(img_1_sift, cv2.COLOR_BGR2RGB)
    sift = cv2.SIFT_create()
    
    keypoints_1, descriptors_1 = sift.detectAndCompute(img_1_sift,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img_2_sift,None)
    
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    features_1 = []
    features_2 = []
    for i in matches:
        features_1.append(keypoints_1[i.queryIdx].pt)
        features_2.append(keypoints_2[i.trainIdx].pt)

    return features_1,features_2,img_1_sift,img_2_sift

window = 5
local_window = 56

features_image_1 , features_image_2 , img1 , img2 = feature_matching()

img1c = img1.copy()
img2c = img2.copy()

BFmat, r_points_1, r_points_2 = Ransac(features_image_1,features_image_2)
print("F matrix \n",BFmat)
E_mat = EssentialMatrix(BFmat)
print("E Matrix \n",E_mat)
Rotation, Translation = PoseInfo(E_mat)
R, T = Triangulation(Rotation, Translation, features_image_1, features_image_2)
print("Rotation \n",R)
print("Translation \n",T)
epilines_1 = cv2.computeCorrespondEpilines(r_points_2.reshape(-1,1,2), 2,BFmat)
epilines_1 = epilines_1.reshape(-1,3)

epilines_2 = cv2.computeCorrespondEpilines(r_points_1.reshape(-1,1,2), 1,BFmat)
epilines_2 = epilines_2.reshape(-1,3)

img1, img2 = EpiLines(img1,img2,epilines_1,r_points_1[:100],r_points_2[:100])
img1, img2 = EpiLines(img2,img1,epilines_2,r_points_1[:100],r_points_2[:100])

one_s = np.ones((r_points_1.shape[0],1))
r_points_1 = np.concatenate((r_points_1,one_s),axis = 1)
r_points_2 = np.concatenate((r_points_2,one_s),axis = 1)

H0 , H1 = Rectification(BFmat,features_image_1, features_image_2)

print('Homography Mat 1 : ',H0)
print('Homography Mat 2 : ',H1)

left_rectified = cv2.warpPerspective(img1, H0, (700,500))
right_rectified = cv2.warpPerspective(img2, H1, (700,500))

left_rec_nolines = cv2.warpPerspective(img1c, H0, (700,500))
right_rec_nolines = cv2.warpPerspective(img2c, H1, (700,500))
#cv2.imshow("Epilines Drawn on Rectified Left Image ",left_rectified)
#cv2.imshow("Epilines Drawn on Rectified Right Image ",right_rectified)
plt.imshow(left_rectified)
plt.show()
plt.imshow(right_rectified)
plt.show()

d = Disparity(left_rec_nolines,right_rec_nolines)
    
#disp[disp >= 3] = 3
cond1 = np.logical_and(d >= 0,d < 10)
cond2 = d > 40
    
d[cond1] = 10
d[cond2] = 40
    
depth = baseline * f / d
    
plt.imshow(depth, cmap='gray', interpolation='bilinear')
plt.title('Depth Plot Gray')
plt.savefig('depth_gray.png')
plt.show()
    
plt.imshow(depth, cmap='hot', interpolation='bilinear')
plt.title('Depth Plot Heat')
plt.savefig('depth_heat.png')
plt.show()
    
