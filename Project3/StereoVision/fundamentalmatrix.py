import numpy as np
import cv2

def FundamentalMatrix(f1,f2):
 
    f1_x = [] ; f1_y = [] ; f2_x = [] ; f2_y = []

    f1 = np.asarray(f1)
    f2 = np.asarray(f2)
    
    f1_xmean = np.mean(f1[:,0])    
    f1_ymean = np.mean(f1[:,1])    
    f2_xmean = np.mean(f2[:,0])        
    f2_ymean = np.mean(f2[:,1])
    
    for i in range(len(f1)): f1[i][0] = f1[i][0] - f1_xmean
    for i in range(len(f1)): f1[i][1] = f1[i][1] - f1_ymean
    for i in range(len(f2)): f2[i][0] = f2[i][0] - f2_xmean
    for i in range(len(f2)): f2[i][1] = f2[i][1] - f2_ymean
    
    f1_x = np.array(f1[:,0])
    f1_y = np.array(f1[:,1])
    f2_x = np.array(f2[:,0])
    f2_y = np.array(f2[:,1])

    sum_f1 = np.sum((f1)**2, axis = 1)
    sum_f2 = np.sum((f2)**2, axis = 1)
    #scaling factors 
    s1 = np.sqrt(2.)/np.mean(sum_f1**(1/2))
    s2 = np.sqrt(2.)/np.mean(sum_f2**(1/2))
                            
    sf1_1 = np.array([[s1,0,0],[0,s1,0],[0,0,1]])
    sf1_2 = np.array([[1,0,-f1_xmean],[0,1,-f1_ymean],[0,0,1]])
    
    sf2_1 = np.array([[s2,0,0],[0,s2,0],[0,0,1]])
    sf2_2 = np.array([[1,0,-f2_xmean],[0,1,-f2_ymean],[0,0,1]])
    
    t_1 = np.dot(sf1_1,sf1_2)
    t_2 = np.dot(sf2_1,sf2_2)
    
    x1 = ( (f1_x).reshape((-1,1)) ) * s1
    y1 = ( (f1_y).reshape((-1,1)) ) * s1
    x2 = ( (f2_x).reshape((-1,1)) ) * s2
    y2 = ( (f2_y).reshape((-1,1)) ) * s2
    
    Alist = []
    for i in range(x1.shape[0]):
        X1, Y1 = x1[i][0],y1[i][0]
        X2, Y2 = x2[i][0],y2[i][0]
        Alist.append([X2*X1 , X2*Y1 , X2 , Y2 * X1 , Y2 * Y1 ,  Y2 ,  X1 ,  Y1, 1])
    A = np.array(Alist)
    
    U, Sigma, VT = np.linalg.svd(A)
    
    v = VT.T
    
    fval = v[:,-1]
    ftemp = fval.reshape((3,3))
    
    Uf, Sigmatemp, Vf = np.linalg.svd(ftemp)
    #forcing the rank 2 constraint
    Sigmatemp[-1] = 0
    
    Sigmafinal = np.zeros(shape=(3,3)) 
    Sigmafinal[0][0] = Sigmatemp[0] 
    Sigmafinal[1][1] = Sigmatemp[1] 
    Sigmafinal[2][2] = Sigmatemp[2] 
    #un-normalizing 
    f_norm = np.dot(Uf , Sigmafinal)
    f_norm = np.dot(f_norm , Vf)
    
    f_un = np.dot(t_2.T , f_norm)
    f_un = np.dot(f_un , t_1)
    
    f_unnormalized = f_un/f_un[-1,-1]
    
    return f_unnormalized
# Here the parameters are tuned using Hartley's 8 points algorithm to obtain a better output 
#Change thresh for higher accuracy but will result in a longer processing time.
def Ransac(features1,features2):
    N = 2000
    sample = 0
    thresh = 0.02
    inliers_atm = 0
    P = 0.99
    RFmat = []

    while sample < N:
        
        rand1 = [] ; rand2 = []
        
        #getting a set of random 8 points
        index = np.random.randint( len(features1) , size = 8)
        
        for i in index:
            
            rand1.append(features1[i])
            rand2.append(features2[i])
        
        Fundamental = FundamentalMatrix(rand1, rand2)
    
        ones = np.ones((len(features1),1))
        x_1 = np.concatenate((features1,ones),axis=1)
        x_2 = np.concatenate((features2,ones),axis=1)
        
        line1 = np.dot(x_1, Fundamental.T)
        line2 = np.dot(x_2,Fundamental)
    
        e1 = np.sum(line2* x_1,axis=1,keepdims=True)**2
        e2 = np.sum(np.hstack((line1[:, :-1],line2[:,:-1]))**2,axis=1,keepdims=True)
        
        error =  e1 / e2 
        inliers = error <= thresh
         
        InlierCount = np.sum(inliers)
        
        #estimating best Fundamental M
        if inliers_atm <  InlierCount:
            
            inliers_atm = InlierCount
            good_ones = np.where(inliers == True)

            x1 = np.array(features1)
            x2 = np.array(features2)
            
            inliers1 = x1[good_ones[0][:]]
            inliers2 = x2[good_ones[0][:]]

            RFmat = Fundamental
            
        #iterating for N number of times
        InlierRatio = InlierCount/len(features1)
        den = np.log(1-(InlierRatio**8))
        num = np.log(1-P)
        if den == 0: continue
        N =  num / den
        sample += 1
        
    return RFmat, inliers1, inliers2
