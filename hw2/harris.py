# issue: need to figure out the difference of np.gradient and cv2.Sobel !!!
# issue: different feature in imgA will map to the same point in imgB !!!
# issue: the descriptor using MSOP might have some bug
import cv2
import argparse
import numpy as np
import os
from tqdm import tqdm
import sys
from scipy.ndimage.filters import gaussian_filter
from operator import itemgetter
from scipy.spatial import cKDTree
import re
import glob
from random import shuffle

parser = argparse.ArgumentParser(description='Multi-Scale Oriented Patches')
parser.add_argument('-d', '--data_dir', type=str,default='test data/1/test/')
parser.add_argument('--sigma',default=5)
parser.add_argument('--response_w',default=0.04)
parser.add_argument('--feature_num',default=3000)
parser.add_argument('--radius',default=1)
parser.add_argument('--debug',default='F')
parser.add_argument('--random_order',default='F')
args = parser.parse_args()

def preImg():
    lst = []
    i = 0
    while True:
        if os.path.exists('%s/%d.png'%(args.data_dir, i)):
            image = cv2.imread('%s/%d.png'%(args.data_dir, i), -1)
            lst.append(image)
            i+=1
        else:
            break
    return np.array(lst)

def Harris(img):

    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    Ix__,Iy__ = np.gradient(img)
    
    Ix = cv2.Sobel(img,cv2.CV_32F,1,0)
    Iy = cv2.Sobel(img,cv2.CV_32F,0,1)

    # Question: why  the result of np.gradient and cv2.Sobel differ?
    print((Ix__>=0).all() == True)
    print((Ix>=0).all() == True)

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = np.multiply(Ix,Iy)

    w = img.shape[1]
    h = img.shape[0]

    featureList = []
    responseList = []

    w_Ixx = cv2.GaussianBlur(Ixx,(5,5),0)
    w_Iyy = cv2.GaussianBlur(Iyy,(5,5),0)
    w_Ixy = cv2.GaussianBlur(Ixy,(5,5),0)

    #     |w_Ixx w_Ixy|
    # M = |           |
    #     |w_Iyx w_Iyy|

    det = np.multiply(w_Ixx,w_Iyy) - w_Ixy**2
    tr = w_Ixx + w_Iyy
    response = det - args.response_w * (tr**2)

    # set the threshold to the 0.0001*max_r
    threshold = response.max()*0.0001

    margin_x, margin_y = int(w*0.03), int(h*0.03) # set the margin (prevent detect warpped img edge)

    for i in tqdm(range(margin_y,h-margin_y)): # mutually remove the bondary case (for simple implement of descriptor)
        for j in range(margin_x,w-margin_x): # mutually remove the bondary case (for simple implement of descriptor)
            
            if response[i,j] > threshold:
                featureList.append([j,i,response[i,j]])
    
    return featureList

def nonMaximalSuppression(fList,h,w,radius):
    f = sorted(fList, key=itemgetter(2),reverse=True)
    space = np.zeros(shape=(h,w))
    newList = []

    for x,y,response in tqdm(f):
        
        x_min = max(x-radius,0)
        x_max = min(x+radius,w)
        y_min = max(y-radius,0)
        y_max = min(y+radius,h)

        if np.sum(space[y_min:y_max,x_min:x_max]) == 0:
            space[y,x] = 1
            newList.append([x,y])
        if len(newList) >= args.feature_num:
            break
    return np.array(newList)


def simpleDes(img,single):
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    w = img.shape[1]
    h = img.shape[0]
    desList = []
    for x,y in single:
        # print(img[x-4:x+5,y-4:y+5].size)
        if img[y-4:y+5,x-4:x+5].size == 81: # I don't know why there are some crop it empty
            desList.append( [ x,y,img[y-4:y+5,x-4:x+5].reshape(-1)] )
    return desList


# the descriptor vector is like MSOP # NO use!!!!
def MSOPDes(img,single):
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    w = img.shape[1]
    h = img.shape[0]

    # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20warpAffine(InputArray%20src,%20OutputArray%20dst,%20InputArray%20M,%20Size%20dsize,%20int%20flags,%20int%20borderMode,%20const%20Scalar&%20borderValue)
    
    # cal orientation

    Ix = cv2.Sobel(img,cv2.CV_32F,1,0)
    Iy = cv2.Sobel(img,cv2.CV_32F,0,1)


    mag = (Ix**2 + Iy **2)**0.5
    cosA = Ix / (mag +1e-6)
    sinA = Iy / (mag +1e-6)
    theta = np.arctan2(sinA,cosA)

    desList = []
    for x,y in tqdm(single):

        M = cv2.getRotationMatrix2D((y,x),-theta[y,x],1)
        dst = cv2.warpAffine(img,M,(h,w))
        
        # crop the dst (handle the boundary)
        x_min =  x-20 if x-20>0 else 0
        x_max =  x+20 if x+20<w else w-1
        y_min =  y-20 if y-20>0 else 0
        y_max =  y+20 if y+20<h else h-1

        if dst[y_min:y_max,x_min:x_max].shape[0] > 0 and dst[y_min:y_max,x_min:x_max].shape[1] > 0:
            descriptor = cv2.resize(dst[y_min:y_max+1,x_min:x_max+1],(8,8),interpolation=cv2.INTER_CUBIC).reshape(-1)

            desList.append([x,y,theta[y,x],descriptor])
    return desList

def previewFeature(img,fList,idx):
    for x,y in fList:
        img[y,x] = np.array([0,0,255])
    cv2.imwrite( str(idx) + '_feature_myown.png',img)
    return

def featureMatching(desList1,desList2):
    point_number1 = len(desList1)
    data = []

    #    - feature_X          : int
    #    - feature_Y          : int
    #    - feature_descriptor : np.array of size (64,)

    for idx in range(len(desList1)):
        data.append(desList1[idx][-1])

    for idx in range(len(desList2)):    
        data.append(desList2[idx][-1])

    tree = cKDTree(data.copy())
    pair = []
    for idx in (range(len(desList1))):
        dd,ii = tree.query(data[idx],k=5)
        # find the first 2Group point
        for t,i in enumerate(ii):
            if i >= len(desList1):
                pair.append([idx,i-len(desList1),dd[t]])
                break
    pair = np.array(pair)
    return pair

def printPair(pairCorrList,a,b):
    x = preImg()
    feature_pairs = pairCorrList[0]
    img0 = x[a].copy()
    img1 = x[b].copy()
    for img0_x, img0_y, img1_x, img1_y in feature_pairs[:50]:
        cv2.circle(img0, (img0_x, img0_y), 2, (0,0,255), -1)
        cv2.circle(img1, (img1_x, img1_y), 2, (0,0,255), -1)
    matching = np.concatenate((img1, img0), axis=1)
    cv2.imwrite('matching_%d_%d.jpg'%(b,a), matching)


def findCorrSeq(desList,mapping_threshold=100):
    

    from pairwise_alignment import ransac
    imgPairSize = len(desList)
    orderPair = []
    for i in tqdm(range(imgPairSize)):
        lst = []
        
        for j in range(imgPairSize):
            if i == j:
                continue
            else:
                pairIdx = featureMatching(desList[i],desList[j])
                pairCorr = pairIdx2CoorSingle(pairIdx,desList,i,j)
                # printPair(pairCorrList,i,j)
                if len(pairCorr) < 50: # the valid matching pait less than 50, continue!
                    continue
                [dx, dy], c = ransac(pairCorr)
                if c >= mapping_threshold:
                    lst.append([j,dx,c])

        # justify the left/right image
        # (0,1) => +
        
        # print(lst) 
        if len(lst) == 1:
            idx1 = lst[0][0]
            dx_idx1 = lst[0][1]
            if dx_idx1 > 0:
                orderPair.append([i,idx1])
            else:
                orderPair.append([idx1,i])
        else:
            lst = sorted(lst, key=itemgetter(2),reverse=True)
            idx1, idx2 = lst[0][0],lst[1][0]
            dx_idx1, dx_idx2 = lst[0][1],lst[1][1]

            if dx_idx1 < 0 and dx_idx2 > 0:
                orderPair.append([idx1,i,idx2])
            else:
                orderPair.append([idx2,i,idx1])


        print(orderPair)
    print(orderPair)
    rightSeq = getRightSeq(orderPair)
    print(rightSeq)
    return rightSeq

def getRightSeq(orderPair):

    num_node = len(orderPair)
    print(num_node)
    graph = np.zeros(shape=(num_node,num_node))
    for p in orderPair:
        if  graph[p[1]][p[0]] == 1:
            exit('seq conflict')
        if len(p) == 2:
            graph[p[0]][p[1]] = 1
        else:
            if graph[p[2]][p[1]] == 1:
                exit('seq conflict')
            graph[p[0]][p[1]] = 1
            graph[p[1]][p[2]] = 1


    sum_row = np.sum(graph, axis=1)
    end, = np.where(sum_row == 0)
    backtrack = [int(end)]

    while len(backtrack) < num_node:
        before, = np.where(graph[:,backtrack[0]] == 1)
        backtrack.insert(0,int(before))
    return backtrack

def openCVHarris(img,idx):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)

    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imwrite( str(idx) + '_feature_opencv.png',img)
    return

def pairIdx2CoorSingle(pairIdx,desList,a,b):
    p = sorted(pairIdx, key=itemgetter(2)) # sorted by distance
    l = []
    for idx1,idx2,d in p:
        idx1 = int(idx1)
        idx2 = int(idx2)
        imgA_x = desList[a][idx1][0]
        imgA_y = desList[a][idx1][1]
        imgB_x = desList[b][idx2][0]
        imgB_y = desList[b][idx2][1]
        l.append([imgA_x,imgA_y,imgB_x,imgB_y])
    return np.array(l)

def pairIdx2Coor(pairIdxList,desList,seq):
    pairCorrList = []
    for idx,p in enumerate(pairIdxList):
        l = []
        # p = sorted(p, key=itemgetter(2)) # sorted by distance
        # for idx1,idx2,d in p:
        #     idx1 = int(idx1)
        #     idx2 = int(idx2)
        #     imgA_x = desList[idx][idx1][0]
        #     imgA_y = desList[idx][idx1][1]
        #     imgB_x = desList[idx+1][idx2][0]
        #     imgB_y = desList[idx+1][idx2][1]
        #     l.append([imgA_x,imgA_y,imgB_x,imgB_y])
        l = pairIdx2CoorSingle(p,desList,seq[idx],seq[idx+1])
        pairCorrList.append(np.array(l))

    return np.array(pairCorrList)

def produceFeature(imginput,existImg=True,featureDesMethod='simple'):
    
    # read the image
    if existImg == False:
        x = preImg()
        # np.random.shuffle(x)
    else:
        assert(imginput != None)
        x = imginput

    if args.debug == 'T':
        for i in range(len(x)):
            openCVHarris(x[i].copy(),i)

    featureList = []
    # do Harris corner detect and return TopX response
    for idx,img in enumerate(x):
        print('Doing Harris corner detector on Image %d' %(idx))
        fList = Harris(img)
        fList = nonMaximalSuppression(fList,img.shape[0],img.shape[1],args.radius)
        print('Get Top %d features on Image %d' %(fList.shape[0],idx))
        featureList.append(fList)
        print('*****************************************************')

    featureList = np.array(featureList)
    # featureList format
    # type 3D numpy array with dimension (image_num, <=args.feature_num, 2)

    if args.debug == 'T':
        for i in range(len(x)):
            previewFeature(x[i],featureList[i],i)

    desList = []
    if featureDesMethod == 'simple':
        for idx,img in enumerate(x):
            desList.append(simpleDes(img,featureList[idx]))
    elif featureDesMethod == 'MSOP':
        for idx,img in enumerate(x):    
            desList.append(MSOPDes(img,featureList[idx]))

    # desList_MSOP:  shape = (image_num,feature_num, )
    #                              ~~~~~~~~~~~~
    #                              the feature_num may smaller than args.features_num
    #
    #  each element in desList have:
    #
    #    - feature_X          : int
    #    - feature_Y          : int
    #    - feature_Theta      : float
    #    - feature_descriptor : np.array of size (64,)
        
    # desList_Simple:  shape = (image_num,feature_num, )
    #                              ~~~~~~~~~~~~
    #                              the feature_num may smaller than args.features_num
    #
    #  each element in desList have:
    #
    #    - feature_X          : int
    #    - feature_Y          : int
    #    - feature_descriptor : np.array of size (81,)

    pairIdxList = []
    if args.random_order == 'F':
        # seq = findCorrSeq(desList)
        seq = list(range(len(x)))
        for idx in tqdm(range(len(x)-1)):
            pairIdxList.append(featureMatching(desList[idx],desList[idx+1]))
        pairIdxList = np.array(pairIdxList)

    elif args.random_order == 'T':
        seq = findCorrSeq(desList)
        for idx in tqdm(range(len(seq)-1)):
            pairIdxList.append(featureMatching(desList[seq[idx]],desList[seq[idx+1]]))
        pairIdxList = np.array(pairIdxList)

    pairCorrList = pairIdx2Coor(pairIdxList,desList,seq)

    #  pairCorrList shape = ( pairNum(N images has N-1), pairpoint , 4 )

    #  pairCorrList[0] get the paired points in img0 and img1
    #  pairCorrList[0][1] get the first paired points in img0 and img1
    #  and the format is [img0_x, img0_y, img1_x, img1_y]

    return pairCorrList,seq


if __name__ == '__main__':
    x = preImg()
    pairCorrList,seq  = produceFeature(None,False,'simple')
    # seq is left to right
    for i in range(pairCorrList.shape[0]):
        feature_pairs = pairCorrList[i]
        img0 = x[seq[i]].copy()
        img1 = x[seq[i+1]].copy()
        for img0_x, img0_y, img1_x, img1_y in feature_pairs[:50]:
            cv2.circle(img0, (img0_x, img0_y), 2, (0,0,255), -1)
            cv2.circle(img1, (img1_x, img1_y), 2, (0,0,255), -1)
        matching = np.concatenate((img0, img1), axis=1)
        cv2.imwrite('matching_%d.jpg'%(i), matching)
        # print(feature_pairs.shape)

