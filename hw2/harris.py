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

parser = argparse.ArgumentParser(description='Multi-Scale Oriented Patches')
parser.add_argument('-d', '--data_dir', type=str,default='data/')
parser.add_argument('--sigma',default=5)
parser.add_argument('--response_w',default=0.04)
parser.add_argument('--feature_num',default=50)
parser.add_argument('--radius',default=3)
parser.add_argument('--debug',default='F')

args = parser.parse_args()

def preImg():
    lst = []
    for name in os.listdir(args.data_dir):
        image = cv2.imread(args.data_dir + name)    
        lst.append(image)
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

    w = img.shape[0]
    h = img.shape[1]

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


    for i in tqdm(range(1,w-1)): # mutually remove the bondary case (for simple implement of descriptor)
        for j in range(1,h-1): # mutually remove the bondary case (for simple implement of descriptor)
            
            if response[i,j] > threshold:
                featureList.append([i,j,response[i,j]])
    
    return featureList

def nonMaximalSuppression(fList,w,h,radius):
    f = sorted(fList, key=itemgetter(2),reverse=True)
    space = np.zeros(shape=(w,h))
    newList = []

    for x,y,response in tqdm(f):
        
        x_min = max(x-radius,0)
        x_max = min(x+radius,w)
        y_min = max(y-radius,0)
        y_max = min(y+radius,h)

        if np.sum(space[x_min:x_max,y_min:y_max]) == 0:
            space[x,y] = 1
            newList.append([x,y])

        if len(newList) >= args.feature_num:
            break
    return np.array(newList)


def simpleDes(img,single):
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    w = img.shape[0]
    h = img.shape[1]

    desList = []
    for x,y in single:
        desList.append( [ x,y,img[x-1:x+2,y-1:y+2].reshape(-1)] )
    return desList


# the descriptor vector is like MSOP # NO use!!!!
def MSOPDes(img,single):
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    w = img.shape[0]
    h = img.shape[1]

    # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20warpAffine(InputArray%20src,%20OutputArray%20dst,%20InputArray%20M,%20Size%20dsize,%20int%20flags,%20int%20borderMode,%20const%20Scalar&%20borderValue)
    
    # cal orientation

    # np.gradient or cv2.Sobel to use ? !!!!!!!!!!!!!!!!!!
    # Ix,Iy = np.gradient(img)
    # Ix = gaussian_filter(Ix, 4.5)
    # Iy = gaussian_filter(Iy, 4.5)

    Ix = cv2.Sobel(img,cv2.CV_32F,1,0)
    Iy = cv2.Sobel(img,cv2.CV_32F,0,1)


    mag = (Ix**2 + Iy **2)**0.5
    cosA = Ix / (mag +1e-6)
    sinA = Iy / (mag +1e-6)
    theta = np.arctan2(sinA,cosA)

    desList = []
    for x,y in single:
        M = cv2.getRotationMatrix2D((x,y),-theta[x,y],1)
        dst = cv2.warpAffine(img,M,(w,h))
        
        # crop the dst (handle the boundary)
        x_min =  x-20 if x-20>0 else 0
        x_max =  x+20 if x+20<w else w-1
        y_min =  y-20 if y-20>0 else 0
        y_max =  y+20 if y+20<h else h-1

        if dst[x_min:x_max,y_min:y_max].shape[0] > 0 and dst[x_min:x_max,y_min:y_max].shape[1] > 0:
            descriptor = cv2.resize(dst[x_min:x_max+1,y_min:y_max+1],(8,8),interpolation=cv2.INTER_CUBIC).reshape(-1)

            desList.append([x,y,theta[x,y],descriptor])
    return desList

def previewFeature(img,fList,idx):
    for x,y in fList:
        img[x,y] = np.array([0,0,255])
    cv2.imwrite( str(idx) + '_feature_myown.png',img)
    return

def featureMatching(desList1,desList2):
    assert(len(desList1) == len(desList2))
    point_number = len(desList1)
    data = []

    #    - feature_X          : int
    #    - feature_Y          : int
    #    - feature_descriptor : np.array of size (64,)
    
    for idx in range(point_number):
        data.append(desList1[idx][-1])
    for idx in range(point_number):    
        data.append(desList2[idx][-1])

    tree = cKDTree(data.copy())
    pair = []
    for idx in range(point_number):
        dd,ii = tree.query(data[idx],k=10)
        # find the first 2Group point
        for i in ii:
            if i >= point_number:
                pair.append([idx,i-point_number])
                break
    pair = np.array(pair)
    return pair

def openCVHarris(img,idx):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)

    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imwrite( str(idx) + '_feature_opencv.png',img)
    return

def pairIdx2Coor(pairIdxList,desList):
    pairCorrList = []
    for idx,p in enumerate(pairIdxList):
        l = []
        for idx1,idx2 in p:
            imgA_x = desList[idx][idx1][0]
            imgA_y = desList[idx][idx1][1]
            imgB_x = desList[idx+1][idx2][0]
            imgB_y = desList[idx+1][idx2][1]
            l.append([imgA_x,imgA_y,imgB_x,imgB_y])
        pairCorrList.append(l)
    return np.array(pairCorrList)

def produceFeature(imginput,existImg=True,featureDesMethod='simple'):
    
    # read the image
    if existImg == False:
        x = preImg()
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
    
    featureList = np.array(featureList,dtype=np.int32)
    # featureList format
    # type 3D numpy array with dimension (image_num,args.feature_num, 2)

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
    #    - feature_descriptor : np.array of size (64,)

    pairIdxList = []
    for idx in range(len(x)-1):
        pairIdxList.append(featureMatching(desList[idx],desList[idx+1]))
    pairIdxList = np.array(pairIdxList)

    pairCorrList = pairIdx2Coor(pairIdxList,desList)

    #  pairCorrList shape = ( pairNum(N images has N-1), pairpoint , 4 )

    #  pairCorrList[0] get the paired points in img0 and img1
    #  pairCorrList[0][1] get the first paired points in img0 and img1
    #  and the format is [img0_x, img0_y, img1_x, img1_y]

    return pairCorrList


if __name__ == '__main__':
    pairCorrList = produceFeature(None,False,'simple')
    print(pairCorrList)
