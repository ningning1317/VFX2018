import cv2
import argparse
import numpy as np
import os
from tqdm import tqdm
import sys
from scipy.ndimage.filters import gaussian_filter

parser = argparse.ArgumentParser(description='Multi-Scale Oriented Patches')
parser.add_argument('-d', '--data_dir', type=str,default='data/')
parser.add_argument('--sigma',default=5)
parser.add_argument('--response_w',default=0.06)
parser.add_argument('--response_threshold',default=20)
parser.add_argument('--feature_num',default=500)
args = parser.parse_args()

def preImg():
    lst = []
    for name in os.listdir(args.data_dir):
        image = cv2.imread(args.data_dir + name)    
        lst.append(image)
    return np.array(lst)

def Harris(img):

    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    Ix,Iy = np.gradient(img) 

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

    for i in tqdm(range(w)):
        for j in range(h):
            #     |w_Ixx w_Ixy|
            # M = |           |
            #     |w_Iyx w_Iyy|

            det = w_Ixx[i,j] * w_Iyy[i,j] - w_Ixy[i,j] * w_Ixy[i,j]
            tr = w_Ixx[i,j] + w_Iyy[i,j]
            response = det - args.response_w * (tr**2)

            if response > args.response_threshold:
                featureList.append([i,j,response])

    return featureList

def search4TopXResponse(featureList,topX):
    from operator import itemgetter
    return sorted(featureList, key=itemgetter(2))[0:topX]



# the descriptor vector is like MSOP
def DesGen(img,single):
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    w = img.shape[0]
    h = img.shape[1]

    # https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20warpAffine(InputArray%20src,%20OutputArray%20dst,%20InputArray%20M,%20Size%20dsize,%20int%20flags,%20int%20borderMode,%20const%20Scalar&%20borderValue)
    
    # cal orientation
    Ix,Iy = np.gradient(img)
    Ix = gaussian_filter(Ix, 4.5)
    Iy = gaussian_filter(Iy, 4.5)
    mag = (Ix**2 + Iy **2)**0.5
    cosA = Ix / mag
    sinA = Iy /mag
    theta = np.arctan2(sinA,cosA)

    desList = []
    for x,y in single:
        M = cv2.getRotationMatrix2D((x,y),-theta[x,y],1)
        dst = cv2.warpAffine(img,M,(w,h))
        
        # crop the dst (handle the boundary)
        x_min =  x-20 if x-20>0 else 0
        x_max =  x+21 if x+21<w else w-1
        y_min =  y-20 if y-20>0 else 0
        y_max =  y+21 if y+21<h else h-1

        if dst[x_min:x_max,y_min:y_max].shape[0] > 0 and dst[x_min:x_max,y_min:y_max].shape[1] > 0:
            descriptor = cv2.resize(dst[x_min:x_max,y_min:y_max],(8,8),interpolation=cv2.INTER_CUBIC).reshape(-1)

            desList.append([x,y,theta[x,y],descriptor])
    return desList


if __name__ == '__main__':
    
    # read the image and convert to gray scale
    x = preImg()
    featureList = []
    # do Harris corner detect and return TopX response
    for idx,img in enumerate(x):
        print('Doing Harris corner detector on Image %d' %(idx))
        fList = Harris(img)
        fList = search4TopXResponse(fList,args.feature_num)
        fList = np.delete(np.array(fList),-1,1)
        print('Get Top %d features on Image %d' %(fList.shape[0],idx))
        featureList.append(fList)
    
    featureList = np.array(featureList,dtype=np.int32)
    # featureList format
    # type 3D numpy array with dimension (image_num,args.feature_num, 2)

    desList = []
    for idx,img in enumerate(x):
        desList.append(DesGen(img,featureList[idx]))

    
    # desList:  shape = (image_num,feature_num, )
    #                              ~~~~~~~~~~~~
    #                              the feature_num may smaller than args.features_num
    #
    #  each element in desList have:
    #
    #    - feature_X          : int
    #    - feature_Y          : int
    #    - feature_Theta      : float
    #    - feature_descriptor : np.array of size (64,)
    
    print(desList[0][0]) # print the img0 #0 feature description