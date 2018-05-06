import cv2
import argparse
import numpy as np
import os
from tqdm import tqdm
import sys

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
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lst.append(gray_image)
    return np.array(lst)

def Harris(img):

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
    featureList = np.array(featureList)
    # featureList format
    # type 3D numpy array with dimension (image_num,args.feature_num, 2)