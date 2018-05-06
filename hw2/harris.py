import cv2
import argparse
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelmax
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


parser = argparse.ArgumentParser(description='Multi-Scale Oriented Patches')
parser.add_argument('-d', '--data_dir', type=str,default='data/')
parser.add_argument('--sigma',default=5)
parser.add_argument('--response_w',default=0.04)
parser.add_argument('--response_threshold',default=20)
args = parser.parse_args()

def preImg():
    lst = []
    for name in os.listdir(args.data_dir):
        image = cv2.imread(args.data_dir + name)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        lst.append(gray_image)
    return np.array(lst)

def weightF(x,y,xbase,ybase):
    return np.exp(( (x-xbase)**2 + (y-ybase)**2) / (2*(args.sigma**2))  * -1)



def gaussianMask(w,h,xbase,ybase):
    mask = np.zeros(shape=(w,h))

    for i in range(w):
        for j in range(h):
            mask[i,j] = weightF(i,j,xbase,ybase)
    return mask

# def gaussianMask(w,h,xbase,ybase):
#     mask = np.zeros(shape=(w,h))

#     num_cores = multiprocessing.cpu_count()
#     results = Parallel(n_jobs=num_cores)(delayed(weightF)(i,j,xbase,ybase) for i in range(w) for j in range(h))
#     mask = np.array(results).reshape((w,h))

#     return mask

def calRespose(w,h,i,j,Ixx,Ixy,Iyy):

    mask = gaussianMask(w,h,i,j)
    w_Ixx = np.sum(np.multiply(Ixx,mask))
    w_Ixy = np.sum(np.multiply(Ixy,mask))
    w_Iyy = np.sum(np.multiply(Iyy,mask))

    #     |w_Ixx w_Ixy|
    # M = |           |
    #     |w_Iyx w_Iyy|

    det = w_Ixx * w_Iyy - w_Ixy * w_Ixy
    tr = w_Ixx + w_Iyy
    response = det - args.response_w * (tr**2)

    return response

# def Harris(img):
#     Ix,Iy = np.gradient(img) 

#     Ixx = Ix**2
#     Iyy = Iy**2
#     Ixy = np.multiply(Ix,Iy)

#     w = img.shape[0]
#     h = img.shape[1]

#     featureList = []
#     responseList = []

#     for i in range(w):
#         for j in tqdm(range(h)):
#             response = calRespose(w,h,i,j,Ixx,Ixy,Iyy)

#             if response > args.response_threshold:
#                 featureList.append([i,j,response])

#     return featureList


def Harris(img):

    Ix,Iy = np.gradient(img) 

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = np.multiply(Ix,Iy)

    w = img.shape[0]
    h = img.shape[1]

    featureList = []

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores,verbose=30)(delayed(calRespose)(w,h,i,j,Ixx,Ixy,Iyy) for i in range(w) for j in range(h))

    response = np.array(results).reshape((w,h))

    for i in range(w):
        for j in range(h):
            if response[i,j] > args.response_threshold:
                featureList.append([i,j,response[i,j] ] )

    return featureList

if __name__ == '__main__':
    
    x = preImg()
    
    for img in x:
        fList = Harris(img)
        print(len(fList))