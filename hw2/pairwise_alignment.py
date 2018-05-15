# Use RANSAC to get the transform between neighboring images
# Usage:
# python3 pairwise_alignment.py -d <data directory>
# <data directory> should contain warpped images and `feature_matching.npy`

import os
import argparse
import cv2
import numpy as np

# RANSAC:
# P=0.99, p=0.6, n=1 => k>=6
# Instead of randomly pick n samples in k trials,
# pick from the samples with toppest 50 score
# output: best_transform, max_c, where best_transform=[dx, dy]

def ransac(feature_pairs):
    img0_features = feature_pairs[:,:2]
    img1_features = feature_pairs[:,2:]
    best_transform = None # [dx, dy]
    max_c = 0 # c: number of inliers
    for k in range(50):
        dx, dy = img0_features[k] - img1_features[k]
        distance = img0_features - (img1_features + [dx, dy])
        inliers = distance[np.sqrt(np.sum(distance**2, axis=1)) < 6]
        c = inliers.shape[0]
        if c > max_c:
            max_c = c
    #         best_transform = [dx, dy] + np.mean(inliers, axis=0)
            best_transform = [dx, dy]
    return best_transform, max_c

def main(args):
    feature_matching = np.load('%s/feature_matching.npy'%args.data_dir)
    pairwise_alignment = []
    for i in range(feature_matching.shape[0]):
        img0 = cv2.imread('%s/%d.jpg'%(args.data_dir, i))
        img1 = cv2.imread('%s/%d.jpg'%(args.data_dir, i+1))
        best_transform, max_c = ransac(feature_matching[i])
        pairwise_alignment.append(best_transform)
    pairwise_alignment = np.array(pairwise_alignment)
    np.save('%s/pairwise_alignment.npy'%args.data_dir, pairwise_alignment)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, required=True)
    main(parser.parse_args())