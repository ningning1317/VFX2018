# Usage:
# python3 cylindrical_warping.py -d <data directory>
# <data directory> should contain only images to be warpped
# and `focal.txt` which records the focal lengths of the images

import os
import argparse
import cv2
import numpy as np

def load_imgs(data_dir):
    imgs = {}
    for f in os.listdir('%s'%data_dir):
        if '.jpg' in f:
            img = cv2.imread('%s/%s'%(data_dir, f), -1)
            imgs[f] = img
    return imgs

def load_focals(data_dir):
    focals = {}
    with open('%s/focal.txt'%data_dir) as f:
        for line in f:
            line = line.rstrip().split(',')
            file_name, focal = line[0], float(line[1])
            focals[file_name] = focal
    return focals

def cylindrical_warp(img, f):
    warp_img = np.zeros(img.shape, dtype=np.uint8)
    height, width = img.shape[0], img.shape[1]
    center_x, center_y = width//2, height//2

    for warp_y in range(img.shape[0]):
        for warp_x in range(img.shape[1]):
            x = np.tan((warp_x-center_x) / f) * f
            y = (warp_y-center_y) * ((x**2 + f**2)**(0.5)) / f
            x += center_x
            y += center_y
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            warp_img[warp_y, warp_x] = img[int(y), int(x)]

    return warp_img

def main(args):
    if not os.path.exists('%s/warp'%args.data_dir):
        os.makedirs('%s/warp'%args.data_dir)
    imgs = load_imgs(args.data_dir)
    focals = load_focals(args.data_dir)
    for img_name in imgs:
        warp_img = cylindrical_warp(imgs[img_name], focals[img_name])
        cv2.imwrite('%s/warp/%s.png'%(args.data_dir, img_name.split('.')[0]), warp_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, required=True)
    main(parser.parse_args())
