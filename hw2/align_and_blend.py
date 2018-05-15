# Align all warpped images and blend them to generate panorama
# Usage:
# python3 align_and_blend.py -d <data directory> --refine
# <data directory> should contain warpped images and `pairwise_alignment.npy`
# --r means to refine the result panorama

import os
import argparse
import cv2
import numpy as np

class Panorama():
    def __init__(self, init_img):
        self.img = init_img.copy()
        self.ori_h = self.img.shape[0]
        self.ori_x, self.ori_y = 0, 0
        self.last_tail = self.img.shape[1]
        self.last_top, self.last_bottom = 0, self.img.shape[0]

    def align_and_blend(self, new_img, dx, dy):
        h1, w1 = new_img.shape[0], new_img.shape[1]
        self.ori_x += dx
        self.ori_y += dy
        new_w = w1 + self.ori_x
        self.img = np.concatenate((self.img, np.zeros((self.img.shape[0], new_w-self.img.shape[1], 3))), axis=1)
        if self.ori_y < 0:
            self.last_top -= self.ori_y
            self.last_bottom -= self.ori_y
            self.img = np.concatenate((np.zeros((-self.ori_y, self.img.shape[1], 3)), self.img), axis=0)
            self.ori_y = 0
        else:
            new_h = h1+self.ori_y if (h1+self.ori_y) > self.img.shape[0] else self.img.shape[0]
            self.img = np.concatenate((self.img, np.zeros((new_h-self.img.shape[0], self.img.shape[1], 3))), axis=0)
        for y in range(h1):
            bound_y = True if self.ori_y+y > self.last_top and self.ori_y+y < self.last_bottom else False
            for x in range(w1):
                if np.sum(new_img[y,x]) > 0: # not margin caused by warpping
                    if bound_y and self.ori_x+x < self.last_tail: # in overlap area
                        if np.sum(self.img[self.ori_y+y,self.ori_x+x]) == 0: # margin caused by warpping
                            self.img[self.ori_y+y,self.ori_x+x] = new_img[y,x]
                        else:
                            w = 2**(x / (self.last_tail-self.ori_x)) - 1
                            self.img[self.ori_y+y,self.ori_x+x] *= (1-w)
                            self.img[self.ori_y+y,self.ori_x+x] += w * new_img[y,x]
                    else:
                        self.img[self.ori_y+y,self.ori_x+x] = new_img[y,x]
        self.last_tail = self.ori_x + w1
        self.last_top, self.last_bottom = self.ori_y, self.ori_y + h1

    def drift_refine(self):
        # inverse warpping by y' = y + mx
        new_img = np.zeros((self.ori_h, self.img.shape[1], 3), dtype=np.uint8)
        m = (self.img.shape[0] - self.ori_h) / self.img.shape[1]
        for warp_y in range(new_img.shape[0]):
            for warp_x in range(new_img.shape[1]):
                y = int(warp_y + warp_x * m)
                new_img[warp_y, warp_x] = self.img[y, warp_x]
        # to crop the black margin
        h_margin, w_margin = int(new_img.shape[0]*0.05), int(new_img.shape[1]*0.03)
        self.img = new_img[h_margin:-h_margin, w_margin:-w_margin]

    def save(self, path):
        cv2.imwrite(path, self.img)

def main(args):
    pairwise_alignment = np.load('%s/pairwise_alignment.npy'%args.data_dir)
    panorama = Panorama(cv2.imread('%s/%d.png'%(args.data_dir, 0), -1))

    for i in range(pairwise_alignment.shape[0]):
        img = cv2.imread('%s/%d.png'%(args.data_dir,i+1), -1)
        dx, dy = pairwise_alignment[i]
        panorama.align_and_blend(img, dx, dy)
    if args.refine:
        panorama.drift_refine()
        panorama.save('%s/refine_panorama.png'%args.data_dir)
    else:
        panorama.save('%s/panorama.png'%args.data_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, required=True)
    parser.add_argument('--refine', action='store_true')
    parser.set_defaults(refine=False)
    main(parser.parse_args())
