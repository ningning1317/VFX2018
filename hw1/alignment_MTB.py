# Implemnt Median Threshold Bitmap for HDR image alignment
# Usage:
# python3 alignment_MTB.py -d <data directory>

import os
import argparse
import cv2
import numpy as np

# setting global parameters
DIRECTIONS = [np.array([[a],[b]]) for a in range(-1,2) for b in range(-1,2)]
SCALE_NUM = 5
BASE_IMGS = ['1_8.jpg', '2_25.jpg', '3_25.jpg'] # hand picked

def get_bitmap_and_exclution(gray):
    bitmap = np.array(gray)
    median = np.median(bitmap)
    bitmap[bitmap<median] = 0
    bitmap[bitmap>=median] = 1

    exclusion_bitmap = cv2.inRange(gray, median-5, median+5)
    exclusion_bitmap = np.logical_not(exclusion_bitmap)
    exclusion_bitmap = exclusion_bitmap.astype(np.uint8)
    return bitmap, exclusion_bitmap

def shift(image_to_shift, direc):
    rows, cols = image_to_shift.shape[:2]
    translation = np.hstack((np.eye(2), direc))
    return cv2.warpAffine(image_to_shift, translation, (cols, rows))

def get_best_direc(gray_base, gray_to_align, basic_direc):
    bitmap_base, exclusion_base = get_bitmap_and_exclution(gray_base)
    bitmap_to_align, exclusion_to_align = get_bitmap_and_exclution(gray_to_align)
    best_direc = None
    min_err = 1000000
    for direc in DIRECTIONS:
        direc = direc + basic_direc
        bitmap_shifted = shift(bitmap_to_align, direc)
        exclusion_shifted = shift(exclusion_to_align, direc)
        xor = np.logical_xor(bitmap_base, bitmap_shifted)
        err = np.logical_and(xor, exclusion_base)
        err = np.logical_and(err, exclusion_shifted)
        err = np.sum(err)
        if err < min_err:
            min_err = err
            best_direc = direc

    return best_direc

def pyramid(gray_base, gray_to_align, scale=1):
    if scale < 2**(SCALE_NUM-1):
        scale_base = cv2.resize(gray_base, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        scale_to_align = cv2.resize(gray_to_align, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        basic_direc = pyramid(scale_base, scale_to_align, scale*2)
    else:
        basic_direc = np.zeros((2,1))
    return get_best_direc(gray_base, gray_to_align, basic_direc*2)

def main(args):
    ORIGIN_DIR = args.data_dir
    ALIGN_DIR = '%s/aligned'%ORIGIN_DIR
    if not os.path.exists(ALIGN_DIR):
        os.makedirs(ALIGN_DIR)

    with open('%s/info.txt'%ALIGN_DIR, 'w') as log:
        log.write('filename, shutter, shift_x, shift_y\n')
    for base_img in BASE_IMGS:
        image_series_num = base_img[:2]
        gray_base = cv2.imread('%s/%s'%(ORIGIN_DIR, base_img), 0)
        images = []
        for f in os.listdir(ORIGIN_DIR):
            if image_series_num in f:
                images.append(f)
        # sort by shutter time
        images = sorted(images, key=lambda f: int(f.split('.')[0].split('_')[1]))

        for f in images:
            to_align = cv2.imread('%s/%s'%(ORIGIN_DIR, f))
            gray_to_align = cv2.cvtColor(to_align, cv2.COLOR_BGR2GRAY)
            direc = pyramid(gray_base, gray_to_align, scale=1)
            shifted = shift(to_align, direc)
            cv2.imwrite('%s/%s'%(ALIGN_DIR, f), shifted)
            shutter = int(f.split('.')[0].split('_')[1])
            with open('%s/info.txt'%ALIGN_DIR, 'a') as log:
                log.write('%s, %d, %d, %d\n'%(f, shutter, direc[0], direc[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str,
                        default='jpg', dest='data_dir',
                        help='Data directory')
    main(parser.parse_args())
