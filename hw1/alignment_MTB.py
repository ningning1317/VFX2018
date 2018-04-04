# Implemnt Median Threshold Bitmap for HDR image alignment
# Usage:
# gray_base = cv2.imread('base_path', 0)
# to_align = cv2.imread('to_align_path')
# gray_to_align = cv2.cvtColor(to_align, cv2.COLOR_BGR2GRAY)
# direc = pyramid(gray_base, gray_to_align, scale=1)
# shifted = shift(to_align, direc)
# cv2.imwrite('stored_path', shifted)

import os
import cv2
import numpy as np

# setting global parameters
directions = [np.array([[a],[b]]) for a in range(-1,2) for b in range(-1,2)]
scale_num = 4

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
    for direc in directions:
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
    if scale < 2**(scale_num-1):
        scale_base = cv2.resize(gray_base, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        scale_to_align = cv2.resize(gray_to_align, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        basic_direc = pyramid(scale_base, scale_to_align, scale*2)
    else:
        basic_direc = np.zeros((2,1))
    return get_best_direc(gray_base, gray_to_align, basic_direc*2)
