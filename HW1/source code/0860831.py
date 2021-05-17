import numpy as np
import cv2
from matplotlib import pyplot as plt
import libraryV2
import sys

if sys.argv[1] == 'bamboo_fox':
    img1 = cv2.imread('head.jpg', 0)  # queryImage
    img2 = cv2.imread('rotate_L.jpg', 0)  # trainImage
if sys.argv[1] == 'mountain':
    img1 = cv2.imread('m0.jpg', 0)  # queryImage
    img2 = cv2.imread('m1.jpg', 0)  # trainImage
if sys.argv[1] == 'tree':
    img1 = cv2.imread('0.jpg', 0)  # queryImage
    img2 = cv2.imread('1.jpg', 0)  # trainImage
if sys.argv[1] == 'my_test':
    img1 = cv2.imread('test1.jpg', 0)  # queryImage
    img2 = cv2.imread('test2.jpg', 0)  # trainImage

# implement sift
kp1, des1 = libraryV2.sift(img1)
kp2, des2 = libraryV2.sift(img2)

# plot all immages and DoG at n-octave
libraryV2.plot_all_image(img1)
libraryV2.plot_nth_DoG(img1,int(sys.argv[2]))

libraryV2.plot_all_image(img2)
libraryV2.plot_nth_DoG(img2,int(sys.argv[2]))

# plot keypoint
libraryV2.plot_keypoint(img1,kp1)
libraryV2.plot_keypoint(img2,kp2)

# plot match image
libraryV2.plot_match_image(img1,img2,kp1,kp2,des1,des2)
