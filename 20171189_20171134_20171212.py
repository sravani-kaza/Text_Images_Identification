# -*- coding: utf-8 -*-
"""
Code for DSAA project
Usage: Match noisy slides extracted from video to original slides in ppt

"""

from os import listdir
from os.path import isfile, join
import sys
import cv2 as cv


# initialising sift, flann matcher parameters
sift = cv.xfeatures2d.SIFT_create()
KDTREE_INDEX = 1
TREE_LEN = 5
CHECK_NUM = 50
index_params = dict(algorithm = KDTREE_INDEX, trees = TREE_LEN)
search_params = dict(checks = CHECK_NUM)
flann = cv.FlannBasedMatcher(index_params, search_params)


# get path of slides and frames
path_frames = sys.argv[2]    
path_slides = sys.argv[1]

#extract  names of image files
frames = [f for f in listdir(path_frames) if isfile(join(path_frames, f))]
slides = [f for f in listdir(path_slides) if isfile(join(path_slides, f))]
frames = sorted(frames)
slides = sorted(slides)

# matrix mat stores number of matches for ith frame and jth slide
mat = [[0 for i in range(len(slides))] for j in range(len(frames))]
sift = cv.xfeatures2d.SIFT_create()


#algo for feature matcher 
for i in range(len(frames)):
    # read ith frame and find features
    imagei = cv.imread(path_frames+frames[i], cv.IMREAD_GRAYSCALE)
    kpi, desi = sift.detectAndCompute(imagei, None)
    for j in range(len(slides)):
        # read jth slide and find features
        imagej = cv.imread(path_slides+slides[j], cv.IMREAD_GRAYSCALE)
        kpj, desj = sift.detectAndCompute(imagej, None)
        best_matches = 2;                         
        matches = flann.knnMatch(desi, desj, k=best_matches)
        # fing good matches only
        # ratio test as per Lowe's paper
        m_len = len(matches)
        for k, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                mat[i][j] = mat[i][j] + 1

# find index with maximum number of matches for each frame
slides_order = ['0' for i in range(len(frames))]
frame_ind = 0
for k in mat:
    maxpos = k.index(max(k))
    # replace index by the name of the slide
    slides_order[frame_ind] = slides[maxpos]
    frame_ind = frame_ind + 1

# write to file
fp = open("20171189_20171134_20171212.txt", "w")
for i in range(len(frames)):
    fp.write(frames[i])
    fp.write(' ')
    fp.write(slides_order[i])
    fp.write('\n')

fp.close()