#import sys

import os
import numpy as np
import cv2

numSegments = 20

def findMus(image):
    vector = np.float32(image.reshape(-1,3))
    compactness, labels, centers = cv2.kmeans(vector,numSegments,None,(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
              1, 2),2,cv2.KMEANS_RANDOM_CENTERS)

    tpc_pmf_init = [i/sum(i) for i in centers]
    priors_init = [[1/numSegments for i in range(numSegments)] for j in range(numSegments)]
    weights = [[1/numSegments for i in range(numSegments)] for j in range(vector.max())]

    result = centers[labels.flatten()]
    segmented = result.reshape((image.shape))
    return segmented.astype(np.uint8)

def main():
    os.chdir('/path/to/project')

#TEST IMAGES
    #image = cv2.imread("balloons.jpg")
    #image = cv2.imread("ocean.jpg")
    image = cv2.imread("nature.jpg")

#SPECIAL IMAGES
    image= cv2.imread("polarlights.jpg")

    cv2.imshow('',findMus(image))
    cv2.waitKey(0)
    pass

if __name__ == '__main__':
    main()
