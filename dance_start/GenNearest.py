import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton


class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """

    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):
        """ generator of image from skeleton """
        min_distance = float('inf')
        indeximage = 0
        for i in range(self.videoSkeletonTarget.skeCount()):
            dist = ske.distance(self.videoSkeletonTarget.ske[i])
            if dist < min_distance:
                min_distance = dist
                indeximage = i
        if indeximage == 0:
            empty = np.ones((64, 64, 3), dtype=np.uint8) * 255
            return empty
        else:
            image = self.videoSkeletonTarget.readImage(indeximage)
            return image
