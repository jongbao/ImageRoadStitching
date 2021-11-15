#!/usr/bin/env python
# coding: utf-8

import cv2
import pandas as pd
import numpy as np

class Stitcher: 
    """
    정사영 이미지들을 Stacking하여 파노라마를 생성하는 class
    """
    
    def __init__(self,imageList):
		# determine if we are using OpenCV v3.X and initialize the
		# cached homography matrix
        self.imageList = imageList
        #self.isv3 = imutils.is_cv3()
        #self.cachedH = None
        
        
    def Stacking(self,stop):
        imageList = self.imageList
      
        add_img = cv2.hconcat(imageList[:stop])
            
        return add_img






