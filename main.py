#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import Stitching
import PerspectiveTransformer 
import cv2


# gps 정보 불러온 후, 투시 변환하여 정사영 이미지 생성
gps_path = pd.read_csv("GH011044_Hero6 Black-GPS5.csv")
video_path = "GH011044.mp4"
transformer = PerspectiveTransformer.PerspectiveTransformer(gps_path, video_path)
imageList = transformer.makeImagesList()

# Stitching : stop 변수를 통해 파노라마 생성에 사용할 정사영 이미지 개수 지정
stitcher = Stitching.Stitcher(imageList)
panorama = stitcher.Stacking(stop=100)

# save panorama
cv2.imwrite("panorama.jpg",panorama)

