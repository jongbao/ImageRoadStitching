#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 10:13:55 2021

@author: KONIDE
"""


##----------------Stitcher------------------

import numpy as np
import cv2
import imutils

class Line :
    """
    투시변환할 부분을 찾아주는 class
    """
    def __init__(self, data1, data2):
   
        self.line1 = data1
        self.line2 = data2
        #print(self.line1)
    def slope(self):
        (x1, y1), (x2, y2) = self.line1
        (x3, y3), (x4, y4) = self.line2
        
        if (y2-y1) == 0 :
            #print('Ys are equal, m1 = 0')
            m1 = 0
        else:
            m1 = (float(y2)-y1)/(float(x2)-x1)
        
        if (y4-y3) == 0 :
            #print('Ys are equal, m2 = 0')
            m2 = 0
        else:
            m2 = (float(y4)-y3)/(float(x4)-x3)
            
        return m1, m2
                    
    def yintercept(self, m1, m2):
        (x1, y1), (x2, y2) = self.line1
        (x3, y3), (x4, y4) = self.line2
        
        if m1 != 0 :
            b1 = y1 - m1*x1
        else :
            b1 = y1
            
        if m2 != 0 :
            b2 = y4 - m2*x4
            
        else: b2 = y4
        
        return b1, b2
    
    def findIntersect(self, m1,m2, b1, b2):
        
        if m1 != 0 | m2 != 0 :
            px = (b2-b1) / (m1-m2)
            py = (b2*m1 - b1*m2)/(m1-m2)
        elif m1 == 0 :
            px = (b1-b2)/m2
            py = b1
        elif m2 == 0 : 
            px = (b2-b1)/m1
            py = b2 
        else :  print('No points')
        
        return px, py
        

class IMP:
    """
    투시변환 실시하여 전경이미지 -> 정사영으로 변환해주는 class
    """
    def __init__(self, img):
        
        #import cv2
        #img = cv2.imread('c:/OpenCV/image-003.jpeg')     
        self.img = img
        
        #self.topHeight = 565
        #self.height, self.width = 1080, 1920
        
    def impTransformer(self):  
        
        import numpy as np
        import cv2 
        

        # 정사영 파라미터 변환 
        topHeight = 545
        height, width = self.img.shape[:2]
        left = [(960, 342), (0, 690)]
        right = [(960, 342), (1920, 650)]
        up =  [(0, topHeight), (width+1000, topHeight)]
        down =  [(-10000,height), (width+100000, height)]     
        
               
        leftup = Line(left, up)
        leftdown = Line(left, down)
        rightup = Line(right, up)
        rightdown = Line(right, down)
        m1, m2 = leftup.slope()
        b1, b2 = leftup.yintercept(m1,m2)
        p1x, p1y = leftup.findIntersect(m1,m2,b1,b2)
        
        #print('point1 : ', p1x, p1y)
        
       
        
        m1, m2 = leftdown.slope()
        b1, b2 = leftdown.yintercept(m1,m2)
        p2x, p2y = leftdown.findIntersect(m1,m2,b1,b2)
        #print('point2 : ', p2x, p2y)
        
       
        
        m1, m2 = rightup.slope()
        b1, b2 = rightup.yintercept(m1,m2)
        p3x, p3y = rightup.findIntersect(m1,m2,b1,b2)
        #print('point3 : ', p3x, p3y)
        
        m1, m2 = rightdown.slope()
        b1, b2 = rightdown.yintercept(m1,m2)
        p4x, p4y = leftup.findIntersect(m1,m2,b1,b2)
        #print('point4 : ', p4x, p4y)
        

        dst = np.array([[0,0], [0, height], [width,0], [width,height]], dtype=np.float32)
        src = np.array([ [p1x,p1y], [p2x,p2y], [p3x,p3y], [p4x,p4y]], dtype=np.float32)
        mtrx = cv2.getPerspectiveTransform(src, dst)
        
        transformedFHD = cv2.warpPerspective(self.img, mtrx, (width,height))
        ##  C++코드에서 정사영 자체는 원본 사이즈로 생성,  ( 1080, 560 ) -> (1080,1920)
        outimg = cv2.resize(transformedFHD,dsize=(960,540))
        #cv2.imshow('out_image',outimg)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        return outimg
    
    
class PerspectiveTransformer:
    """
    비디오에서 프레임 뽑아서 정사영 형태로 변환한 이미지들을 리스트에 저장하는 역할. 
    -> 속도(m/s)의 변화를 고려하여 프레임 추출 
    """
    def __init__(self, gps_path, video_path):
        self.gps_path = gps_path
        self.video_path = video_path
        

    # 보고서에서는 km/h 단위를 사용했지만 실제 csv는 m/s 이므로 13 -> 3.6 변경
    def CalcFrameSkip(self, speed, fps):
        skipVal = 0.0
        magicNum = 3.6
        if speed >= 3.6:
            skipVal = (magicNum * fps) / speed
        else:
            skipVal = (magicNum * fps) / 3.6
        return skipVal
    
    
    def makeImagesList(self):       
        gps_path = self.gps_path
        video_path = self.video_path
        cap = cv2.VideoCapture(video_path)  # 비디오 객체 생성

        fps = round(cap.get(cv2.CAP_PROP_FPS))  # get frame numbers per seconds
        if (fps == 0) :
            fps = 60 
            
        currentFrameNumber = 0
        currentFrameNumberTemp = 0
        skipGPS = 0
        result = []
        
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                currentFrameNumber += 1
                
                speed = gps_path['GPS (2D speed) [m/s]'][skipGPS]
                skipVal = self.CalcFrameSkip(speed,fps)

                if (currentFrameNumber != 1) & ((currentFrameNumber - skipVal) < currentFrameNumberTemp) :
                    continue

                currentFrameNumberTemp = currentFrameNumber
                skipGPS = skipGPS + round(skipVal / fps * 18)
                #print(currentFrameNumberTemp)
              
                curr_frame = IMP(frame)   
                curr_outimg = curr_frame.impTransformer() # 이미지 정사영 변환
                curr_cropimg = curr_outimg[0:250, 0:960] # 위에 있는 도로만 사용 -> (250, 1080)
                curr_cropimg = cv2.rotate(curr_cropimg, cv2.ROTATE_90_CLOCKWISE) # 시계방향으로 90도 회전 (1080, 250)
                result.append(curr_cropimg)
                
                #cv2.imwrite('./data/frame_front/' + 'frame'+ '_' + str(currentFrameNumber) + '.jpg', frame)
                #cv2.imwrite('./data/frameTest2/' + 'frame'+ '_' + str(currentFrameNumber) + '.jpg', curr_cropimg)
            #print(result)
            if currentFrameNumber == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break

        return result






