# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:58:26 2020

@author: shafi
"""

import os
import pandas as pd
import cv2 as cv
import time 
import numpy as np
import matplotlib.pyplot as plt 
#%%
path_to_image = r'C:/Users/shafi/Downloads/Sp Cup 2020 Resources/AbnormalityDetection/data/2020-01-17-11-34-43/images/'

a = time.time()
lags_normal = np.zeros(0)

for i in range(len(os.listdir(path_to_image))-1):
    
     start = time.time()
    
     first_frame = cv.imread(path_to_image + '49.jpg')
     stop1 = time.time()
     print('Time Required for loading first image into cpu:'+ str(stop1-start))
     
     ratio = 256 / first_frame.shape[1]
     dim = (256, int(first_frame.shape[0] * ratio))
     
     first_frame_gpu = cv.cuda_GpuMat()
     first_frame_gpu.upload(first_frame)
     first_frame_gpu = cv.cuda.resize(first_frame_gpu, dim, interpolation=cv.INTER_AREA)
     first_frame_gpu = cv.cuda.cvtColor(first_frame_gpu, cv.COLOR_BGR2GRAY)
     stop2 = time.time()
     print('Time Required for uploading first image from cpu to gpu:'+ str(stop2-stop1) ) 
     
     
     second_frame = cv.imread(path_to_image + '50.jpg')
     
     stop3 = time.time()
     print('Time Required for loading second image into cpu:'+ str(stop3-stop2))
     
     
     second_frame_gpu = cv.cuda_GpuMat()
     second_frame_gpu.upload(second_frame)
     second_frame_gpu = cv.cuda.resize(second_frame_gpu, dim, interpolation=cv.INTER_AREA)
     second_frame_gpu = cv.cuda.cvtColor(second_frame_gpu, cv.COLOR_BGR2GRAY)
     
     stop4 = time.time()
     print('Time Required for uploading second image from cpu to gpu:'+ str(stop4-stop3))
      
     optical_flow = cv.cuda.OpticalFlowDual_TVL1_create()
     flow = optical_flow.calc(first_frame_gpu, second_frame_gpu, None)
     flow = flow.download()
      
     stop5 = time.time()
     print('Time Required for optical flow calculation:'+ str(stop5-stop4))
     
     print('Total time Required : '+ str(stop5-start) + '\n')
     
     lags_normal = np.append(lags_normal, stop5-start)
     
     #b = time.time()

     #print(b-a)      
stop6 = time.time()     
print('Total time required for whole bag file: '+ str(stop6-start))


plt.clf()
plt.plot(lags, label = '5th Abnormal Dataset, images = 53')
plt.plot(lags_normal, label = '5th Normal Dataset, images = 51')
plt.legend()

      
      
      
      
      
      
      
      
      
      
      
      
      
      