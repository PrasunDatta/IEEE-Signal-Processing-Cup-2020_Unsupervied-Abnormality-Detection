# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:21:12 2020

@author: shafi
"""




import numpy as np
import cv2 as cv
import time
start = time.time()
npTmp = np.random.random((1024, 1024)).astype(np.float32)
stop1 = time.time()
print('Time Required for loading into cpu:'+ str(stop1-start))
npMat1 = np.stack([npTmp,npTmp],axis=2)
#npMat2 = npMat1
cuMat1 = cv.cuda_GpuMat()
#cuMat2 = cv.cuda_GpuMat()
cuMat1.upload(npMat1)
stop2 = time.time()
print('Time Required for uploading from cpu to gpu:'+ str(stop2-start))
#cuMat2.upload(npMat2)
#cv.cuda.gemm(cuMat1, cuMat2,1,None,0,None,1)