# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:17:38 2020

@author: shafi
"""

import numpy as np
import scipy
#%% 

sensor_mean = np.zeros((12))
sensor_dev = np.zeros((12))
image_mean = np.zeros((12))
image_dev = np.zeros((12))

m =['A','B']
for j in m: 

    for i in range(0,6):
        
        sensor_file = './All Scores Formatted/Normal File Scores/'+str(j)+str(i)+'_sensor_scores.txt'
        
        image_file = './All Scores Formatted/Normal File Scores/'+str(j)+str(i)+'_image_scores.txt'
        
        print(sensor_file)
        print(image_file)
        
        readSensor = np.loadtxt(sensor_file)
        
        readImage = np.loadtxt(image_file)
        
        if j=='A': 
            
        
            sensor_mean[i] = np.mean(readSensor)
            
            sensor_dev[i] = np.std(readSensor)
            
            
            image_mean[i] = np.mean(readImage)
            
            image_dev[i] = np.std(readImage)
        
        else:
            
            sensor_mean[i+6] = np.mean(readSensor)
            
            sensor_dev[i+6] = np.std(readSensor)
            
            
            image_mean[i+6] = np.mean(readImage)
            
            image_dev[i+6] = np.std(readImage)
    
    
sensor_mean = np.reshape(sensor_mean, (12,1))
sensor_dev = np.reshape(sensor_dev, (12,1))
image_mean = np.reshape(image_mean, (12,1))
image_dev = np.reshape(image_dev, (12,1))


stats_all = np.concatenate((sensor_mean, sensor_dev, image_mean, image_dev),axis=1)
    

np.savetxt('stats_all.txt',stats_all)    
    
    
    
    
    