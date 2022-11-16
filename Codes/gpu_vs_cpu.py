# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 21:14:11 2020

@author: shafi
"""

import numpy as np
import matplotlib.pyplot as plt 

#%%

Cpu_normal = np.load('CpuTime_anom.npy')
Gpu_normal = np.load('GpuTime_anom.npy')

plt.clf()
plt.plot(Cpu_normal, label = 'CPU')
plt.plot(Gpu_normal, label = 'GPU')
plt.legend()
plt.title('CPU vs GPU Perfromance For Optical Flow Calculation')
plt.xlabel('Frame Number')
plt.ylabel('Seconds Required Per Frame')
