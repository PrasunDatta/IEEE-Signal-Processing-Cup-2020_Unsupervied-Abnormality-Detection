# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:32:06 2020

@author: shafi
"""

import numpy as np
import os

from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from featureScale_test import featureScale_test
from create_dataset import create_dataset
from utils import get_file_name_without_ext, get_sensor_csv_path, get_bag_info_path, get_new_model_path,\
    get_scale_params_path, get_trained_model_path, get_model_info_path, get_mean_loss_path, get_sigma_loss_path,\
    get_info_path, get_default_model_path
import json
import math

#%%

model_path = 'C:/Users/shafi/Downloads/Sp Cup 2020 Resources/AbnormalityDetection/models/Submitted'

#make change to test_bag.txt

with open(get_info_path(model_path)) as json_file:
        data = json.load(json_file)
        time_step = data['Sequence Length (Sensor)']
        
        
f = open('test_bag.txt')
line = f.readline() #reads text line by line

filename = get_file_name_without_ext(line)
print(filename)


testpath = get_sensor_csv_path(filename)
df_test = pd.read_csv(testpath)

test_size = int(len(df_test))
print(test_size)
col_list = list(df_test.columns)

loaded_params = np.loadtxt(get_scale_params_path(model_path))
df_test = featureScale_test(df_test, col_list, loaded_params)

X_test = create_dataset(df_test[col_list[1:]], time_step)
timestamps = df_test.iloc[:, 0].values

json_file = open(get_model_info_path(model_path))
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json)

# load weights into new model
model.load_weights(get_trained_model_path(model_path))
print("Loaded model from disk")
model.compile(loss='mae', optimizer='adam')
X_test_pred = model.predict(X_test)

X_test_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

p = np.zeros(len(X_test_loss))
q = np.zeros(len(X_test_loss))
r = np.zeros(len(X_test_loss))
s = np.zeros(len(X_test_loss))

mu = np.loadtxt(get_mean_loss_path(model_path))
sigma = np.loadtxt(get_sigma_loss_path(model_path))

for i in range(len(X_test_loss)):
   p[i], q[i], r[i], s[i] = normal_dist_sensor(X_test_loss[i], mu, sigma, 6, 3, 0.06)

five_normal = s

np.savetxt(filename+'_sensor_scores.txt',five_normal)

#%%

plt.clf()

plt.plot(range(1,len(five_normal)+1), five_normal, label = 'five_normal')
#plt.plot(range(1,154), first_normal, label = 'first_normal')
plt.legend()
#plt.savefig('B2_sensor.png')
plt.show()



#%% Score Function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def scaled_shifted_sigmoid(x, scale, shift):
    return sigmoid(x/scale-shift)


def normal_dist_sensor(x, mu, sigma, scaling_factor, shifting_factor, min_sigma):
    p=0
    for i in range(3):
        effecive_sigma = max(sigma[i], min_sigma)
        cutoff_loss =  mu[i] + shifting_factor*effecive_sigma
        scaling = scaling_factor*effecive_sigma/8
        shift = cutoff_loss/scaling + 4
        p = p+(scaled_shifted_sigmoid(x[i], scaling, shift))
    q = 0
    for i in range(3,6):
        effecive_sigma = max(sigma[i], min_sigma)
        cutoff_loss =  mu[i] + shifting_factor*effecive_sigma
        scaling = scaling_factor*effecive_sigma/8
        shift = cutoff_loss/scaling + 4
        q = q+(scaled_shifted_sigmoid(x[i], scaling, shift))
    r = 0
    for i in range(6, 10):
        effecive_sigma = max(sigma[i], min_sigma)
        cutoff_loss = mu[i] + shifting_factor * effecive_sigma
        scaling = scaling_factor * effecive_sigma / 8
        shift = cutoff_loss / scaling + 4
        r = r + (scaled_shifted_sigmoid(x[i], scaling, shift))
    return p/3, q/3, r/4, (p+q+r)/10












        
        
