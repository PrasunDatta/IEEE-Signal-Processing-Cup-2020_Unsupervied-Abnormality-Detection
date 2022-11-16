import os
import glob
import numpy as np
import pandas as pd
import cv2 as cv

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


from calculate_flows import calculate_flows
from Architecture import ConvAutoencoder
#from numba import cuda
import multiprocessing
import math

def get_extracted_dir(line):
    bag_name = os.path.basename(line).split('.')[0]
    dir_name = os.path.join('data', bag_name)
    
    return dir_name

#%%
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def scaled_shifted_sigmoid(x, scale, shift):
    return sigmoid(x/scale-shift)


def normal_dist_flow(x, mu, sigma, scaling_factor=10, shifting_factor=1, min_sigma=0.5):

    effecive_sigma = max(sigma, min_sigma)
    cutoff_loss = mu + shifting_factor*effecive_sigma
    scaling = scaling_factor*effecive_sigma/8
    shift = cutoff_loss/scaling + 4
    p= []
    for loss in x:
        p.append(scaled_shifted_sigmoid(loss, scaling, shift))
    return p

#%%

model_path = 'C:/Users/shafi/Downloads/Sp Cup 2020 Resources/AbnormalityDetection/models/5 Normal Exp'
    

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
with open(bag_path) as f: 
    line = f.readline()
extracted_dir = get_extracted_dir(line)
flow_csv_path = os.path.join(extracted_dir, 'flows.csv')
            
test_data = flowfolders(flow_csv_path, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
    
wghts_pth = os.path.join(model_path, 'optical_wghts.pth')
model = ConvAutoencoder().to(device)
model_state = torch.load(wghts_pth, map_location=device)
model.load_state_dict(model_state)


time_list = []
loss_list = []
model.eval()
with torch.no_grad():
    for time, X in test_loader:
        X = X.to(device)
        y = model(X)
        loss = torch.mean((y - X)**2, axis=[1, 2, 3])
        loss_list.extend(loss.cpu().numpy())
        time_list.extend(time.numpy())



#%%
dataframe = pd.read_csv(os.path.join(model_path, 'mean_std_loss.csv'))
mean = dataframe['mean_loss'][0]
std = dataframe['std_loss'][0]

abn_score = normal_dist_flow(loss_list, mean, std)

np.savetxt(filename+'_image_scores.txt',abn_score)

plt.plot(range(1,len(abn_score)+1), abn_score)




