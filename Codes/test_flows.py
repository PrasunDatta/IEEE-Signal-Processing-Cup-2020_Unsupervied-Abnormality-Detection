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


def calculate_test_loss(bag_path, model_path='.'):

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
            
    
    """error_df = pd.DataFrame({
                        'time': time_list,
                        'recons_err': loss_list
                    })
    #error_df['avg_recons_err'] = error_df['recons_err'].rolling(5,1).mean()

    loss_filename = 'tem_loss_img.csv'
    error_df.to_csv(loss_filename, index=False)"""
    del model
    del X
    torch.cuda.empty_cache()
    return loss_list, time_list
        
class flowfolders(Dataset):
  
  def __init__(self, flow_csv_path, transform=None):
        
    self.flow_df = pd.read_csv(flow_csv_path)
    self.transform = transform
  
  def __getitem__(self, index):
    time = self.flow_df['time'][index]
    pth = self.flow_df['flow_path'][index]
    flow = cv.readOpticalFlow(pth)

    if self.transform is not None:
        flow = self.transform(flow)
    
    return time, flow

  def __len__(self):
    return self.flow_df.shape[0]

def calculate_and_test(model_path):
    bag_path = 'test_bag.txt'
    if not os.path.exists(bag_path):
        print("Test file not found")
        return None
    else:
        calculate_flows(bag_path)
        loss_list, time_list = calculate_test_loss(bag_path, model_path)

        """process = multiprocessing.Process(target=calculate_test_loss_fun, args=(bag_path, model_path,))
        process.start()
        process.join()
        dataframe = pd.read_csv('tem_loss_img.csv')
        loss_list = dataframe['recons_err']
        time_list = dataframe['time']"""

        dataframe = pd.read_csv(os.path.join(model_path, 'mean_std_loss.csv'))
        mean = dataframe['mean_loss'][0]
        std = dataframe['std_loss'][0]

        abn_score = normal_dist(loss_list, mean, std)

        return abn_score, time_list

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def scaled_shifted_sigmoid(x, scale, shift):
    return sigmoid(x/scale-shift)


def normal_dist(x, mu, sigma, scaling_factor=10, shifting_factor=1, min_sigma=0.5):

    effecive_sigma = max(sigma, min_sigma)
    cutoff_loss = mu + shifting_factor*effecive_sigma
    scaling = scaling_factor*effecive_sigma/8
    shift = cutoff_loss/scaling + 4
    p= []
    for loss in x:
        p.append(scaled_shifted_sigmoid(loss, scaling, shift))
    return p

if __name__ == "__main__":
    
    bag_path = 'test_bag.txt'
    if not os.path.exists(bag_path):
        print("Test file not found")
    else:
        calculate_flows(bag_path)
        #calculate_test_loss(bag_path)