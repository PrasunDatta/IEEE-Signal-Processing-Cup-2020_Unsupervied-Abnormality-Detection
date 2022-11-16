import os
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from calculate_flows import calculate_flows
from Architecture import ConvAutoencoder
from utils import get_info_path
import json



def get_extracted_dir(line):
    bag_name = os.path.basename(line).split('.')[0]
    dir_name = os.path.join('data', bag_name)
    return dir_name


def noise_input(feature, NOISE_RATIO):
    return feature * (1 - NOISE_RATIO) + torch.rand_like(feature) * NOISE_RATIO


def train_model_and_save_weights(bag_paths='train_bags.txt', batch_size=32, learning_rate=1e-3, num_epochs=2
                                 , NOISE_RATIO=0.1, model_save_path='.', model_info=None):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    with open(bag_paths) as f: 
        all_train_flow_paths = []
        for line in f: 
            extracted_dir = get_extracted_dir(line)
            train_flow_path = pd.read_csv(os.path.join(extracted_dir, 'flows.csv'))
            all_train_flow_paths.append(train_flow_path)
    
    all_train_flow_paths = pd.concat(all_train_flow_paths, ignore_index=True)
    
    train_data = flowdata(all_train_flow_paths, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


    model = ConvAutoencoder().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)  

    start_time = time.time()
    log_loss=[]
    
    for epoch in range(1, num_epochs+1):
        running_loss = 0.
        cnt = 0
        model.train()
        for X in train_loader:
            X = X.to(device)
            num_examples = X.shape[0]
            noisy_X = noise_input(X, NOISE_RATIO)
            optimizer.zero_grad()
            output = model(noisy_X)
            loss = loss_fn(output, X)
            loss.backward()
            optimizer.step()

            running_loss = (cnt * running_loss + loss.item()*num_examples) / (cnt + num_examples)
            cnt += num_examples
        
        log_loss.append(running_loss)

    elapsed_time = time.time() - start_time
    print('Training Complete')
    print('Time Elapsed: {:.0f}m {:.0f}s\n'.format(elapsed_time // 60, elapsed_time % 60))

    plt.figure()
    plt.plot(log_loss)
    plt.xlabel('epoch')
    plt.ylabel('training loss')
    plt.show()

    wghts_pth = os.path.join(model_save_path, 'optical_wghts.pth')
    torch.save(model.state_dict(), wghts_pth)

    dataloader = DataLoader(train_data, batch_size=10, shuffle=False)
    loss_list = []
    model.eval()
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            y = model(X)
            loss = torch.mean((y - X) ** 2, axis=[1, 2, 3])
            loss_list.extend(loss)

        all_loss = torch.tensor(loss_list)
        mean = torch.mean(all_loss).cpu().numpy()
        std = torch.std(all_loss).cpu().numpy()

        mean_std = pd.DataFrame({
            'mean_loss': [mean],
            'std_loss': [std]
        })
        csv_pth = os.path.join(model_save_path, 'mean_std_loss.csv')
        mean_std.to_csv(csv_pth, index=False)

    del model
    del X
    torch.cuda.empty_cache()

    if model_info:
        model_info['Total Optical Flow (Images)'] = len(train_data)
    with open(get_info_path(model_save_path), 'w') as outfile:
        json.dump(model_info, outfile)
    
    return None


class flowdata(Dataset):
  
  def __init__(self, all_train_flow_paths, transform=None):
    
    self.flow_df = all_train_flow_paths
    self.transform = transform
  
  def __getitem__(self, index):
        
    pth = self.flow_df['flow_path'][index]
    flow = cv.readOpticalFlow(pth)

    if self.transform is not None:
        flow = self.transform(flow)
    
    return flow

  def __len__(self):
    return self.flow_df.shape[0]


def calculate_flow_and_train(model_path, about_model):
    bag_paths = 'train_bags.txt'

    if not os.path.exists(bag_paths):
        print("Train file not found")
    else:
        calculate_flows(bag_paths)
        train_model_and_save_weights(bag_paths, model_save_path=model_path, model_info=about_model)

            
if __name__ == "__main__":
    
    bag_paths = 'train_bags.txt'
    
    if not os.path.exists(bag_paths):
        print("Train file not found")
    else:
        calculate_flows(bag_paths)
        train_model_and_save_weights(bag_paths)
