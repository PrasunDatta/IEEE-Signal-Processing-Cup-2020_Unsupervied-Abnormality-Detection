import os
import pandas as pd
import numpy as np
import cv2 as cv
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import get_file_name_without_ext, get_bag_info_path, get_info_path


from calculate_flows import calculate_flows
from Architecture import ConvAutoencoder
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
            loss = torch.mean((y - X) ** 2, (1, 2, 3))
            loss_list.extend(loss.cpu().numpy())
            time_list.extend(time.numpy())
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


def calculate_and_test(model_path='models/1st Normal Data(22 Nov)'):
    bag_path = 'test_bag.txt'
    if not os.path.exists(bag_path):
        print("Test file not found")
        return None
    else:
        calculate_flows(bag_path)
        loss_list, time_list = calculate_test_loss(bag_path, model_path)

        dataframe = pd.read_csv(os.path.join(model_path, 'mean_std_loss.csv'))
        mean = dataframe['mean_loss'][0]
        std = dataframe['std_loss'][0]

        abn_score = normal_dist(loss_list, mean, std)

        return abn_score, time_list

def create_flow_dir(extracted_dir):
    flow_dir = os.path.join(extracted_dir, 'flows')
    if os.path.exists(os.path.join(flow_dir, '2.flo')):
        return None
    os.makedirs(flow_dir, exist_ok=True)

    return flow_dir

class RealTimeImgTest():
    def __init__(self, model_path='models/Submitted'):
        self.model_path = model_path
        self.result = True

        if not os.path.exists(get_info_path(model_path)):
            print("Model not found")
            print("Have you deleted the submitted model?")
            self.result = None

        if not os.path.exists('test_bag.txt'):
            print("Test file not found")
            self.result = None
        f = open('test_bag.txt')
        line = f.readline()  # reads text line by line
        if not line:
            print("No test bag file given")
            self.result = None
        filename = get_file_name_without_ext(line)

        if os.path.exists(get_bag_info_path(filename)):

            extracted_dir = get_extracted_dir(line)
            image_csv = os.path.join(extracted_dir, 'images.csv')
            dataframe = pd.read_csv(image_csv)
            self.image_times = dataframe['time']
            self.image_paths = dataframe['image_path']
        else:
            print('the path dosn\'t exist')
            self.result = None

        first_frame = cv.imread(self.image_paths[0])
        
        ratio = 256 / first_frame.shape[1]
        self.dim = (256, int(first_frame.shape[0] * ratio))
        
        first_frame = cv.rotate(first_frame, cv.ROTATE_180)

        if cv.cuda.getCudaEnabledDeviceCount() >= 1:
            first_frame_gpu = cv.cuda_GpuMat()
            first_frame_gpu.upload(first_frame)
            first_frame_gpu = cv.cuda.resize(first_frame_gpu, self.dim, interpolation=cv.INTER_AREA)
            first_frame_gpu = cv.cuda.cvtColor(first_frame_gpu, cv.COLOR_BGR2GRAY)
            #first_frame_gpu = cv.cuda.rotate(np.float32(first_frame_gpu), cv.ROTATE_180)
            self.first_frame_gpu = first_frame_gpu
        else:
            first_frame = cv.resize(first_frame, self.dim, interpolation=cv.INTER_AREA)
            first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
            #first_frame = cv.rotate(first_frame, cv.ROTATE_180)
            self.first_frame = first_frame

        self.second_frame = None
        self.second_frame_gpu = None

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        wghts_pth = os.path.join(model_path, 'optical_wghts.pth')
        self.model = ConvAutoencoder().to(self.device)
        model_state = torch.load(wghts_pth, map_location=self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()

        dataframe = pd.read_csv(os.path.join(model_path, 'mean_std_loss.csv'))
        self.mean = dataframe['mean_loss'][0]
        self.std = dataframe['std_loss'][0]

        self.transform=transforms.ToTensor()

        self.flow_csv_path = os.path.join(extracted_dir, 'flows.csv')
        self.flow_df = pd.read_csv(self.flow_csv_path)



    def calculate_step(self, index):
        global model_g
        time = self.image_times[index]
        abnormality = None
        if index >= 1:
            second_frame = cv.imread(self.image_paths[index])
            
            second_frame = cv.rotate(second_frame, cv.ROTATE_180)

            if cv.cuda.getCudaEnabledDeviceCount() >= 1:
                second_frame_gpu = cv.cuda_GpuMat()
                second_frame_gpu.upload(second_frame)
                second_frame_gpu = cv.cuda.resize(second_frame_gpu, self.dim, interpolation=cv.INTER_AREA)
                second_frame_gpu = cv.cuda.cvtColor(second_frame_gpu, cv.COLOR_BGR2GRAY)
                #second_frame_gpu = cv.cuda.rotate(np.float32(second_frame_gpu), cv.ROTATE_180)

                optical_flow = cv.cuda.OpticalFlowDual_TVL1_create()
                flow = optical_flow.calc(self.first_frame_gpu, second_frame_gpu, None)
                flow = flow.download()
                self.second_frame_gpu = second_frame_gpu
                self.first_frame_gpu = self.second_frame_gpu

            else:
                second_frame = cv.resize(second_frame, self.dim, interpolation=cv.INTER_AREA)
                second_frame = cv.cvtColor(second_frame, cv.COLOR_BGR2GRAY)
                #second_frame = cv.rotate(second_frame, cv.ROTATE_180)

                optical_flow = cv.optflow.createOptFlow_DualTVL1()
                flow = optical_flow.calc(self.first_frame, second_frame, None)

                self.second_frame = second_frame
                self.first_frame = self.second_frame

            X = self.transform(flow)
            x_input = torch.zeros(1,2,192,256,dtype=torch.float)
            x_input[0] = X
            X = x_input

            with torch.no_grad():
                X = X.to(self.device)
                y = self.model(X)
                loss = torch.mean((y - X) ** 2)
                abn_score = normal_dist([loss.cpu().numpy()], self.mean, self.std)
                abnormality = abn_score[0]
        return abnormality, time
    def clear_session(self):
        return None

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
        calculate_test_loss(bag_path)