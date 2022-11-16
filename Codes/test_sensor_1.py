#%%
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
def test_sensor_values(model_path, scaling_factor, shifting_factor, min_sigma):
    if not os.path.exists(get_info_path(model_path)):
        print("Model not found")
        print("Have you deleted the submitted model?")
        return None

    with open(get_info_path(model_path)) as json_file:
        data = json.load(json_file)
        time_step = data['Sequence Length (Sensor)']

    if not os.path.exists('test_bag.txt'):
        print("Test file not found")
        return None
    f = open('test_bag.txt')
    line = f.readline() #reads text line by line
    if not line:
        print("No test bag file given")
        return None
    filename = get_file_name_without_ext(line)
    print(filename)

    if os.path.exists(get_bag_info_path(filename)):

        testpath = get_sensor_csv_path(filename)
        df_test = pd.read_csv(testpath)

        test_size = int(len(df_test))
        print(test_size)
        col_list = list(df_test.columns)

        loaded_params = np.loadtxt(get_scale_params_path(model_path))
        df_test = featureScale_test(df_test, col_list, loaded_params)

        X_test = create_dataset(df_test[col_list[1:]], time_step)
        timestamps = df_test.iloc[:, 0].values
    else:
        print('the path dosn\'t exist')
        return None
    line = f.readline()  # reads text line by line

    #%%
    # load json and create model
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
        p[i], q[i], r[i], s[i] = normal_dist(X_test_loss[i], mu, sigma, scaling_factor, shifting_factor, min_sigma)

    if __name__ == "__main__":
        plt.plot(np.array(range(len(p))), p, color='red')
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.xlabel('Time')
        plt.ylabel('Abnormality')
        plt.show()

    avoid_1 =  int((time_step-1)/2)
    avoid_2 = time_step-1-avoid_1
    time_axis = timestamps[avoid_1:-avoid_2]

    return p, q, r, s, time_axis

class RealTimeTestObj():
    def __init__(self, model_path, scaling_factor, shifting_factor, min_sigma):
        """Load options"""
        self.model_path = model_path
        self.scaling_factor = scaling_factor
        self.shifting_factor = shifting_factor
        self.min_sigma = min_sigma
        self.result = True

        if not os.path.exists(get_info_path(model_path)):
            print("Model not found")
            print("Have you deleted the submitted model?")
            self.result = None

        with open(get_info_path(model_path)) as json_file:
            data = json.load(json_file)
            time_step = data['Sequence Length (Sensor)']
            self.time_step = time_step
            self.delay = int(time_step/2)

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

            testpath = get_sensor_csv_path(filename)
            df_test = pd.read_csv(testpath)

            test_size = int(len(df_test))
            col_list = list(df_test.columns)

            loaded_params = np.loadtxt(get_scale_params_path(model_path))
            df_test = featureScale_test(df_test, col_list, loaded_params)

            self.X_test = create_dataset(df_test[col_list[1:]], time_step)
            self.total_seq = self.X_test.shape[0]
            timestamps = df_test.iloc[:, 0].values
            self.timestamps = timestamps
        else:
            print('the path dosn\'t exist')
            self.result = None
        line = f.readline()  # reads text line by line

        # %%
        # load json and create model
        json_file = open(get_model_info_path(model_path))
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = keras.models.model_from_json(loaded_model_json)

        # load weights into new model
        self.model.load_weights(get_trained_model_path(model_path))
        print("Loaded model from disk")
        self.model.compile(loss='mae', optimizer='adam')

        self.mu = np.loadtxt(get_mean_loss_path(model_path))
        self.sigma = np.loadtxt(get_sigma_loss_path(model_path))

    def calculate_step(self, index):
        time = self.timestamps[index]
        abnormality = None
        if index >= self.time_step-1 and index < self.total_seq:
            X_test = self.X_test[index]
            X_test_pred = self.model.predict(np.expand_dims(X_test, axis=0))
            X_test_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

            p,q,r,abnormality = normal_dist(X_test_loss[0], self.mu, self.sigma, self.scaling_factor, self.shifting_factor,
                                      self.min_sigma)
        return abnormality, time



#%% Score Function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def scaled_shifted_sigmoid(x, scale, shift):
    return sigmoid(x/scale-shift)


def normal_dist(x, mu, sigma, scaling_factor, shifting_factor, min_sigma):
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

#%%

"""def normal_dist(x,mu,sigma,scaling_factor, shifting_factor, min_sigma):
    p=0
    for i in range(len(x)):
        #gaussian = scipy.stats.norm(mu[i],sigma[i])
        # p = p*(gaussian.pdf(x[i])/gaussian.pdf(mu[i]))
        #p = p*(gaussian.cdf(x[i]))
        #cutoff_loss =  mu[i] + 3*sigma[i]
        effecive_sigma = max(sigma[i],0.06)
        cutoff_loss =  mu[i] + 3*effecive_sigma
        scaling = 6*effecive_sigma/8
        shift = cutoff_loss/scaling + 4
        p = p+(scaled_shifted_sigmoid(x[i],scaling,shift))
    return p/10, p/10, p/10, p/10"""


if __name__ == "__main__":
    test_sensor_values(get_default_model_path(), 6, 3, 0.06)
