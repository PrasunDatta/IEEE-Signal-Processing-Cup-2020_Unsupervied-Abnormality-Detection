#%%
import numpy as np
import csv
import tensorflow as tf
import os

from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import time
import datetime
import json
from featureScale_train import featureScale_train, applyFeatureScale_train
from create_dataset import create_dataset
from utils import get_file_name_without_ext, get_sensor_csv_path, get_bag_info_path, get_new_model_path,\
    get_scale_params_path, get_trained_model_path, get_model_info_path, get_mean_loss_path, get_sigma_loss_path,\
    get_info_path
from tensorflow.keras import backend as K
#from numba import cuda


def train_sensor_values(time_step=5):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    session = tf.compat.v1.Session(config=config)
    with session:
        if not os.path.exists('train_bags.txt'):
            print("Train file not found")
            return None
        f = open('train_bags.txt')
        line = f.readline()
        model_path = get_new_model_path()
        X_train = np.empty((0, time_step, 10))
        bag_count = 0
        bags = ''
        while line: #until line is null

            filename = get_file_name_without_ext(line)
            print(filename)

            if os.path.exists(get_bag_info_path(filename)):

                normal_trainpath = get_sensor_csv_path(filename)
                df_normaltrain = pd.read_csv(normal_trainpath)
                col_list = list(df_normaltrain.columns)

                DF_Train = df_normaltrain if bag_count.__eq__(0) else DF_Train.append(df_normaltrain, ignore_index = True)

                train = create_dataset(df_normaltrain[col_list[1:]], time_step)
                X_train = np.append(X_train, train, axis=0)
                bags = bags + (('' if bags.__len__().__eq__(0) else ', ')+filename)
                bag_count += 1
            else:
                print('the path dosn\'t exist')
                return None
            line = f.readline()  # reads text line by line

        f.close()
        if bag_count.__eq__(0):
            print("No train bag file is given")
            return None

        final_params = featureScale_train(DF_Train, col_list)
        np.savetxt(get_scale_params_path(model_path), final_params)
        X_train = applyFeatureScale_train(X_train, final_params)

        split_on = int(len(X_train) * 0.9)
        X_train_1 = X_train[:split_on]
        X_train_2 = X_train[split_on:]

        training_generator = DataGenerator(X_train_1, noise=True)
        validation_generator = DataGenerator(X_train_2)

        # %%
        model = keras.Sequential()
        model.add(keras.layers.LSTM(
            units=64,
            input_shape=(X_train.shape[1], X_train.shape[2])
        ))
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
        model.add(keras.layers.LSTM(units=64, return_sequences=True))
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
        model.compile(loss='mae', optimizer='adam')

        time_callback = TimeHistory()

        history = model.fit_generator(generator=training_generator,
                                      validation_data=validation_generator,
                                      use_multiprocessing=False,
                                      epochs=200, verbose=1, callbacks=[time_callback],
                                      workers=0)

        print("Elapsed Time During Training(in seconds): ")
        print(np.sum(np.array(time_callback.times)))

        """plt.plot(history.history['loss'], label='Train Set')
        plt.plot(history.history['val_loss'], label='Validation Set')
        plt.legend()
        plt.show()"""

        # %%
        # serialize model to JSON

        model_json = model.to_json()
        with open(get_model_info_path(model_path), "w") as json_file:
            json_file.write(model_json)

        about = {}
        about['Creation Time'] = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())
        about['Train Files'] = bags
        about['Total Train Files'] = bag_count
        about['Total Sequences (Sensor)'] = X_train.shape[0]
        about['Sequence Length (Sensor)'] = time_step
        with open(get_info_path(model_path), 'w') as outfile:
            json.dump(about, outfile)

        # serialize weights to HDF5
        model.save_weights(get_trained_model_path(model_path))
        print("Saved model to disk")

        X_train_pred = model.predict(X_train)  # Generation of Decoder output
        X_train_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)  # Reconstruction error for each feature in each sequence

        mu = np.mean(X_train_loss, axis=0)  # Mean Reconstruction error for each feature across all sequences
        sigma = np.std(X_train_loss, axis=0)

        np.savetxt(get_mean_loss_path(model_path), mu)
        np.savetxt(get_sigma_loss_path(model_path), sigma)

    K.clear_session()

    #cuda.select_device(0)
    #cuda.close()
    #if __name__ == "__main__":
    plt.plot(history.history['loss'])
    plt.xlabel('Number of epochs')
    plt.ylabel('Train Loss')
    #plt.plot(history.history['val_loss'])
    plt.legend()
    plt.show()

    return model_path, about
#%%


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, input_data, batch=32, noise=False, shuffle=False):
        'Initialization'
        self.data = input_data #ndarray(664,5,13)
        self.batch_size = batch
        self.shuffle = shuffle
        self.on_epoch_end()
        self.noise = noise

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        current_data = self.data[index*self.batch_size:(index+1)*self.batch_size]
        if self.noise:
            #noise = tf.random.normal(shape=tf.shape(current_data), mean=0.0, stddev=0.1, dtype=tf.float32)
            current_data = current_data + np.random.normal(0, 0.1, current_data.shape)

        return current_data, current_data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.data)


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


if __name__ == "__main__":
    train_sensor_values()
