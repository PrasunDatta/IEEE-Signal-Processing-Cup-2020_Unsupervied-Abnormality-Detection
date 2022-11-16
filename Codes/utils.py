"""
	This file contains a variety of common functions
"""
from os import listdir
import os
from os.path import isfile, join
import time

def create_directory_if_needed(dir_path):
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)


def get_files_in_directory(dir_path, full_path = True):
	only_files = [os.path.join(dir_path, f) if full_path else f for f in listdir(dir_path) if isfile(join(dir_path, f))]
	return only_files


def get_four_digit_string(input_string, digits=4):
	formatter = '%0' + str(digits) + 'd'
	return formatter % int(input_string)


def rename_files_in_directory(dir_path, prefix=''):
	old_names = get_files_in_directory(dir_path, full_path=False)
	for old_name in old_names:
		old_name_without_ext, ext = old_name.split('.')
		new_name = prefix + get_four_digit_string(old_name_without_ext) + '.' + ext
		os.rename(os.path.join(dir_path, old_name), os.path.join(dir_path, new_name))


def get_file_name_without_ext(file_full_path):
	return file_full_path.split(os.path.sep)[-1].split('.')[0]


def get_extraction_directory(filename):
	return os.path.join('data', filename)


def get_images_directory(filename):
	return os.path.join(get_extraction_directory(filename), 'images')


def get_sensor_csv_path(filename):
	return os.path.join(get_extraction_directory(filename), 'sensor.csv')


def get_sensor_abnormality_path(filename):
	return os.path.join(get_extraction_directory(filename), 'sensor_abnormality.csv')


def get_bag_info_path(filename):
	return os.path.join(get_extraction_directory(filename), 'info.json')


def get_bag_image_sample(filename):
	return os.path.join(get_extraction_directory(filename), 'images', '1.jpg')


def get_new_model_path():
	time_str = time.strftime("%Y%m%d-%H%M%S")
	create_directory_if_needed(os.path.join('models', time_str))
	return os.path.join('models', time_str)


def get_default_model_path():
	return os.path.join('models', 'A1-A5')


def get_scale_params_path(model_path):
	return os.path.join(model_path, 'Scale_Parameters.csv')


def get_trained_model_path(model_path):
	return os.path.join(model_path, 'model.h5')


def get_model_info_path(model_path):
	return os.path.join(model_path, 'model.json')


def get_info_path(model_path):
	return os.path.join(model_path, 'about_model.json')


def get_mean_loss_path(model_path):
	return os.path.join(model_path, 'Mean_losses.csv')


def get_sigma_loss_path(model_path):
	return os.path.join(model_path, 'Sigma_losses.csv')


def rename_dataset_images(dir_path, dataset_no):
	rename_files_in_directory(dir_path, get_four_digit_string(dataset_no, digits=2) + '_')




