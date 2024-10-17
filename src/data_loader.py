# coding=utf-8
import pandas as pd
import numpy as np
import math
import glob
from keras.utils import to_categorical
import pathlib
import os
from sklearn.decomposition import PCA
from config import cfig

# code credit from 
def MaxMinNormalize(x):
    x_max = np.max(x)
    x_min = np.min(x)
    for i in range(len(x)):
        x[i][0] = (x[i][0] - x_min)/(x_max - x_min)
    return x

def get_feature(data):
    feature = np.zeros((1608,1), dtype=np.float64)
    j = 2
    for i in range(len(data)-2):
        feature[i] = float(data.array[j])
        j += 1
    return MaxMinNormalize(feature)

def get_data(file_path):
    data_label = []
    data_x = []
    df = pd.read_excel(file_path, sheet_name='Sheet5')
    treat_label = []

    for index,row in enumerate(df.items()):
        if index >= 2:
            if math.isnan(float(row[1][2])) == True:
                continue
            value = row[1][0]
            if value == 'BCC':
                label = 0
            elif value == 'NORMAL':
                label = 1
            elif value == 'SCC':
            # else:
                label = 2
            else:
                print(label)
            data_label.append(label)
            feature = get_feature(row[1])[:, 0]
            data_x.append(feature)

            treat_label.append(row[1][1])


    data_x = np.array(data_x)
    data_label = np.array(data_label)
    return data_x, data_label, np.array(treat_label)

def pca(X):
    pca = PCA(n_components=100)
    pca.fit(X)
    return pca.transform(X)

def moving_average(data, window_size=5):
    mov_ave = []
    for x in data:
        x = np.convolve(x, np.ones(window_size)/window_size, mode='same')
        mov_ave.append(x)
    return np.array(mov_ave)


def generate_faked_data():
    x = np.random.rand(149, 1608)
    y = np.random.choice([0, 1, 2], size=(149))
    
    return x, y, ''

def load_data(dtype='origin'):
    
    file_path = '../data/data_feature.xlsx'
    if os.path.exists(file_path) == True:
        x, y, _ = get_data(file_path)
    else:
        x, y, _ = generate_faked_data()
    
    if dtype == 'origin':
        return x, y, _
    elif dtype == 'pca':
        return pca(x), y, _
    
    return x, y, _


def norm_func(x, a=0, b=1):
    return ((b - a) * (x - min(x))) / (max(x) - min(x)) + a

def normalize(x, y=None):
    x = np.apply_along_axis(norm_func, axis=1, arr=x)
    return x

def get_cell_data():
    data={}
    abs_path = pathlib.Path(__file__).parent.parent.resolve()
    for file in glob.glob(f"{abs_path}/data/cells-raman-spectra/dataset_i/**/*.csv"):
        path = file.split(os.path.sep)
        label = path[-2]
        kind = path[-1][:-4]
        if label not in data.keys():
            data[label] = {}
        data[label][kind] = normalize(pd.read_csv(file).values)
    return data


def extend_smooth_wave(original_array):
    extended_array = []
    for i in range(len(original_array)):
        if len(extended_array) >= 1608:
            break
        extended_array.append(original_array[i])
        if i < len(original_array) - 1:
            diff = (original_array[i + 1] - original_array[i])
            extended_array.append(original_array[i] + (diff / 3))
            extended_array.append(original_array[i] + (2 * diff / 3))
    return extended_array

def extend_hard_wave(original_array):
    extended_array = []

    for i in range(len(original_array)):
        if len(extended_array) >= 1608:
            break  
        extended_array.append(original_array[i])
        if i < len(original_array) - 1:
            extended_array.append(original_array[i])
            extended_array.append(original_array[i + 1])
    return extended_array

def get_cell_data_with_keys(keys, kinds, batch = 49, dtype='origin'):
    data = get_cell_data()
    samples = []
    labels = []
    y_i = 0
    for key in keys:
        for i in range(batch):
            tmp_x = []
            for kind in kinds:
                tmp_x = data[key][kind][i][0 :1608].tolist()
                samples.append(tmp_x)
                labels.append(y_i)
        y_i += 1
    
    samples = np.array(samples)
    
    if dtype == 'pca':
        samples = pca(samples)
        
    return samples, to_categorical(labels, num_classes=len(keys)), np.array(labels)