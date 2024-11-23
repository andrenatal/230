import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
import json
from tensorflow.keras.layers import LSTM,Dense,GRU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from keras.utils import plot_model
from keras.losses import CosineSimilarity
import numpy as np
import matplotlib.pyplot as plt


autoencoder_model = load_model('models/autoencoder/backhand.json.autoencoder.keras')
mean_X = np.load('models/means/mean_X_backhand.json.npy')
std_X = np.load('models/means/std_X_backhand.json.npy')

# Function to normalize input data
def normalize_input(input_data, mean_X, std_X):
    return (input_data - mean_X) / std_X

print(autoencoder_model.summary())

def read_all_jsons(folder_path):
    poses_per_swing = {}
    for filename in os.listdir(folder_path):
        json_data = []
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                json_data.append(data)
                poses_per_swing[filename] = json_data
    return poses_per_swing
folder_path = "/mnt/crucial1tb_ssd/cs230/data/dataset/Tennis Player Actions Dataset for Human Pose Estimation/annotations"
poses_per_swing = read_all_jsons(folder_path)

item_to_start = 9
time_step_size = 4
inputs = []
for i in range(item_to_start, item_to_start + time_step_size, 1):
    inputs.append(np.array(poses_per_swing["backhand.json"][0]["annotations"][i]["keypoints"]))
inputs = np.array(inputs)
input_classifier = normalize_input(inputs, mean_X, std_X)
input_classifier = input_classifier.reshape(1, time_step_size, 54)
test_output = autoencoder_model.predict(input_classifier, verbose=0)
print("Autoencoder output:", test_output.shape)