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

X = []
Y = []
time_step_size = 4
epochs = 1000
for key in poses_per_swing:
    for idx in range(0,len(poses_per_swing[key][0]["annotations"]),time_step_size):
        temp_X = []
        for i in range(idx,idx+time_step_size):
            temp_X.append(poses_per_swing[key][0]["annotations"][i]["keypoints"])
        X.append(temp_X)
        if poses_per_swing[key][0]["annotations"][idx]["category_id"] == 5: # backhand
            Y.append(0)
        if poses_per_swing[key][0]["annotations"][idx]["category_id"] == 6: # forehand
            Y.append(1)
        if poses_per_swing[key][0]["annotations"][idx]["category_id"] == 7: # ready position
            Y.append(2)
        if poses_per_swing[key][0]["annotations"][idx]["category_id"] == 8: # serve
            Y.append(3)
        #Y.append(poses_per_swing[key][0]["annotations"][idx]["category_id"])
        #print(annotation)

X = np.array(X)
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)
np.save("models/means/mean_X.npy", mean_X)
np.save("models/means/std_X.npy", std_X)

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y = np.array(Y)

print("Checking for NaN values in X:", np.isnan(X).any())
print("Checking for infinite values in X:", np.isinf(X).any())

def get_model():
    model = Sequential()
#    model.add(GRU(units=24, dropout=0.1, input_shape=(time_step_size, 54)))
    model.add(LSTM(24, input_shape=(time_step_size,54), dropout=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0), metrics=['accuracy'])
    return model

num_classes = len(np.unique(Y))

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("\n End epoch {} of training; got log keys: {} \n".format(epoch, keys))

tf.keras.backend.clear_session()
model = get_model()
model.summary()

log_dir = "logs/classification/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
_ = model.fit(X, Y, epochs=epochs, validation_split=0.1, batch_size=2, verbose=1, callbacks=[CustomCallback(), tensorboard_callback, early_stopping_callback])
model.save("models/classification/stroke_classification.keras")


test_output = model.predict(X[0:1], verbose=1)  # Use a batch of inputs for prediction
predicted_class = np.argmax(test_output, axis=1)
print(test_output)
print(predicted_class)