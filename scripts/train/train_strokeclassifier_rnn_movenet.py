import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import LSTM,Dense,GRU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
import datetime
import sys

from tools.dataloader import DataLoader
from tools.customcallback import CustomCallback

time_step_size = 30
epochs = 1000
data_loader = DataLoader(time_step_size)
pose_type = "movenet" # "movenet" or "openpose"
recurrent_units = 26
dropout = 0.02
dense_units = 28
lstm_dropout = 0.01

if pose_type == "movenet":
    data_loader.load_data_movenet() # time_step_size always 30
else:
    data_loader.load_data_openpose(time_step_size)

def get_model():
    model = Sequential()
#    model.add(GRU(units=26, dropout=0.01, input_shape=(time_step_size, 26)))
    model.add(LSTM(recurrent_units, input_shape=(time_step_size,26), dropout=lstm_dropout))
    model.add(Dropout(dropout))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = len(np.unique(data_loader.Y))
model = get_model()
model.summary()

log_dir = f"logs/{pose_type}/classification/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=False)
history = model.fit(data_loader.X, data_loader.Y, epochs=epochs, validation_split=0.25, batch_size=32, verbose=0, callbacks=[CustomCallback(), tensorboard_callback, early_stopping_callback])
model.save(f"models/{pose_type}/classification/stroke_classification_{history.epoch[-1]}.keras")

test_output = model.predict(data_loader.X[0:1], verbose=1)  # Use a batch of inputs for prediction
predicted_class = np.argmax(test_output, axis=1)
print(test_output)
print(predicted_class)