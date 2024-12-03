import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM,Dense,GRU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from keras.utils import plot_model
from keras.losses import CosineSimilarity

import numpy as np
import datetime
import itertools

from tools.dataloader import DataLoader
from tools.customcallback import CustomCallback

time_step_size = 30
epochs = 1000
data_loader = DataLoader()
pose_type = "movenet" # "movenet" or "openpose"
recurrent_units = 26
dropout = 0.02
lstm_dropout = 0.01
dense_units = 28
batch_size = 2
validation_split = 0.2
learning_rate = 0.0001

if pose_type == "movenet":
    data_loader.load_data_movenet("train") # time_step_size always 30
else:
    data_loader.load_data_openpose(time_step_size)

def get_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    encoded = LSTM(recurrent_units, activation='relu', return_sequences=False, dropout=lstm_dropout)(inputs)
    encoded = Dropout(dropout)(encoded)
    encoded = Dense(input_shape[1], activation='relu')(encoded)
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = LSTM(recurrent_units, dropout=dropout,  activation='relu', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(input_shape[1], activation="sigmoid"))(decoded)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0), loss="mse", metrics=['accuracy'])
    return autoencoder

swings = ["backhand", "forehand", "serve"]
players = ["sinner002", "sinne005"]
for swing, player in list(itertools.product(swings, players)):
    X = []
    if len(data_loader.poses_per_swing_per_player[player][swing]) == 0: continue;
    for idx in range(0,len(data_loader.poses_per_swing_per_player[player][swing])):
        X.append(data_loader.poses_per_swing_per_player[player][swing][idx])

    X = np.array(X)

    """ # Train the autoencoder model """
    print("Checking for NaN values in X:", np.isnan(X).any())
    print("Checking for infinite values in X:", np.isinf(X).any())
    print("Training autoencoder model for", player, swing)

    log_dir = f"logs/movenet/autoencoder/{swing}/{player}/" + swing + "/"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=False)

    input_shape = (X.shape[1], X.shape[2])
    autoencoder = get_autoencoder(input_shape)
    #print("Autoencoder model summary:")
    #autoencoder.summary()
    history = autoencoder.fit(X, X, verbose=0, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[CustomCallback(), tensorboard_callback, early_stopping_callback])
    autoencoder.save(f"models/movenet/autoencoder/{swing}{player}.autoencoder.keras")

    #print("Encoder model summary:")
    encoder = Model(autoencoder.input, autoencoder.layers[3].output)
    #encoder.summary()
    encoder.save(f"models/movenet/encoder/{swing}{player}.encoder.keras")
    # Use the layers of the encoder model to generate the embeddings for this player and swing
    embeddings = encoder.predict(X)
    np.save(f"models/movenet/embeddings/{swing}{player}.encoder.embeddings.npy", embeddings)