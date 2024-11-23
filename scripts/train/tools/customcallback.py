import tensorflow as tf

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 != 0:
            return
        keys = list(logs.keys())
        print("===================")
        print("Epoch {}:".format(epoch))
        for each_key in keys:
            print(f"{each_key}: {logs[each_key]}")


    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("===================")
        print("Train end:")
        for each_key in keys:
            print(f"{each_key}: {logs[each_key]}")
