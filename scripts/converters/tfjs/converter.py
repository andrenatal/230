import tensorflowjs as tfjs
from tensorflow.keras.models import load_model,Model
import os
import sys
sys.path.append("/media/4tbdrive/engines/cs230/scripts/train/")
sys.path.append("/media/4tbdrive/engines/cs230/scripts/")


classification_model = load_model('/media/4tbdrive/engines/cs230/models/movenet/classification/classification_351_withnadal.keras')
tfjs.converters.save_keras_model(classification_model, "/media/4tbdrive/engines/cs230/models/tfjs/classification/")

# The conversion of the encoder model using keras 3 is not supported by tensorflowjs
# An artifact is produced, but the model can't be loaded in the browser
# After applying the changes suggested here: https://github.com/tensorflow/tfjs/issues/8328#issuecomment-2255097136
# and here https://discuss.ai.google.dev/t/corrupted-configuration-and-batch-input-shape-loading-pre-trained-layers-model-in-tensorflow-js/24454
# the model loads but hangs the browser
path_models = "/media/4tbdrive/engines/cs230/models/movenet/autoencoder/"
for filename in os.listdir(path_models):
    if filename.endswith('.keras'):
        model = load_model(os.path.join(path_models, filename))
        tfjs.converters.save_keras_model(model, "/media/4tbdrive/engines/cs230/models/tfjs/autoencoder/"+filename.replace('.keras','') + "/")
