import sys
sys.path.append("/media/4tbdrive/engines/cs230/scripts/train/")
sys.path.append("/media/4tbdrive/engines/cs230/scripts/")

import random
import tensorflow as tf
from tensorflow.keras.models import load_model,Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import tf2onnx
import onnx


#classification_model = load_model('models/movenet/classification/stroke_classification_351_withnadal.keras')
#classification_model.output_names = ['output']
#input_signature = [tf.TensorSpec([None, 30, 26], tf.float32, name='input')]
#onnx_model, _ = tf2onnx.convert.from_keras(classification_model, input_signature, opset=13)
#onnx.save(onnx_model, "/media/4tbdrive/engines/cs230/models/onnx/stroke_classification_351_withnadal/model.onnx")

encoder_model = load_model('models/movenet/encoder/movenetbackhandalcaraz.encoder.keras')
encoder_model.output_names = ['output']
input_signature = [tf.TensorSpec([None, 30, 26], tf.float32, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(encoder_model, input_signature, opset=13)
onnx.save(onnx_model, "/media/4tbdrive/engines/cs230/models/onnx/movenetbackhandalcaraz_encoder/model.onnx")
