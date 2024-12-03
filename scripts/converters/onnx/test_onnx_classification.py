import onnxruntime as ort
import numpy as np
from tensorflow.keras.models import load_model,Model

# The conversion of the classifcation model to onnx is not
# working due to issues:
# https://github.com/onnx/tensorflow-onnx/issues/2372
# and
# https://github.com/onnx/tensorflow-onnx/issues/2359

# Change shapes and types to match model
input1 = np.zeros((1, 30, 26), np.float32)


sess = ort.InferenceSession("/media/4tbdrive/engines/cs230/models/onnx/stroke_classification_351_withnadal/model.onnx", providers=["CUDAExecutionProvider"])

results_ort = sess.run(["output"], {"input": input1})

import tensorflow as tf

classification_model = load_model('models/movenet/classification/stroke_classification_351_withnadal.keras')
test_output = classification_model.predict(input, verbose=0)

for ort_res, tf_res in zip(results_ort, test_output):
    np.testing.assert_allclose(ort_res, tf_res, rtol=1e-5, atol=1e-5)

print("Results match")

