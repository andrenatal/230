import sys
sys.path.append("/media/4tbdrive/engines/cs230/scripts/train/")
sys.path.append("/media/4tbdrive/engines/cs230/scripts/")

import random
import tensorflow as tf
from tensorflow.keras.models import load_model,Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools

classification_model = load_model('models/movenet/classification/stroke_classification_351_withnadal.keras')

print("\n")
file_path = ["/media/4tbdrive/engines/cs230/app/backhand_001.csv", "/media/4tbdrive/engines/cs230/app/forehand_010.csv", "/media/4tbdrive/engines/cs230/app/serve_001.csv"]

for file in file_path:
    print("Testing file:", file)
    X = []
    with open(file, 'r') as file:
        temp_X = []
        for line in file:
            if line.startswith("nose_y"):
                continue
            data = line.split(",")
            data = [float(x) for x in data]
            temp_X.append(data)
            for element in data[:-1]:
                if not isinstance(element, float):
                    raise TypeError(f"List contains non-float data: {type(element)}")
        X.append(temp_X)

    input = np.array(X)
    #input = input.reshape(1, input.shape[0],input.shape[1])

    # classify the input
    test_output = classification_model.predict(input, verbose=0)
    predicted_class = np.argmax(test_output, axis=1)
    print("Output:", test_output)
    print("Classifier output:",predicted_class, "with a score of", np.max(test_output) * 100, "%")


    # test the similarity of the input to the training data with the encoder model
    #scores = []
    #encoder_model = load_model(f'models/movenet/encoder/movenet{swings[predicted_class[0]]}{reference_player}.encoder.keras')
    #input_embeddings = encoder_model.predict(input, verbose=0)
    #training_embeddings = np.load(f'/media/4tbdrive/engines/cs230/models/movenet/embeddings/movenet{swings[predicted_class[0]]}{reference_player}.encoder.embeddings.npy')
    #similarity_scores = cosine_similarity(input_embeddings, training_embeddings)
    #print("Similarity: Average of the scores:", np.mean(similarity_scores), "for", swings[predicted_class[0]], "with", reference_player, "using encoder.")