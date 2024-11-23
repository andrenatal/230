import sys
sys.path.append("/media/4tbdrive/engines/cs230/scripts/train/")
sys.path.append("/media/4tbdrive/engines/cs230/scripts/")

import random
import tensorflow as tf
from tensorflow.keras.models import load_model,Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools

from tools.dataloader import DataLoader
data_loader = DataLoader()

classification_model = load_model('models/movenet/classification/stroke_classification_351_withnadal.keras')
data_loader.load_data_movenet() # time_step_size always 30

swings = ["backhand", "forehand", "neutral", "serve"]
players = ["roland"]
reference_player = "nadal"

for swing, player in list(itertools.product(["backhand", "forehand"], players)):
    print("\n")
    print("Retrieving a random", swing, "from", player)

    X = []
    for idx in range(0,len(data_loader.poses_per_swing_per_player[player][swing])):
        X.append(data_loader.poses_per_swing_per_player[player][swing][idx])
    if len(X) == 0:
        print("No", swing, "found for", player)
        continue
    input = np.array(random.choice(X))
    input = input.reshape(1, input.shape[0],input.shape[1])

    # classify the input
    test_output = classification_model.predict(input, verbose=0)
    predicted_class = np.argmax(test_output, axis=1)
    print("Classifier output:", swings[predicted_class[0]], "with a score of", np.max(test_output) * 100, "%")

    if swings[predicted_class[0]] == "neutral":
        print("Neutral swing, skipping similarity check.")
        continue

    # test the similarity of the input to the training data with the encoder model
    scores = []
    encoder_model = load_model(f'models/movenet/encoder/movenet{swings[predicted_class[0]]}{reference_player}.encoder.keras')
    input_embeddings = encoder_model.predict(input, verbose=0)
    training_embeddings = np.load(f'/media/4tbdrive/engines/cs230/models/movenet/embeddings/movenet{swings[predicted_class[0]]}{reference_player}.encoder.embeddings.npy')
    similarity_scores = cosine_similarity(input_embeddings, training_embeddings)
    print("Similarity: Average of the scores:", np.mean(similarity_scores), "for", swings[predicted_class[0]], "with", reference_player, "using encoder.")