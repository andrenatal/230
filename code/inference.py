import tensorflow as tf
from tensorflow.keras.models import load_model,Model
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

mean_X = np.load('models/means/mean_X.npy')
std_X = np.load('models/means/std_X.npy')
classification_model = load_model('models/classification/stroke_classification.keras')

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


# Function to normalize input data
def normalize_input(input_data, mean_X, std_X):
    return (input_data - mean_X) / std_X


swings = ["backhand.json", "forehand.json", "ready_position.json", "serve.json"]

for test_swing in swings:
    print("\n\n\n")
    print("Testing Swing:", test_swing)
    item_to_start = 9
    time_step_size = 4
    inputs = []
    for i in range(item_to_start, item_to_start + time_step_size, 1):
        inputs.append(np.array(poses_per_swing[test_swing][0]["annotations"][i]["keypoints"]))
    inputs = np.array(inputs)
    input_classifier = normalize_input(inputs, mean_X, std_X)
    input_classifier = input_classifier.reshape(1, time_step_size, 54)

    # first test the similarity of the input to the training data with the classification model
    test_output = classification_model.predict(input_classifier, verbose=0)
    predicted_class = np.argmax(test_output, axis=1)
    print("Classifier output:", swings[predicted_class[0]])
    print("Classifier output probabilities:", test_output)

     # test similarities
    scores = []
    for swing in swings:
        similarity_model = Model(inputs=classification_model.inputs, outputs=classification_model.layers[-2].output)
        #print(similarity_model.summary())
        #print("Input shape:", input_classifier.shape)
        input_embeddings = similarity_model.predict(input_classifier, verbose=0)
        training_embeddings = np.load('models/embeddings/' + swing + '.classification.embeddings.npy')
        similarity_scores = cosine_similarity(input_embeddings, training_embeddings)
        print("Average of the scores:", np.mean(similarity_scores), "for", swing, "using classifier.")
        scores.append(np.mean(similarity_scores))

    print("Similarity Winner:", swings[np.argmax(scores)], "with a score of:", scores[np.argmax(scores)], "using classifier.")


    # test the similarity of the input to the training data with the encoder model
    scores = []
    for swing in swings:
        encoder_model = load_model('models/encoder/'+test_swing+'.encoder.keras')
        input_embeddings = encoder_model.predict(input_classifier, verbose=0)
        training_embeddings = np.load('models/embeddings/' + swing + '.encoder.embeddings.npy')
        similarity_scores = cosine_similarity(input_embeddings, training_embeddings)
        print("Average of the scores:", np.mean(similarity_scores), "for", swing, "using encoder.")
        scores.append(np.mean(similarity_scores))

    print("Similarity Winner:", swings[np.argmax(scores)], "with a score of:", scores[np.argmax(scores)], "using encoder.")