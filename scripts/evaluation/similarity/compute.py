import sys
sys.path.append("./scripts/train/")
sys.path.append("./scripts/")

import random
import tensorflow as tf
from tensorflow.keras.models import load_model,Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from tools.dataloader import DataLoader
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

classification_model = load_model('models/movenet/classification/stroke_classification_357.keras')
data_loader_test = DataLoader()
data_loader_test.load_data_movenet("test") # time_step_size always 30

swings = ["backhand", "forehand", "neutral", "serve"]
pro_players = ["kermnovic", "ruu004", "ruud002", "ruud006", "sinner001", "sinnerbackhand001"]
amateur_players = ["andre_yuki", "bawt_jason_wee"]

reference_players = ["alcaraz003", "federer", "monfils002", "djokovic001", "sinne005", "sinner002"]

def compute_similarity(reference_player, players, players_type):
    scores = []
    for swing, player in list(itertools.product(["backhand", "forehand", "serve"], players)):
        X = []
        for idx in range(0,len(data_loader_test.poses_per_swing_per_player[player][swing])):
            X.append(data_loader_test.poses_per_swing_per_player[player][swing][idx])
        if len(X) == 0:
            print("No", swing, "found for", player)
            continue
        input = np.array(random.choice(X))
        input = input.reshape(1, input.shape[0],input.shape[1])

        # test the similarity of the input to the training data with the encoder model
        if os.path.exists(f'models/movenet/encoder/{swing}{reference_player}.encoder.keras'):
            encoder_model = load_model(f'models/movenet/encoder/{swing}{reference_player}.encoder.keras')
            input_embeddings = encoder_model.predict(input, verbose=0)
            training_embeddings = np.load(f'./models/movenet/embeddings/{swing}{reference_player}.encoder.embeddings.npy')
            similarity_scores = cosine_similarity(input_embeddings, training_embeddings)
            score = np.mean(similarity_scores)
            scores.append(score)
            print("Similarity: Average of the scores:", score, "for", player, swing, "with", reference_player, "using encoder.")

    scores_array = np.array(scores)
    # Plotting the histogram of the scores
    plt.figure(figsize=(10, 6))
    sns.histplot(scores_array, bins=30, kde=True, color='blue')

    # Calculate mean and standard deviation
    mean = np.mean(scores_array)
    std_dev = np.std(scores_array)
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)

    # Plot the normal distribution curve
    x = np.linspace(min(scores_array), max(scores_array), 100)
    plt.plot(x, norm.pdf(x, mean, std_dev), color='red', label='Normal Distribution')

    # Add mean and standard deviation lines
    plt.axvline(mean, color='green', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
    plt.axvline(mean + std_dev, color='orange', linestyle='dashed', linewidth=1, label=f'+1 Std Dev: {mean + std_dev:.2f}')
    plt.axvline(mean - std_dev, color='orange', linestyle='dashed', linewidth=1, label=f'-1 Std Dev: {mean - std_dev:.2f}')
    plt.axvline(min_score, color='purple', linestyle='dashed', linewidth=1, label=f'Min: {min_score:.2f}')
    plt.axvline(max_score, color='brown', linestyle='dashed', linewidth=1, label=f'Max: {max_score:.2f}')

    plt.title('Distribution of Similarity Scores for ' + players_type + ' and ' + reference_player)
    plt.xlabel('Scores')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'distribution_{players_type}_and_{reference_player}.png', dpi=300)
    plt.show()

for reference_player in reference_players:
    compute_similarity(reference_player, pro_players, "pro-players")
    compute_similarity(reference_player, amateur_players, "amateur-players")

# we want to compute the similiarity between the same player but between
# the test set and the training embeddings for reference and analysis
for reference_player in ["sinne005", "sinner002"]:
    compute_similarity(reference_player, ["sinner001", "sinnerbackhand001"], "Sinner")
