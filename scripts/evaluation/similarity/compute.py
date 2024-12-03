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
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
            training_embeddings = np.load(f'/media/4tbdrive/engines/cs230/models/movenet/embeddings/{swing}{reference_player}.encoder.embeddings.npy')
            similarity_scores = cosine_similarity(input_embeddings, training_embeddings)
            score = np.mean(similarity_scores)
            scores.append(score)
            print("Similarity: Average of the scores:", score, "for", player, swing, "with", reference_player, "using encoder.")

    scores_array = np.array(scores)
    percentile_bins = np.percentile(scores_array, [0, 25, 50, 75, 100])
    percentile_labels = ['0th Percentile', '25th Percentile', '50th Percentile', '75th Percentile', '100th Percentile']
    # Bin the scores based on percentiles
    binned_scores = np.digitize(scores_array, percentile_bins)
    # Count the frequency of scores in each bin
    count_scores_per_bin = [np.sum(binned_scores == i) for i in range(1, len(percentile_bins)+1)]

    # Plotting the frequency of scores in each percentile bin
    plt.figure(figsize=(10, 6))
    sns.barplot(x=percentile_labels, y=count_scores_per_bin)  # Skip the first bin as it is below the 0th percentile
    plt.title('Frequency of Similarity Scores in Each Percentile for ' + players_type + ' and ' + reference_player)
    plt.xlabel('Percentile Range')
    plt.ylabel('Frequency')
    plt.ylim(0, max(count_scores_per_bin) + 1)  # Adjust y-axis limit for better visualization
    plt.savefig(f'percentile_frequency_{players_type}_and_{reference_player}.png', dpi=300)

for reference_player in reference_players:
    compute_similarity(reference_player, pro_players, "pro-players")
    compute_similarity(reference_player, amateur_players, "amateur-players")

# we want to compute the similiarity between the same player but between
# the test set and the training embeddings for reference and analysis
for reference_player in ["sinne005", "sinner002"]:
    compute_similarity(reference_player, ["sinner001", "sinnerbackhand001"], "Sinner")
