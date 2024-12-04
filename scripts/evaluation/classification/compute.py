import sys
sys.path.append("scripts/train/")
sys.path.append("scripts/")

import random
import tensorflow as tf
from tensorflow.keras.models import load_model,Model
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tools.dataloader import DataLoader
data_loader = DataLoader()

classification_model = load_model('models/movenet/classification/stroke_classification_357.keras')
data_loader.load_data_movenet("test") # time_step_size always 30

swings = ["backhand", "forehand", "neutral", "serve"]
pro_players = ["kermnovic", "ruu004", "ruud002", "ruud006", "sinner001", "sinnerbackhand001"]
amateur_players = ["andre_yuki", "bawt_jason_wee"]

def compute_f1(players, players_type):
    y_true = []
    y_pred = []
    total_preds = 0
    for swing, player in list(itertools.product(["backhand", "forehand", "neutral", "serve"], players)):
        for poses in data_loader.poses_per_swing_per_player[player][swing]:
            X = []
            X.append(poses)
            input = np.array(X)
            output = classification_model.predict(input, verbose=0)
            predicted_class = np.argmax(output, axis=1)
            #print("Classifier output:", swings[predicted_class[0]], "with a score of", np.max(output) * 100, "%", "for", swing)
            total_preds += 1
            y_true.append(swings.index(swing))
            y_pred.append(predicted_class[0])
    print("Total predictions:", total_preds)
    f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=swings, yticklabels=swings)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix. {} F1 Score: {:.2f}, Accuracy: {:.2f}%'.format(players_type, f1, accuracy * 100))
    plt.savefig(f'confusion_matrix_{players_type}.png', dpi=300)  # Save as PNG with high resolution
    plt.close()  # Close the plot to free up memory
    return f1, accuracy

# we use only f1 score as the dataset is imbalanced
print("F1 Score and accuracy for pro players:", compute_f1(pro_players, "pro-players"))
print("F1 Score and accuracy for amateur players:", compute_f1(amateur_players, "amateur-players"))
print("F1 Score and accuracy for all players:", compute_f1(pro_players + amateur_players, "all-players"))

