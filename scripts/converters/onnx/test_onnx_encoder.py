import onnxruntime as ort
import numpy as np
from tensorflow.keras.models import load_model,Model
from sklearn.metrics.pairwise import cosine_similarity

file_path = ["/media/4tbdrive/engines/cs230/app/backhand_001.csv", "/media/4tbdrive/engines/cs230/app/forehand_010.csv", "/media/4tbdrive/engines/cs230/app/serve_001.csv"]

sess = ort.InferenceSession("/media/4tbdrive/engines/cs230/models/onnx/movenetbackhandalcaraz_encoder/model.onnx", providers=["CUDAExecutionProvider"])
encoder_model = load_model(f'/media/4tbdrive/engines/cs230/models/movenet/encoder/movenetbackhandalcaraz.encoder.keras')
training_embeddings = np.load(f'/media/4tbdrive/engines/cs230/models/movenet/embeddings/movenetbackhandalcaraz.encoder.embeddings.npy')

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
    # test the similarity of the input to the training data with the encoder model
    #scores = []
    input_embeddings = encoder_model.predict(input, verbose=0)
    similarity_scores = cosine_similarity(input_embeddings, training_embeddings)
    print("Similarity: Average of the scores:", np.mean(similarity_scores), "for", "backhandalcaraz", "using keras encoder.")

    results_ort = sess.run(None, {"input": input.astype(np.float32)})
    similarity_scores = cosine_similarity(results_ort[0], training_embeddings)
    print("Similarity: Average of the scores:", np.mean(similarity_scores), "for", "backhandalcaraz", "using onnx encoder.")

