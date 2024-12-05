# AI Tennis Coach


 ### See this code in action live on https://brain.tennis/ 

![Camila Giorgi Video](cgiorgi.png)

## Description

This repository contains the proceedings of the final project developed during my participation in [CS230](https://cs230.stanford.edu/) at Stanford University. 

It was proposed that a deep learning-based computer vision system be developed to analyze tennis matches in real time and classify strokes and similarities with reference players using models trained on pose estimation from videos of professional players.

The ultimate motivation is to have an extensible AI platform that could help build performance improvements, technique enhancement, and injury prevention that can be used as an additional tool to coaching.

Given the low dimensionality and high-level representation of poses and body joints, the same mechanics and systems can be extended to other sports where biomechanics and body movement are key to achieving high performance. 

Recreational players can also benefit from it. By having their footage analyzed by the system, they can potentially learn their handicaps but also prevent injuries that arise from bad swings.

## Data

Youtube footage of professional players, as well as footage of myself playing recreationally, was extracted and annotated. A custom tool for annotating tennis data was built and is incorporated in this repo. 

## Architecture

### Features

Pose estimations are extracted from real-time videos of tennis players in action using [Movenet](https://www.tensorflow.org/hub/tutorials/movenet). Depending on the player's ability, a tennis swing could last 1-1.5 seconds, so we use 30 poses (30fps) and 26 joints from Movenet, excluding face joints, as inputs. I also experimented with OpenPose but decided on Movenet because it's claimed to perform better for sports and stands out on the web in terms of performance.

### Classification and Similarity

Two RNN-based architectures were utilized to classify the shots and compute their similarity against the reference players: one unidirectional LSTM for shot classification and an autoencoder whose decoder portion is removed and only the encoder is used to pre-compute embeddings from reference players, which are then compared with the embeddings from the player being inferred using cosine similarity.

### Application

After being trained using Keras, the models were converted to Tensorflow.js and ONNX to be utilized in a Web Application. The lack of valuable tools like scikit-learn and Numpy in Javascript required the rewrite of some equations, like L2 normalization and cosine similarity, from scratch with the help of Math.js and npyjs.js. In some cases, a performance of +100 fps is achieved, and real-time analysis of the videos can be performed, corroborating the viability and high performance of deep learning on the Edge, especially in web environments.

### Models and data
If you are interested on knowing more about the data and the models, you can reach out directly to me: https://andrenatal.com/ 


### Codebase

> ./app

- This is the placeholder for the web app container packaging. All models, videos, and web apps are placed here to be containerized and deployed.

> ./backend_label

- Backend for the annotation tool. Poses are uploaded to this backend and stored as they are annotated.

> ./scripts/converters

- Tools to convert and test the converted models from Keras to ONNX and Tensorflow.js

> ./scripts/evaluation

- The scripts and charts used to evaluate the similarity and classification models.

> ./scripts/inference

- The Python scripts used to perform inference on the models. 

> ./train

- The scripts used to train the models. 

> ./tfjs-models

- A submodule containing Movenet tooling. The demo application on https://github.com/andrenatal/tfjs-models/tree/master/pose-detection was heavily customized to produce the annotation and inference web app. 

### Acknowledgments

I want to thank my TA, Bassem Akoush, and instructors, Andrew Ng and Kian Katanforoosh, for the great class experience.

--------------
 ### See this code in action live on https://brain.tennis/ 
