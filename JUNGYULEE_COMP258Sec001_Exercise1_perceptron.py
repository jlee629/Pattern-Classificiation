# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 22:35:37 2024

Jungyu Lee
301236221

Pattern Classfication App
"""

import pandas as pd
import numpy as np
from JUNGYULEE_COMP258Sec001_Exercise1_utils import (
    introduce_noise, 
    count_noise,
    predict_and_evaluate 
)

class PerceptronModel:
    def __init__(self, no_of_features):
        self.weights = np.zeros(no_of_features)
        self.bias = 0
    
    # as instructed 1 or -1 
    def predict(self, features):
        weighted_sum = np.dot(features, self.weights) + self.bias
        return 1 if weighted_sum > 0 else -1
    
    def train(self, X, y, epochs=1):
        n_samples = len(y)
        for epoch in range(epochs):
            # shuffle the training dataset
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
    
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # updates weights and biases when there is misprediction.
            # no learning rate (fixed step size)
            for features, label in zip(X_shuffled, y_shuffled):
                prediction = self.predict(features)
                if prediction != label:
                    error = label - prediction
                    self.weights += error * features
                    self.bias += error
            
            # display epochs and accuracies
            correct_predictions = sum(self.predict(X[i]) == y[i] for i in range(n_samples))
            accuracy = correct_predictions / n_samples
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy * 100:.2f}%")
        return self

df = pd.read_csv('training_patterns.csv')

df['Pattern'] = df['Pattern'].apply(lambda x: 1 if x == 'B' else -1)
    
X = np.array([np.array([value for row in eval(pattern) for value in row]) for pattern in df['PatternData']])
y = np.array(df['Pattern'])

no_of_features = X.shape[1]
perceptron_model = PerceptronModel(no_of_features=no_of_features)

# Train the model
epochs = 5
perceptron_model.train(X, y, epochs=epochs)

# missing data test
missing_df = pd.read_csv('missing_testing_patterns.csv')

missing_df['Pattern'] = missing_df['Pattern'].apply(lambda x: 1 if x == 'B' else -1)

X_missing = np.array([np.array([item for sublist in eval(features) for item in sublist]) for features in missing_df['PatternData']])
y_missing = np.array(missing_df['Pattern'])

# print the result for missing data 
predict_and_evaluate(perceptron_model, X_missing, y_missing)

# noise 5 data test
noise_5_df = pd.read_csv('noise_5_testing_patterns.csv')

noise_5_df['Pattern'] = noise_5_df['Pattern'].apply(lambda x: 1 if x == 'B' else -1)

X_noise_5 = np.array([np.array([item for sublist in eval(features) for item in sublist]) for features in noise_5_df['PatternData']])
y_noise_5 = np.array(noise_5_df['Pattern'])

num_of_noise = 5 
noisy_patterns_5 = []

for original_pattern, label in zip(X_noise_5, y_noise_5):

    noisy_pattern = introduce_noise(original_pattern, num_of_noise)
    noisy_patterns_5.append(noisy_pattern)
    
    num_noise = count_noise(original_pattern, noisy_pattern)
    print(f"Number of noise: {num_noise}")

X_noise_5 = np.array(noisy_patterns_5)

predict_and_evaluate(perceptron_model, X_noise_5, y_noise_5)


# noise 10 data test
noise_10_df = pd.read_csv('noise_10_testing_patterns.csv')

noise_10_df['Pattern'] = noise_10_df['Pattern'].apply(lambda x: 1 if x == 'B' else -1)

X_noise_10 = np.array([np.array([item for sublist in eval(features) for item in sublist]) for features in noise_10_df['PatternData']])
y_noise_10 = np.array(noise_10_df['Pattern'])

num_of_noise = 10  
noisy_patterns_10 = []

for original_pattern, label in zip(X_noise_10, y_noise_10):

    noisy_pattern = introduce_noise(original_pattern, num_of_noise)
    noisy_patterns_10.append(noisy_pattern)
    
    num_noise = count_noise(original_pattern, noisy_pattern)
    print(f"Number of noise: {num_noise}")

X_noise_10 = np.array(noisy_patterns_10)

predict_and_evaluate(perceptron_model, X_noise_10, y_noise_10)


# noise 15 data test
noise_15_df = pd.read_csv('noise_15_testing_patterns.csv')

noise_15_df['Pattern'] = noise_15_df['Pattern'].apply(lambda x: 1 if x == 'B' else -1)

X_noise_15 = np.array([np.array([item for sublist in eval(features) for item in sublist]) for features in noise_15_df['PatternData']])
y_noise_15 = np.array(noise_15_df['Pattern'])

num_of_noise = 15
noisy_patterns_15 = []

for original_pattern, label in zip(X_noise_15, y_noise_15):
   
    noisy_pattern = introduce_noise(original_pattern, num_of_noise)
    noisy_patterns_15.append(noisy_pattern)
    
    num_noise = count_noise(original_pattern, noisy_pattern)
    print(f"Number of noise: {num_noise}")

X_noise_15 = np.array(noisy_patterns_15)
predict_and_evaluate(perceptron_model, X_noise_15, y_noise_15)

# noise 20 data test
noise_20_df = pd.read_csv('noise_20_testing_patterns.csv')

noise_20_df['Pattern'] = noise_20_df['Pattern'].apply(lambda x: 1 if x == 'B' else -1)

X_noise_20 = np.array([np.array([item for sublist in eval(features) for item in sublist]) for features in noise_20_df['PatternData']])
y_noise_20 = np.array(noise_20_df['Pattern'])

num_of_noise = 20  
noisy_patterns_20 = []

for original_pattern, label in zip(X_noise_20, y_noise_20):
    noisy_pattern = introduce_noise(original_pattern, num_of_noise)
    noisy_patterns_20.append(noisy_pattern)
    
    num_noise = count_noise(original_pattern, noisy_pattern)
    print(f"Number of noise: {num_noise}")

X_noise_20 = np.array(noisy_patterns_20)
predict_and_evaluate(perceptron_model, X_noise_20, y_noise_20)

