# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 22:44:32 2024

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

class AdalineModel:
    def __init__(self, no_of_features, learning_rate=0.01, epochs=5):
        self.weights = np.zeros(no_of_features)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_ = []
    
    # as instructed 1 or -1 
    def predict(self, features):
        weighted_sum = np.dot(features, self.weights) + self.bias
        return np.where(weighted_sum >= 0.0, 1, -1)
    
    # updates weights and biases
    def train(self, X, y):
        for epoch in range(self.epochs):
            indices = np.arange(X.shape[0])
            # shuffle the training data every epoch
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            errors = []
            # gradient descent approach / minimization of a continuous cost function
            for features, label in zip(X_shuffled, y_shuffled):
                prediction = self.predict(features)
                error = label - prediction
                self.weights += self.learning_rate * error * features
                self.bias += self.learning_rate * error
                errors.append(error)
            
            # (mean squared error) for this epoch
            cost = (np.array(errors) ** 2).mean() / 2.0
            self.cost_.append(cost)
            
            # accuracy for this epoch
            correct_predictions = sum(self.predict(X[i]) == y[i] for i in range(X.shape[0]))
            accuracy = correct_predictions / X.shape[0]
            print(f"Epoch {epoch+1}/{self.epochs}, Cost: {cost:.4f}, Accuracy: {accuracy * 100:.2f}%")
            
        return self

# load the dataset
df = pd.read_csv('training_patterns.csv')
df['Pattern'] = df['Pattern'].apply(lambda x: 1 if x == 'B' else -1)

X = np.array([np.array([value for row in eval(pattern) for value in row]) for pattern in df['PatternData']])
y = np.array(df['Pattern'])

# Initialize and train the Adaline model
no_of_features = X.shape[1]
adaline_model = AdalineModel(no_of_features=no_of_features)
adaline_model.train(X, y)

# missing data test
missing_df = pd.read_csv('missing_testing_patterns.csv')

missing_df['Pattern'] = missing_df['Pattern'].apply(lambda x: 1 if x == 'B' else -1)

X_missing = np.array([np.array([item for sublist in eval(features) for item in sublist]) for features in missing_df['PatternData']])
y_missing = np.array(missing_df['Pattern'])

predict_and_evaluate(adaline_model, X_missing, y_missing)

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

predict_and_evaluate(adaline_model, X_noise_5, y_noise_5)

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

predict_and_evaluate(adaline_model, X_noise_10, y_noise_10)

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
predict_and_evaluate(adaline_model, X_noise_15, y_noise_15)

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
predict_and_evaluate(adaline_model, X_noise_20, y_noise_20)

