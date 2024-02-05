# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:20:40 2024

Jungyu Lee
301236221

Pattern Classfication App
"""
import numpy as np

def predict_and_evaluate(model, patterns, actual_labels):
    correct_predictions = 0

    for pattern, label in zip(patterns, actual_labels):
        flattened_pattern = np.array(pattern).flatten()
        prediction = model.predict(flattened_pattern)

        # increment correct_predictions 
        if prediction == label:
            correct_predictions += 1

        prediction_text = "It's B" if prediction == 1 else "It's not B"
        actual_text = "It's B" if label == 1 else "It's not B"
        print(f"Pattern Prediction: {prediction_text}, Actual: {actual_text}")

    # accuracy
    accuracy = correct_predictions / len(actual_labels)
    print(f"Accuracy : {accuracy * 100:.2f}%")


def introduce_noise(pattern, num_of_noise):
    np.random.seed(21)
    original_shape = pattern.shape  
    pattern = np.array(pattern).flatten() 
    
    num_of_noise = min(num_of_noise, len(pattern))
    positions_to_change = np.random.choice(len(pattern), num_of_noise, replace=False)
    pattern[positions_to_change] *= -1  
    
    return pattern.reshape(original_shape)

def count_noise(original_pattern, changed_pattern):
    original_flat = np.array(original_pattern).flatten()
    changed_flat = np.array(changed_pattern).flatten()

    return np.sum(original_flat != changed_flat)

