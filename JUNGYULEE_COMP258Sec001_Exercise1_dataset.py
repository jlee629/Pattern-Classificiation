# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:19:39 2024

Jungyu Lee
301236221

Pattern Classfication App
"""
# Testing dataset - matrix-like format 9x7

a_pattern1 = (
    (0, 0, 2, 2, 0, 0, 0),
    (0, 0, 0, 2, 0, 0, 0),
    (0, 0, 0, 2, 0, 0, 0),
    (0, 0, 2, 0, 2, 0, 0),
    (0, 0, 2, 0, 2, 0, 0),
    (0, 2, 2, 2, 2, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (2, 2, 2, 0, 2, 2, 2)
)

b_pattern1 = (
    (2, 2, 2, 2, 2, 2, 0),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 2, 2, 2, 2, 0),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (2, 2, 2, 2, 2, 2, 0)    
)

c_pattern1 = (
    (0, 0, 2, 2, 2, 2, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 0, 2, 2, 2, 2, 0)    
)


d_pattern1 = (
    (2, 2, 2, 2, 2, 0, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 2, 0),
    (2, 2, 2, 2, 2, 0, 0)    
)

e_pattern1 = (
    (2, 2, 2, 2, 2, 2, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 0),
    (0, 2, 0, 2, 0, 0, 0),
    (0, 2, 2, 2, 0, 0, 0),
    (0, 2, 0, 2, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 0),
    (0, 2, 0, 0, 0, 0, 2),
    (2, 2, 2, 2, 2, 2, 2)    
)

j_pattern1 = (
    (0, 0, 0, 2, 2, 2, 2),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 0, 2, 2, 2, 0, 0)    
)

k_pattern1 = (
    (2, 2, 2, 0, 0, 2, 2),
    (0, 2, 0, 0, 2, 0, 0),
    (0, 2, 0, 2, 0, 0, 0),
    (0, 2, 2, 0, 0, 0, 0),
    (0, 2, 2, 0, 0, 0, 0),
    (0, 2, 0, 2, 0, 0, 0),
    (0, 2, 0, 0, 2, 0, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (2, 2, 2, 0, 0, 2, 2)    
)

a_pattern2 = (
    (0, 0, 0, 2, 0, 0, 0),
    (0, 0, 0, 2, 0, 0, 0),
    (0, 0, 0, 2, 0, 0, 0),
    (0, 0, 2, 0, 2, 0, 0),
    (0, 0, 2, 0, 2, 0, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 2, 2, 2, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 2, 0)
)

b_pattern2 = (
    (2, 2, 2, 2, 2, 2, 0),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 2, 2, 2, 2, 2, 0),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 2, 2, 2, 2, 2, 0)    
)

c_pattern2 = (
    (0, 0, 2, 2, 2, 0, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 0, 2, 2, 2, 0, 0)    
)

d_pattern2 = (
    (2, 2, 2, 2, 2, 0, 0),
    (2, 0, 0, 0, 0, 2, 0),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 2, 0),
    (2, 2, 2, 2, 2, 0, 0)    
)

e_pattern2 = (
    (2, 2, 2, 2, 2, 2, 2),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 2, 2, 2, 2, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 2, 2, 2, 2, 2, 2)    
)

j_pattern2 = (
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 0, 2, 2, 2, 0, 0)    
)

k_pattern2 = (
    (2, 0, 0, 0, 0, 2, 0),
    (2, 0, 0, 0, 2, 0, 0),
    (2, 0, 0, 2, 0, 0, 0),
    (2, 0, 2, 0, 0, 0, 0),
    (2, 2, 0, 0, 0, 0, 0),
    (2, 0, 2, 0, 0, 0, 0),
    (2, 0, 0, 2, 0, 0, 0),
    (2, 0, 0, 0, 2, 0, 0),
    (2, 0, 0, 0, 0, 2, 0)    
)

a_pattern3 = (
    (0, 0, 0, 2, 0, 0, 0),
    (0, 0, 0, 2, 0, 0, 0),
    (0, 0, 2, 0, 2, 0, 0),
    (0, 0, 2, 0, 2, 0, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 2, 2, 2, 2, 0),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 2, 0, 0, 0, 2, 2)
)

b_pattern3 = (
    (2, 2, 2, 2, 2, 2, 0),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 2, 2, 2, 2, 0),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (2, 2, 2, 2, 2, 2, 0)    
)

c_pattern3 = (
    (0, 0, 2, 2, 2, 0, 2),
    (0, 2, 0, 0, 0, 2, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 0, 2, 2, 2, 0, 0)    
)

d_pattern3 = (
    (2, 2, 2, 2, 2, 0, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 2, 0),
    (2, 2, 2, 2, 2, 0, 0)    
)

e_pattern3 = (
    (2, 2, 2, 2, 2, 2, 2),
    (0, 2, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 2, 0, 0),
    (0, 2, 2, 2, 2, 0, 0),
    (0, 2, 0, 0, 2, 0, 0),
    (0, 2, 0, 0, 0, 0, 0),
    (0, 2, 0, 0, 0, 0, 0),
    (0, 2, 0, 0, 0, 0, 2),
    (2, 2, 2, 2, 2, 2, 2)    
)

j_pattern3 = (
    (0, 0, 0, 0, 2, 2, 2),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 0, 2, 2, 2, 0, 0)    
)

k_pattern3 = (
    (2, 2, 2, 0, 0, 2, 2),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 2, 0, 0),
    (0, 2, 0, 2, 0, 0, 0),
    (0, 2, 2, 0, 0, 0, 0),
    (0, 2, 0, 2, 0, 0, 0),
    (0, 2, 0, 0, 2, 0, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (2, 2, 2, 0, 0, 2, 2)    
)

import pandas as pd
import numpy as np

data = {
        'PatternData': [a_pattern1, b_pattern1, c_pattern1, d_pattern1, e_pattern1, j_pattern1, k_pattern1,
                        a_pattern2, b_pattern2, c_pattern2, d_pattern2, e_pattern2, j_pattern2, k_pattern2,
                        a_pattern3, b_pattern3, c_pattern3, d_pattern3, e_pattern3, j_pattern3, k_pattern3],
        'Pattern': ['A', 'B', 'C', 'D', 'E', 'J', 'K',
                    'A', 'B', 'C', 'D', 'E', 'J', 'K',
                    'A', 'B', 'C', 'D', 'E', 'J', 'K']
        }

df = pd.DataFrame(data)

# convert the pattern data into the bipolar form
def convert_to_bipolar_list(pattern):
    pattern_array = np.array(pattern)
    bipolar_array = np.where(pattern_array == 2, 1, -1)
    return bipolar_array.tolist()

df['PatternData'] = df['PatternData'].apply(convert_to_bipolar_list)

df.to_csv('training_patterns.csv', columns=['PatternData', 'Pattern'], index=False)

# testing data - noise
noise_5_data = {
        'PatternData': [a_pattern3, b_pattern3, c_pattern3, d_pattern3, e_pattern3, j_pattern3, k_pattern3],
        'Pattern': ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        }

noise_5_df = pd.DataFrame(noise_5_data)
noise_5_df['PatternData'] = noise_5_df['PatternData'].apply(convert_to_bipolar_list)

noise_5_df.to_csv('noise_5_testing_patterns.csv', columns=['PatternData', 'Pattern'], index=False)


noise_10_data = {
        'PatternData': [a_pattern3, b_pattern3, c_pattern3, d_pattern3, e_pattern3, j_pattern3, k_pattern3],
        'Pattern': ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        }

noise_10_df = pd.DataFrame(noise_10_data)
noise_10_df['PatternData'] = noise_10_df['PatternData'].apply(convert_to_bipolar_list)

noise_10_df.to_csv('noise_10_testing_patterns.csv', columns=['PatternData', 'Pattern'], index=False)


noise_15_data = {
        'PatternData': [a_pattern3, b_pattern3, c_pattern3, d_pattern3, e_pattern3, j_pattern3, k_pattern3],
        'Pattern': ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        }

noise_15_df = pd.DataFrame(noise_15_data)
noise_15_df['PatternData'] = noise_15_df['PatternData'].apply(convert_to_bipolar_list)

noise_15_df.to_csv('noise_15_testing_patterns.csv', columns=['PatternData', 'Pattern'], index=False)


noise_20_data = {
        'PatternData': [a_pattern3, b_pattern3, c_pattern3, d_pattern3, e_pattern3, j_pattern3, k_pattern3],
        'Pattern': ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        }

noise_20_df = pd.DataFrame(noise_20_data)
noise_20_df['PatternData'] = noise_20_df['PatternData'].apply(convert_to_bipolar_list)

noise_20_df.to_csv('noise_20_testing_patterns.csv', columns=['PatternData', 'Pattern'], index=False)



# testing data - missing 

a_pattern4 = (
    (0, 0, 0, 2, 0, 0, 0),
    (0, 0, 2, 0, 2, 0, 0),
    (0, 0, 2, 0, 2, 0, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 2, 2, 2, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0)
)

b_pattern4 = (
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 2, 2, 2, 2, 2, 0),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 2, 2, 2, 2, 2, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0)    
)

c_pattern4 = (
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 2),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 0, 2, 2, 2, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0)    
)

d_pattern4 = (
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 0, 2),
    (2, 0, 0, 0, 0, 2, 0),
    (2, 2, 2, 2, 2, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0)
    
)

e_pattern4 = (
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 2, 2, 2, 2, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 0, 0, 0, 0, 0, 0),
    (2, 2, 2, 2, 2, 2, 2),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0)    
)

j_pattern4 = (
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 2, 0, 0, 0, 2, 0),
    (0, 0, 2, 2, 2, 0, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0)    
)

k_pattern4 = (
    (2, 0, 0, 2, 0, 0, 0),
    (2, 0, 2, 0, 0, 0, 0),
    (2, 2, 0, 0, 0, 0, 0),
    (2, 0, 2, 0, 0, 0, 0),
    (2, 0, 0, 2, 0, 0, 0),
    (2, 0, 0, 0, 2, 0, 0),
    (2, 0, 0, 0, 0, 2, 0),
    (0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0)    
)

missing_data = {
        'PatternData': [a_pattern4, b_pattern4, c_pattern4, d_pattern4, e_pattern4, j_pattern4, k_pattern4],
        'Pattern': ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        }

missing_df = pd.DataFrame(missing_data)

missing_df['PatternData'] = missing_df['PatternData'].apply(convert_to_bipolar_list)

missing_df.to_csv('missing_testing_patterns.csv', columns=['PatternData', 'Pattern'], index=False)


