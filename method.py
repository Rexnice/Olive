# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import numpy as np

# def get_error(X, Y):
#     model = LinearRegression()
#     model.fit(X, Y)
#     return model.score(X, Y)

# file_path_csv = 'cleaned_transformed_ds1.csv'
# data = pd.read_csv(file_path_csv)

# input_columns = ['in1','in2', 'in3', 'in4']
# # input_columns = ['input0','Input1', 'Input2', 'Input3', 'Input4', 'Input5', 'Input6']
# output_column = 'out1'

# input_data = data[input_columns]
# original_output_data = data[output_column]

# results = []

# for x in range(1729):
#     delay = x * 2
#     shifted_output = original_output_data.shift(-x)  # Negative shift for forward shifting
#     shifted_output.fillna(original_output_data.mean(), inplace=True)
    
#     error = get_error(input_data, shifted_output)
#     results.append((delay, error))
#     print(f"Model Evaluation - Delay: {delay} seconds, Regression Score: {error:.4f}")


# print(results)
# max_score = max(results, key=lambda item: item[1])
# print(f"Maximum Score: Delay = {max_score[0]} sec, Score = {max_score[1]:.4f}")


# data = pd.read_csv("cleaned_transformed_ds1.csv")

# n = 11
# df2 = data.iloc[n:]
# print("After dropping first n rows:\n", df2.head(10))
# #print(data.describe())
# df2.to_csv("cleaned_transformed_ds1.csv")



# target_delay = int(output_col.split('_')[0])
# print(target_delay)

import re

# Input string
output_col = 'out1_12sec_shift_filt'
# Regular expression pattern to extract the integer
pattern = r'(\d+)sec'

# Using re.search to find the first match of the pattern in the string
match = re.search(pattern, output_col)
integer_value = int(match.group(1))
print(integer_value)

delay = [10, 3, 4, 6]
target_delay = integer_value

# Find the delay in the list that is closest to the target delay
best_delay = min(delay, key=lambda x: abs(x - target_delay))

print("Best delay:", best_delay)

