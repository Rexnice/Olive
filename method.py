import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def get_error(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    return model.score(X, Y)

file_path_csv = 'dataset2.csv'
data = pd.read_csv(file_path_csv)

input_columns = ['input0','Input1', 'Input2', 'Input3', 'Input4', 'Input5', 'Input6']
output_column = 'Output1_delay6sec'

input_data = data[input_columns]
original_output_data = data[output_column]

results = []

for x in range(40):
    delay = x * 2
    shifted_output = original_output_data.shift(-x)  # Negative shift for forward shifting
    shifted_output.fillna(original_output_data.mean(), inplace=True)
    
    error = get_error(input_data, shifted_output)
    results.append((delay, error))
    print(f"Model Evaluation - Delay: {delay} seconds, Regression Score: {error:.4f}")


print(results)
max_score = max(results, key=lambda item: item[1])
print(f"Maximum Score: Delay = {max_score[0]} sec, Score = {max_score[1]:.4f}")
