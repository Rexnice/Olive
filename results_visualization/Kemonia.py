import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load the existing dataset
data = pd.read_csv('Modified_Before&AfterOpti3.csv')

# Define the regular expression pattern
pattern = r'_shift$'
pattern2 = r'_shift_filt$'
pattern3 = r'_shift_noise$'
pattern4 = r'_shift_noise_filt$'
# Filter the dataset based on the regular expression pattern for Output column
filtered_data = data[data['Output'].str.contains(pattern)]
filtered_data2 = data[data['Output'].str.contains(pattern2)]
filtered_data3 = data[data['Output'].str.contains(pattern3)]
filtered_data4 = data[data['Output'].str.contains(pattern4)]

# Select the columns of interest
new_dataset = filtered_data[['Filename', 'Output', 'LM2']]
new_dataset2 = filtered_data2[['Filename', 'Output', 'LM2']]
new_dataset3 = filtered_data3[['Filename', 'Output', 'LM2']]
new_dataset4 = filtered_data4[['Filename', 'Output', 'LM2']]

# Plot a line plot using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=new_dataset, x=new_dataset.index, y='LM2', color='black')
plt.title('LM2 Visualization for "_shift" after Optimization')
plt.xlabel('Data Index')
plt.ylabel('LM2 Delay(s)')
plt.xticks(rotation=45)
# plt.tight_layout()
plt.show()

# Plot a line plot using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=new_dataset2, x=new_dataset.index, y='LM2', color='black')
plt.title('LM2 Visualization for "_shift_filt" after Optimization')
plt.xlabel('Data Index')
plt.ylabel('LM2 Delay(s)')
plt.xticks(rotation=45)
# plt.tight_layout()
plt.show()

# Plot a line plot using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=new_dataset3, x=new_dataset.index, y='LM2', color='black')
plt.title('LM2 Visualization for "_shift_noise" after Optimization')
plt.xlabel('Data Index')
plt.ylabel('LM2 Delay(s)')
plt.xticks(rotation=45)
# plt.tight_layout()
plt.show()

# Plot a line plot using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(data=new_dataset4, x=new_dataset.index, y='LM2', color='black')
plt.title('LM2 Visualization for "_shift_noise_filt" after Optimization')
plt.xlabel('Data Index')
plt.ylabel('LM2 Delay(s)')
plt.xticks(rotation=45)
# plt.tight_layout()
plt.show()














# # Load your dataset
# data = pd.read_csv('Modified_Before&AfterOpti3.csv')
# # Extract delay and convert to numeric
# data['Delay'] = data['Output'].apply(lambda x: re.search(r'(\d+sec)', x).group(0) if re.search(r'(\d+sec)', x) else None)
# data['Delay_Numeric'] = data['Delay'].str.extract(r'(\d+)').astype(float)

# # Categorize data by conditions
# data['Condition'] = data['Output'].apply(lambda x: 'Both Noise and Filter' if 'noise' in x and 'filt' in x else
#                                           ('Noise Only' if 'noise' in x else
#                                            ('Filtered Only' if 'filt' in x else 'None')))

# print(data['Delay_Numeric'])



# # Plotting
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 18), sharey=True)
# axes = axes.flatten()

# for idx, (method1, method2) in enumerate(method_pairs):
#     sns.linelot(ax=axes[idx], data=melted_data[melted_data['Method'].isin([f'{method1}_Deviation', f'{method2}_Deviation'])],
#                 x='Condition', y='Deviation', hue='Method')
#     axes[idx].set_title(f'{method1} vs {method2} Deviation Comparison')
#     axes[idx].set_xlabel('Condition')
#     axes[idx].set_ylabel('Deviation from Delay (seconds)')
#     axes[idx].legend(title='Method')

# plt.tight_layout()
# plt.show()


















































# # #Import the neccessary python libraries for the Time Delay Analysis
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from sklearn.metrics import r2_score
# import pandas as pd
# import numpy as np
# from scipy.linalg import toeplitz
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from scipy.signal import correlate
# from scipy.optimize import minimize
# import re
# import warnings
# warnings.filterwarnings("ignore")

# data = [

#     'cleaned_transformed_ds1.csv','cleaned_transformed_ds2.csv','cleaned_transformed_ds3.csv','cleaned_transformed_ds4.csv','cleaned_transformed_ds5.csv',
# ]

# def perform_time_delay_estimation(file_paths, num_input_signals_list, num_output_signals_list):
#     #Defining Parameters    
#     sampling_rate = 1000  # The sampling rate between the input and output signals
#     degree = 2  # Degree of the polynomial regression Model
#     order = 2 # Order of ARX model
#     allresults = []
#     allresults2 = []
#     for file_path, num_input_signals, num_output_signals in zip(file_paths, num_input_signals_list, num_output_signals_list):
#         # Read the CSV files from the file_path
#         data = pd.read_csv(file_path)
    
#         input_columns = data.columns[:num_input_signals]
#         output_columns = data.columns[num_input_signals:]

#         input_signals = data[input_columns]

#         for output_col in output_columns:
#             output_signal = data[output_col]
            
#             def arx_modeling_time_delay2(input_signals, output_signal, order):
#                 input_signals_array = input_signals.to_numpy()
#                 output_signal_array = output_signal.to_numpy().flatten()

#                 num_channels = input_signals_array.shape[1]

#                 # Initialize an array to store time delays for each channel
#                 time_delays = np.zeros(num_channels)
#                 # Perform ARX modeling for each channel
#                 for channel in range(num_channels):
#                     # Extract the input signal for the current channel
#                     input_channel = input_signals_array[:, channel]

#                     # Construct the Hankel matrix
#                     hankel_matrix = toeplitz(output_signal_array[:-order+1], np.flip(output_signal_array[:order]))

#                     # Construct the matrix of input signals
#                     input_matrix = toeplitz(input_channel[:-order+1], np.flip(input_channel[:order]))

#                     # Concatenate the input matrix with the Hankel matrix
#                     augmented_matrix = np.column_stack((input_matrix, hankel_matrix))

#                     # Solve for the coefficients using least squares
#                     coefficients = np.linalg.lstsq(augmented_matrix, output_signal_array[order-1:], rcond=None)[0]

#                     # The time delay is proportional to the coefficient of the first input
#                     time_delays[channel] = -coefficients[0] / coefficients[1]

#                 # Return the time delays for all channels
#                 return time_delays

#         estimated_ARXtime_delay2 = arx_modeling_time_delay2(input_signals, output_signal, order)
#         print(np.argmax(estimated_ARXtime_delay2))
        


# def main():
#     num_input_signals_list = [4, 4]
#     num_output_signals_list = [42, 42]
#     # num_input_signals_list = [4, 4, 4, 8, 8, 5, 6, 6, 7, 6]
#     # num_output_signals_list = [42, 42, 42, 168, 168, 147, 168, 168, 126, 168]
    
#     result = perform_time_delay_estimation(data, num_input_signals_list, num_output_signals_list)
#     return result

# if __name__ == "__main__":
#     main()