#Import the neccessary python libraries for the Analysis
import pandas as pd
import numpy as np
from scipy.linalg import toeplitz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.signal import correlate
from scipy.optimize import minimize
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler







def perform_time_delay_estimation(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Perform time delay estimation
    inputs = ['input1', 'input2', 'input3', 'input4', 'input5', 'input6']
    outputs = ['output1', 'output2', 'output3', 'output4', 'output5', 'output6']
    
    for input_col, output_col in zip(inputs, outputs):
        def cross_correlation(signal1, signal2):
            len_signal1 = len(signal1)
            len_signal2 = len(signal2)
            len_cross_corr = len_signal1 + len_signal2 - 1

            # Initialize cross-correlation array with zeros
            cross_corr = [0] * len_cross_corr

            # Reverse the second signal for convolution (cross-correlation)
            signal2_rev = signal2[::-1]

            # Perform cross-correlation Calculation
            for i in range(len_cross_corr):
                for j in range(len_signal1):
                    if i - j >= 0 and i - j < len_signal2:
                        cross_corr[i] += signal1[j] * signal2_rev[i - j]

            return cross_corr

    def find_time_delay(input_signals, output_signal, sampling_rate):
        input_signals_array = input_signals.to_numpy()
        output_signal_array = output_signal.to_numpy().flatten()

        num_channels = input_signals_array.shape[1]

        # Initialize an array to store overall time delays for each channel
        time_delays = np.zeros(num_channels)

        # Calculate overall time delay for each channel
        for channel in range(num_channels):
            # Extract the input signal for the current channel..
            input_channel = input_signals_array[:, channel]

            # Compute cross-correlation between the input channel and the output signal
            cross_corr = cross_correlation(output_signal_array, input_channel)

            # Find the index of the maximum correlation value
            max_corr_index = cross_corr.index(max(cross_corr))

            # Calculate the time delay in seconds
            time_delays[channel] = max_corr_index / sampling_rate

        # Return the maximum time delay across all channels
        return np.argmin(time_delays)

    def polynomial_regression_time_delay(input_signals, output_signal, degree):
        input_signals_array = input_signals.to_numpy()
        output_signal_array = output_signal.to_numpy().flatten()

        num_channels = input_signals_array.shape[1]

        # Initialize an array to store time delays for each channel
        time_delays = np.zeros(num_channels)

        # Perform polynomial regression for each channel
        for channel in range(num_channels):
            # Extract the input signal for the current channel
            input_channel = input_signals_array[:, channel]

            # Fit a polynomial regression model
            coeffs = np.polyfit(input_channel, output_signal_array, degree)

            # The time delay is proportional to the coefficient of the highest-degree term
            time_delays[channel] = -coeffs[-2] / (degree * coeffs[-1])

        # Return the Maximum time delay across all channels
        return np.argmin(time_delays)

    def linear_regression_time_delay(input_signals, output_signal):
        # Convert input and output signals to NumPy arrays
        input_signals_array = input_signals.to_numpy()
        output_signal_array = output_signal.to_numpy().flatten()

        num_channels = input_signals_array.shape[1]

        # Initialize an array to store time delays for each channel
        time_delays = np.zeros(num_channels)

        # Perform linear regression for each channel
        for channel in range(num_channels):
            # Extract the input signal for the current channel
            input_channel = input_signals_array[:, channel]

            # Calculate the covariance and variance
            cov = np.cov(input_channel, output_signal_array)[0, 1]
            var_x = np.var(input_channel)

            # Calculate the slope (coefficient) of the linear regression
            slope = cov / var_x

            # Calculate the intercept of the linear regression
            intercept = np.mean(output_signal_array) - slope * np.mean(input_channel)

            # The time delay is proportional to the slope
            time_delays[channel] = -intercept / slope

        # Return the Maximum time delay across all channels
        return np.argmin(time_delays)

    def arx_modeling_time_delay(input_signals, output_signal, order):
        input_signals_array = input_signals.to_numpy()
        output_signal_array = output_signal.to_numpy().flatten()

        num_channels = input_signals_array.shape[1]

        # Initialize an array to store time delays for each channel
        time_delays = np.zeros(num_channels)

        # Perform ARX modeling for each channel
        for channel in range(num_channels):
            # Extract the input signal for the current channel
            input_channel = input_signals_array[:, channel]

            # Construct the Hankel matrix
            hankel_matrix = toeplitz(output_signal_array[:-order+1], np.flip(output_signal_array[:order]))

            # Construct the matrix of input signals
            input_matrix = toeplitz(input_channel[:-order+1], np.flip(input_channel[:order]))

            # Concatenate the input matrix with the Hankel matrix
            augmented_matrix = np.column_stack((input_matrix, hankel_matrix))

            # Solve for the coefficients using least squares
            coefficients = np.linalg.lstsq(augmented_matrix, output_signal_array[order-1:], rcond=None)[0]

            # The time delay is proportional to the coefficient of the first input
            time_delays[channel] = -coefficients[0] / coefficients[1]

        # Return the Maximum time delay across all channels
        return np.argmin(time_delays)

    def lstm_time_delay(input_signals, output_signal, epochs=60, batch_size=32):
        # Normalize input and output data
        scaler = MinMaxScaler(feature_range=(0, 1))
        input_signals_scaled = scaler.fit_transform(input_signals)
        output_signal_scaled = scaler.fit_transform(output_signal.values.reshape(-1, 1))

        # Reshape input data for LSTM
        input_signals_reshaped = np.reshape(input_signals_scaled, (input_signals_scaled.shape[0], 1, input_signals_scaled.shape[1]))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(1, input_signals_scaled.shape[1])))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(input_signals_reshaped, output_signal_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

        # Predict output using the trained model
        predicted_output_scaled = model.predict(input_signals_reshaped)

        # Inverse transform to get the original scale
        predicted_output = scaler.inverse_transform(predicted_output_scaled)

        # Calculate the time delay
        time_delay = find_time_delay(pd.DataFrame(predicted_output), output_signal, sampling_rate)

        return time_delay

    # Importing the dataset and splitting into input_signal and output_signal
    data = pd.read_csv("C:/Users/#emmyCode/Desktop/ErnestProject/Olive/dataset2.csv")
    input_columns = ['input0', 'Input1', 'Input2', 'Input3', 'Input4', 'Input5', 'Input6']

    #input_columns = ['input1', 'input1.1', 'input1.2', 'input1.3', 'input1.4']
    output_column = 'Output1_delay6sec'
    sampling_rate = 2000  # The sampling rate is as shown in the dataset
    degree = 2  # Degree of the polynomial regression
    order = 2 # Order of ARX model


    input_signals = data[input_columns]
    output_signal = data[output_column]

    overall_time_delay = find_time_delay(input_signals, output_signal, sampling_rate)
    estimated_time_delay = polynomial_regression_time_delay(input_signals, output_signal, degree)
    estimated_Linear_time_delay = linear_regression_time_delay(input_signals, output_signal)
    estimated_ARXtime_delay = arx_modeling_time_delay(input_signals, output_signal, order)
    estimated_LSTM = lstm_time_delay(input_signals, output_signal, epochs=50, batch_size=32)


    def calculate_score(time_delay, target_delay=6):
        # Calculate a score based on how close the time delay is to the target delay (6 seconds in this case)
        return np.abs(target_delay - time_delay)

    # Function to optimize
    def objective_function(params):
        # Extract parameters for optimization
        overall_time_delay, estimated_time_delay, estimated_Linear_time_delay, estimated_ARXtime_delay = params

        # Calculate scores for each method
        scores = [
            calculate_score(overall_time_delay),
            calculate_score(estimated_time_delay),
            calculate_score(estimated_Linear_time_delay),
            calculate_score(estimated_ARXtime_delay)
        ]

        # Sum of scores, aiming to minimize the total score
        return sum(scores)

    # Initial guesses for the time delays
    initial_guesses = [1, 1, 1, 1]

    # Define bounds for the time delays (non-negative)
    bounds = [(0, None), (0, None), (0, None), (0, None)]

    # Minimize the objective function using SciPy
    result = minimize(objective_function, initial_guesses, bounds=bounds)

    # Get the optimized time delays
    overall_time_delay_opt = result.x[0]
    estimated_time_delay_opt = result.x[1]
    estimated_Linear_time_delay_opt = result.x[2]
    estimated_ARXtime_delay_opt = result.x[3]

    # Choose the method with the best time delay based on the optimization results
    optimal_delays = [overall_time_delay_opt, estimated_time_delay_opt, estimated_Linear_time_delay_opt, estimated_ARXtime_delay_opt]
    best_method_index_opt = np.argmin(optimal_delays)


    # Choose the method with the best time delay, for this case closer to 6 secs
    delays = [overall_time_delay, estimated_time_delay, estimated_Linear_time_delay, estimated_ARXtime_delay]
    best_method_index = np.argmax(delays)


    # Add LSTM method
    lstm_time_delay_opt = lstm_time_delay(input_signals, output_signal)
    optimal_delays.append(lstm_time_delay_opt)


    print(f"Polynomial Regression Time Delay: {estimated_time_delay} seconds")
    print(f"Cross Correlation Time Delay: {overall_time_delay} seconds")
    print(f"Linear Regression Time Delay: {estimated_Linear_time_delay} seconds")
    print(f"ARX Modeling Time Delay: {estimated_ARXtime_delay} seconds")
    print(f"Long Short Term Memory (LSTM) Time Delay: {estimated_LSTM} seconds")
    print()
    print(f"Best Compensation Method: {['Linear Regression', 'Polynomial Regression', 'Cross Correlation', 'ARX'][best_method_index]}")
    print()


    print(f"Optimized Polynomial Regression Time Delay: {estimated_time_delay_opt} seconds")
    print(f"Optimized Cross Correlation Time Delay: {overall_time_delay_opt} seconds")
    print(f"Optimized Linear Regression Time Delay: {estimated_Linear_time_delay_opt} seconds")
    print(f"Optimized ARX Modeling Time Delay: {estimated_ARXtime_delay_opt} seconds")
    print(f"Optimized LSTM Time Delay: {lstm_time_delay_opt} seconds")
    print()
    print(f"Best Compensation Method (Optimized): {['Linear Regression', 'Polynomial Regression', 'Cross Correlation', 'ARX'][best_method_index_opt]}")

        

# List of file paths
file_paths = [
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds1.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds2.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds3.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds8.csv',
]
# Process each file
for file_path in file_paths:
    print(f"Output Result for {file_path}")
    print()
    perform_time_delay_estimation(file_path)





