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


def perform_time_delay_estimation(file_paths, num_input_signals_list, num_output_signals_list):
    for file_path, num_input_signals, num_output_signals in zip(file_paths, num_input_signals_list, num_output_signals_list):
        # Read the CSV files from the file_path
        data = pd.read_csv(file_path)
        

        # Importing the dataset and splitting into input_signal and output_signal        
        input_columns = [f'in{i}' for i in range(1, num_input_signals + 1)]
        output_columns = [f'out{i}' for i in range(1, num_output_signals + 1)]

        #Defining Parameters    
        sampling_rate = 2000  # The sampling rate between the input and output signals
        degree = 2  # Degree of the polynomial regression Model
        order = 2 # Order of ARX model

        input_signals = data[input_columns]

        for output_col in output_columns:
            print(f"Time Delay Estimation for {output_col}, with respect to DataFrame {file_path}")
            output_signal = data[output_col]
        
            for input_col, output_col in zip(input_signals, output_signal):
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
                return np.argmax(time_delays)

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
                return np.argmax(time_delays)

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
                return np.argmax(time_delays)

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
                return np.argmax(time_delays)

            def lstm_time_delay(input_signals, output_signal, epochs=100, batch_size=32):
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
            

            estimated_time_delay_CrossCorr = find_time_delay(input_signals, output_signal, sampling_rate)
            estimated_time_delay_poly = polynomial_regression_time_delay(input_signals, output_signal, degree)
            estimated_Linear_time_delay = linear_regression_time_delay(input_signals, output_signal)
            estimated_ARXtime_delay = arx_modeling_time_delay(input_signals, output_signal, order)
            estimated_LSTM = lstm_time_delay(input_signals, output_signal, epochs=100, batch_size=25)

            """

            def calculate_score(time_delay, target_delay=4):
                # Calculate a score based on how close the time delay is to the target delay (6 seconds in this case)
                return np.abs(target_delay - time_delay)
            """            
            # Function to optimize
            def objective_function(params):
                # Extract parameters for optimization
                estimated_time_delay_CrossCorr, estimated_time_delay_poly, estimated_Linear_time_delay, estimated_ARXtime_delay = params


                # Calculate squared differences between estimated and actual delays
                squared_diff = [
                    (estimated_time_delay_CrossCorr - find_time_delay(input_signals, output_signal, sampling_rate))**2,
                    (estimated_time_delay_poly - polynomial_regression_time_delay(input_signals, output_signal, degree))**2,
                    (estimated_Linear_time_delay - linear_regression_time_delay(input_signals, output_signal))**2,
                    (estimated_ARXtime_delay - arx_modeling_time_delay(input_signals, output_signal, order))**2
                ]
                
                # Sum of scores, aiming to minimize the total score
                return sum(squared_diff)

            # Initial guesses for the time delays
            initial_guesses = [0, 0, 0, 0]

            # Define bounds for the time delays (non-negative)
            bounds = [(0, None), (0, None), (0, None), (0, None)]

            # Minimize the objective function using SciPy
            result = minimize(objective_function, initial_guesses, bounds=bounds)

            # Get the optimized time delays
            estimated_CrossCorr_time_delay_opt = result.x[0]
            estimated_Poly_time_delay_opt = result.x[1]
            estimated_Linear_time_delay_opt = result.x[2]
            estimated_ARXtime_delay_opt = result.x[3]


            # Choose the method with the best time delay
            delays = [estimated_time_delay_CrossCorr, estimated_time_delay_poly, estimated_Linear_time_delay, estimated_ARXtime_delay]
            best_method_index = np.argmax(delays)

            # Choose the method with the best time delay based on the optimization results
            optimal_delays = [estimated_CrossCorr_time_delay_opt, estimated_Poly_time_delay_opt, estimated_Linear_time_delay_opt, estimated_ARXtime_delay_opt]
            best_method_index_opt = np.argmin(optimal_delays)


            # Add LSTM method
            lstm_time_delay_opt = lstm_time_delay(input_signals, output_signal)
            optimal_delays.append(lstm_time_delay_opt)


            print(f"Cross Correlation Time Delay: {estimated_time_delay_CrossCorr} seconds")
            print(f"Polynomial Regression Time Delay: {estimated_time_delay_poly} seconds")
            print(f"Linear Regression Time Delay: {estimated_Linear_time_delay} seconds")
            print(f"ARX Modeling Time Delay: {estimated_ARXtime_delay} seconds")
            print(f"Long Short Term Memory (LSTM) Time Delay: {estimated_LSTM} seconds")
            print()
            print(f"Best Compensation Method: {['Cross Correlation', 'Polynomial Regression', 'Linear Regression', 'ARX'][best_method_index]}")
            print()


            print(f"Optimized Cross Correlation Time Delay: {estimated_CrossCorr_time_delay_opt} seconds")
            print(f"Optimized Polynomial Regression Time Delay: {estimated_Poly_time_delay_opt} seconds")
            print(f"Optimized Linear Regression Time Delay: {estimated_Linear_time_delay_opt} seconds")
            print(f"Optimized ARX Modeling Time Delay: {estimated_ARXtime_delay_opt} seconds")
            print(f"Optimized LSTM Time Delay: {lstm_time_delay_opt} seconds")
            print()
            print(f"Best Compensation Method (Optimized): {['Cross Correlation', 'Polynomial Regression', 'Linear Regression', 'ARX'][best_method_index_opt]}")
            print('########################################################################################################################################################')
            

# List of file paths
data = [

    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds1.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds2.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds3.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds4.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds5.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds6.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds7.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds8.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds9.csv',
    'C:/Users/#emmyCode/Desktop/ErnestProject/Olive/ds10.csv',

]


# Specify the number of input and output signals for each CSV file
num_input_signals_list = [4, 4, 4, 4, 4, 5, 6, 6, 7, 6]          #This list specifies the number of input signals each dataframe takes, so index 0 is for ds1.csv and so on
num_output_signals_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]         #This list specifies the number of output signals each dataframe takes, so index 0 is for ds1.csv and so on

perform_time_delay_estimation(data, num_input_signals_list, num_output_signals_list)
