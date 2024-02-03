# Import the necessary python libraries for the Analysis
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import toeplitz
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

        # Defining Parameters    
        sampling_rate = 10  # The sampling rate between the input and output signals
        degree = 2  # Degree of the polynomial regression Model
        order = 2  # Order of ARX model
        lengthData = len(data)
        batchSize = int(lengthData / 2)

        input_signals = data[input_columns]

        for output_col in output_columns:
            print(f"Time Delay Estimation for {output_col}, with respect to DataFrame {file_path}")
            output_signal = data[output_col]

            def cross_correlation(signal1, signal2):
                len_signal1 = len(signal1)
                len_signal2 = len(signal2)
                len_cross_corr = len_signal1 + len_signal2 - 1

                # Use fast Fourier transform (FFT) for convolution
                fft_result = np.fft.fft(signal1, len_cross_corr) * np.fft.fft(signal2, len_cross_corr).conj()
                cross_corr = np.fft.ifft(fft_result).real

                return cross_corr

            def find_time_delay(input_signals, output_signal, sampling_rate):
                input_signals_array = input_signals.to_numpy()
                output_signal_array = output_signal.to_numpy().flatten()

                num_channels = input_signals_array.shape[1]

                # Initialize an array to store overall time delays for each channel
                time_delays = np.zeros(num_channels)

                # Calculate overall time delay for each channel
                for channel in range(num_channels):
                    # Extract the input signal for the current channel
                    input_channel = input_signals_array[:, channel]

                    # Compute cross-correlation between the input channel and the output signal
                    cross_corr = cross_correlation(output_signal_array, input_channel)

                    # Find the index of the maximum correlation value
                    max_corr_index = np.argmax(cross_corr)

                    # Calculate the time delay in seconds
                    time_delays[channel] = max_corr_index / sampling_rate

                # Return the index of the minimum time delay across all channels
                return np.argmin(time_delays), -np.min(time_delays)

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
                    poly_features = PolynomialFeatures(degree=degree)
                    input_channel_poly = poly_features.fit_transform(input_channel.reshape(-1, 1))

                    # Fit linear regression on polynomial features
                    model = LinearRegression()
                    model.fit(input_channel_poly[:, 1:], output_signal_array)

                    # The time delay is proportional to the coefficient of the highest-degree term
                    time_delays[channel] = -model.coef_[-1] / (degree * model.coef_[-2])

                # Return the index and value of the minimum time delay across all channels
                return np.argmin(time_delays), -np.min(time_delays), model.score(input_channel_poly[:, 1:], output_signal_array)

            # The rest of the code remains unchanged...
