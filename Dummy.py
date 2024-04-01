# #Import the neccessary python libraries for the Time Delay Analysis
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from scipy.linalg import toeplitz
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.signal import correlate
from scipy.optimize import minimize
import re
import warnings
warnings.filterwarnings("ignore")

data = [

    # 'cleaned_transformed_ds1.csv',
    # 'cleaned_transformed_ds2.csv',
    # 'cleaned_transformed_ds3.csv',
    # 'cleaned_transformed_ds4.csv',
    # 'cleaned_transformed_ds5.csv',
    # 'cleaned_transformed_ds6.csv',
    # 'cleaned_transformed_ds7.csv',
    'cleaned_transformed_ds8.csv',
    'cleaned_transformed_ds9.csv',
    'cleaned_transformed_ds10.csv',    
]
    
    
lengthData = len(data)
batchSize = int(lengthData / 2)

def perform_time_delay_estimation(file_paths, num_input_signals_list, num_output_signals_list):
    #Defining Parameters    
    sampling_rate = 1000  # The sampling rate between the input and output signals
    degree = 2  # Degree of the polynomial regression Model
    order = 2 # Order of ARX model
    allresults = []
    allresults2 = []
    for file_path, num_input_signals, num_output_signals in zip(file_paths, num_input_signals_list, num_output_signals_list):
        # Read the CSV files from the file_path
        data = pd.read_csv(file_path)
    
        input_columns = data.columns[:num_input_signals]
        output_columns = data.columns[num_input_signals:]

        input_signals = data[input_columns]

        for output_col in output_columns:
            results = {}
            results2 = {}
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

            def lstm_time_delay(input_signals, output_signal, epochs=100, batch_size=batchSize):
                # Normalize input and output data
                scaler_input = MinMaxScaler(feature_range=(0, 1))
                scaler_output = MinMaxScaler(feature_range=(0, 1))
                input_signals_scaled = scaler_input.fit_transform(input_signals)
                output_signal_scaled = scaler_output.fit_transform(output_signal.values.reshape(-1, 1))

                # Reshape input data for LSTM
                input_signals_reshaped = np.reshape(input_signals_scaled, (input_signals_scaled.shape[0], 1, input_signals_scaled.shape[1]))

                # Build and train the LSTM model
                model = Sequential()
                model.add(LSTM(units=100, activation='relu', input_shape=(1, input_signals_scaled.shape[1])))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(input_signals_reshaped, output_signal_scaled, epochs=4, batch_size=batchSize, verbose=0)

                # Predict output using the trained model
                predicted_output_scaled = model.predict(input_signals_reshaped)

                # Inverse transform to get the original scale
                predicted_output = scaler_output.inverse_transform(predicted_output_scaled)

                # Calculate the time delay
                time_delay = find_time_delay(pd.DataFrame(predicted_output), output_signal, sampling_rate)

                return time_delay
            

            def cross_correlation2(signal1, signal2):
                len_signal1 = len(signal1)
                len_signal2 = len(signal2)
                len_cross_corr = len_signal1 + len_signal2 - 1

                # Use fast Fourier transform (FFT) for convolution
                fft_result = np.fft.fft(signal1, len_cross_corr) * np.fft.fft(signal2, len_cross_corr).conj()
                cross_corr2 = np.fft.ifft(fft_result).real

                return cross_corr2

            def find_time_delay2(input_signals, output_signal, sampling_rate):
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
                    cross_corr = cross_correlation2(output_signal_array, input_channel)

                    # Find the index of the maximum correlation value
                    max_corr_index = np.mean(cross_corr)

                    # Calculate the time delay in seconds
                    time_delays[channel] = max_corr_index / sampling_rate

                # Return the time delays for all channels
                return time_delays

            def polynomial_regression_time_delay2(input_signals, output_signal, degree):
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

                    # Calculate the roots of the polynomial
                    roots = np.roots(np.append(coeffs, -1))  # Append -1 to represent the delay term

                    # The time delay is the difference between the two real roots
                    real_roots = roots[np.isreal(roots)]
                    time_delay = np.max(real_roots) - np.min(real_roots)

                    # Store the time delay for the current channel
                    time_delays[channel] = time_delay

                # Return the time delays for all channels
                return time_delays

            def linear_regression_time_delay2(input_signals, output_signal):
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

                # Return the time delays for all channels
                return time_delays

            def arx_modeling_time_delay2(input_signals, output_signal, order):
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

                # Return the time delays for all channels
                return time_delays

            def lstm_time_delay2(input_signals, output_signal, epochs=100, batch_size=batchSize):
                # Normalize input and output data
                scaler_input = MinMaxScaler(feature_range=(0, 1))
                scaler_output = MinMaxScaler(feature_range=(0, 1))
                input_signals_scaled = scaler_input.fit_transform(input_signals)
                output_signal_scaled = scaler_output.fit_transform(output_signal.values.reshape(-1, 1))

                # Reshape input data for LSTM
                input_signals_reshaped = np.reshape(input_signals_scaled, (input_signals_scaled.shape[0], 1, input_signals_scaled.shape[1]))

                # Build and train the LSTM model
                model = Sequential()
                model.add(LSTM(units=100, activation='relu', input_shape=(1, input_signals_scaled.shape[1])))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(input_signals_reshaped, output_signal_scaled, epochs=2, batch_size=batchSize, verbose=0)

                # Predict output using the trained model
                predicted_output_scaled = model.predict(input_signals_reshaped)

                # Inverse transform to get the original scale
                predicted_output = scaler_output.inverse_transform(predicted_output_scaled)

                # Calculate the time delay
                time_delay = find_time_delay(pd.DataFrame(predicted_output), output_signal, sampling_rate)

                return time_delay


            estimated_time_delay_CrossCorr = find_time_delay(input_signals, output_signal, sampling_rate)
            estimated_time_delay_poly = polynomial_regression_time_delay(input_signals, output_signal, degree)
            estimated_Linear_time_delay = linear_regression_time_delay(input_signals, output_signal)
            estimated_ARXtime_delay = arx_modeling_time_delay(input_signals, output_signal, order)
            estimated_LSTM = lstm_time_delay(input_signals, output_signal, epochs=2, batch_size=batchSize)
            
            estimated_time_delay_CrossCorr2 = np.argmax(find_time_delay2(input_signals, output_signal, sampling_rate))
            estimated_time_delay_poly2 = np.argmax(polynomial_regression_time_delay2(input_signals, output_signal, degree))
            estimated_Linear_time_delay2 = np.argmax(linear_regression_time_delay2(input_signals, output_signal))
            estimated_ARXtime_delay2 = np.argmax(arx_modeling_time_delay2(input_signals, output_signal, order))
            estimated_LSTM2 = np.argmax(lstm_time_delay2(input_signals, output_signal, epochs=2, batch_size=batchSize))
         


            # Choose the method with the best time delay
            # delays = [ estimated_time_delay_CrossCorr, estimated_time_delay_poly, estimated_Linear_time_delay, estimated_ARXtime_delay]
            # best_method_index = np.argmin(delays)

            # Choose the method with the best time delay based on the optimization results for Method 1
            optimal_delays1 = [estimated_time_delay_CrossCorr, estimated_time_delay_poly, estimated_Linear_time_delay, estimated_ARXtime_delay]
            best_method_index_opt1 = np.argmax(optimal_delays1)


            # Choose the method with the best time delay based on the optimization results for Method 2
            optimal_delays2 = [estimated_time_delay_CrossCorr2, estimated_time_delay_poly2, estimated_Linear_time_delay2, estimated_ARXtime_delay2]
            best_method_index_opt2 = np.argmax(optimal_delays2)

            # Add LSTM model for Method 1
            lstm_time_delay_opt = lstm_time_delay(input_signals, output_signal)
            optimal_delays1.append(lstm_time_delay_opt)

            # Add LSTM model for Method 2
            lstm_time_delay_opt2 = lstm_time_delay2(input_signals, output_signal)
            optimal_delays2.append(lstm_time_delay_opt2)

            # Input string
            output_col_string = output_col
            # Regular expression pattern to extract the integer
            pattern = r'(\d+)sec'
            # Using re.search to find the first match of the pattern in the string
            match = re.search(pattern, output_col_string)
            integer_value = int(match.group(1))
            target_delay = integer_value

            results['Filename'] = file_path
            results['Input'] = input_columns
            results['Output'] = output_col
            results['CrossCorr1'] = estimated_time_delay_CrossCorr
            results['PolyRegre1'] = estimated_time_delay_poly
            results['LinRegre1'] = estimated_Linear_time_delay
            results['ARXM1'] = estimated_ARXtime_delay
            results['LSTM1'] = lstm_time_delay_opt

            # Find the best method and its time delay
            methods1 = ['CrossCorr1', 'PolyRegre1', 'LinRegre1', 'ARXM1', 'LSTM1']
            time_delays1 = [estimated_time_delay_CrossCorr, estimated_Linear_time_delay,
                           estimated_Linear_time_delay, estimated_ARXtime_delay, lstm_time_delay_opt]

            
            # Find the delay in the list that is closest to the target delay
            # best_delay1 = min(time_delays1, key=lambda x: abs(x - target_delay))
            # best_method_index = time_delays1.index(best_delay1)
            # best_method1 = methods1[best_method_index]            
            # results['BestMethod1'] = best_method1
            # results['BestDelay1'] = best_delay1

            results2['Filename'] = file_path
            results2['Input'] = input_columns
            results2['Output'] = output_col
            results2['CrossCorr2'] = estimated_time_delay_CrossCorr2
            results2['PolyRegre2'] = estimated_time_delay_poly2
            results2['LinRegre2'] = estimated_Linear_time_delay2
            results2['ARXM2'] = estimated_ARXtime_delay2
            results2['LSTM2'] = lstm_time_delay_opt2

            # Find the best method and its time delay
            methods2 = ['CrossCorr2', 'PolyRegre2', 'LinRegre2', 'ARXM2', 'LSTM2']
            time_delays2 = [estimated_time_delay_CrossCorr2, estimated_time_delay_poly2,
                           estimated_Linear_time_delay2, estimated_ARXtime_delay2, lstm_time_delay_opt2]
            # Find the delay in the list that is closest to the target delay
            # best_delay2 = min(time_delays2, key=lambda x: abs(x - target_delay))
            # best_method_index2 = time_delays2.index(best_delay2)
            # best_method2 = methods2[best_method_index2] 
            # results2['BestMethod2'] = best_method2
            # results2['BestDelay2'] = best_delay2

            allresults.append(results)
            allresults2.append(results2)

            print(f"Optimized Cross Correlation Time Delay Method1 : {estimated_time_delay_CrossCorr} seconds")
            print(f"Optimized Polynomial Regression Time Delay Method1: {estimated_time_delay_poly} seconds")
            print(f"Optimized Linear Regression Time Delay Method1: {estimated_Linear_time_delay} seconds")
            print(f"Optimized ARX Modeling Time Delay Method1: {estimated_ARXtime_delay} seconds")
            print(f"Optimized LSTM Time Delay Method1: {lstm_time_delay_opt} seconds")
            print()
            # print(f"Best Optimized Time Delay for Method 1: {[ 'Cross Correlation', 'Polynomial Regression', 'Linear Regression', 'ARX', 'LSTM Method: '][best_method_index]}")
            print()
            print()

            print(f"Optimized Cross Correlation Time Delay Method2: {np.abs(estimated_time_delay_CrossCorr2)} seconds")
            print(f"Optimized Polynomial Regression Time Delay Method 2: {np.abs(estimated_time_delay_poly2)} seconds")
            print(f"Optimized Linear Regression Time Delay Method 2: {np.abs(estimated_Linear_time_delay2)} seconds")
            print(f"Optimized ARX Modeling Time Delay Method 2: {np.abs(estimated_ARXtime_delay2)} seconds")
            print(f"Optimized LSTM Time Delay Method 2: {lstm_time_delay_opt2} seconds")
            print()
            # print(f"Best Optimized Time Delay for Method 2: {[ 'Cross Correlation: ', 'Polynomial Regression: ', 'Linear Regression: ', 'ARX method: ', 'LSTM Method: '][best_method_index2]}")
            print('########################################################################################################################################################')

        
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(allresults)
        df2 = pd.DataFrame(allresults2)
    
        # Save the results to a CSV file
        df.to_csv('Code110.csv', index=False)
        df2.to_csv('Code220.csv', index=False)


def main():

    num_input_signals_list = [ 6, 7, 6]
    num_output_signals_list = [ 168, 126, 168]
    # num_input_signals_list = [4, 4, 4, 8, 8, 5, 6, 6, 7, 6]
    # num_output_signals_list = [42, 42, 42, 168, 168, 147, 168, 168, 126, 168]
    
    result = perform_time_delay_estimation(data, num_input_signals_list, num_output_signals_list)
    return result

if __name__ == "__main__":
    main()
            
