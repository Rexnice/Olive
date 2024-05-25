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

# import re

# # Input string
# output_col = 'out1_12sec_shift_filt'
# # Regular expression pattern to extract the integer
# pattern = r'(\d+)sec'

# # Using re.search to find the first match of the pattern in the string
# match = re.search(pattern, output_col)
# integer_value = int(match.group(1))
# print(integer_value)

# delay = [10, 3, 4, 6]
# target_delay = integer_value

# # Find the delay in the list that is closest to the target delay
# best_delay = min(delay, key=lambda x: abs(x - target_delay))

# print("Best delay:", best_delay)



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('Modified_Before&AfterOpti2.csv')


mcrossCorr1 = df['CC1'][420:462]
Polyreg = df['PR1'][420:462]
linreg = df['LR1'][420:462]
arxm = df['AR1'][420:462]
lstm = df['LM1'][420:462]

iterations = np.arange(1, len(mcrossCorr1) + 1)
iterations2 = np.arange(1, len(Polyreg) + 1)
iterations3 = np.arange(1, len(linreg) + 1)
iterations4 = np.arange(1, len(arxm) + 1)
iterations5 = np.arange(1, len(lstm) + 1)


# plt.plot(iterations, mcrossCorr1)
plt.plot(iterations, mcrossCorr1 , linestyle="-", color="black")
plt.title("Cross correlation method Before Optimization (Output_signal_8)")
plt.xlabel('signals')
plt.ylabel('Delay (s)')
plt.show()


# plt.plot(iterations, mcrossCorr1)
plt.plot(iterations2, Polyreg , linestyle="-", color="black")
plt.title("Polynomial Regression method Before Optimization (Output_signal_8)")
plt.xlabel('signals')
plt.ylabel('Delay (s)')
plt.show()


# plt.plot(iterations, mcrossCorr1)
plt.plot(iterations3, linreg , linestyle="-", color="black")
plt.title("Linear Regression method Before Optimization (Output_signal_8)")
plt.xlabel('signals')
plt.ylabel('Delay (s)')
plt.show()


# plt.plot(iterations, mcrossCorr1)
plt.plot(iterations4, arxm , linestyle="-", color="black")
plt.title("ARX method Before Optimization (Output_signal_8)")
plt.xlabel('signals')
plt.ylabel('Delay (s)')
plt.show()


# plt.plot(iterations, mcrossCorr1)
plt.plot(iterations5, lstm , linestyle="-", color="black")
plt.title("LSTM method Before Optimization (Output_signal_8)")
plt.xlabel('signals')
plt.ylabel('Delay (s)')
plt.show()


"""
# plt.plot(iterations, mcrossCorr1)
plt.plot(iterations01, mcrossCorr0 , linestyle="-", color="black")
plt.title("Cross correlation method After Optimization (Output_signal_3)")
plt.xlabel('signals')
plt.ylabel('Delay (s)')
plt.show()


# plt.plot(iterations, mcrossCorr1)
plt.plot(iterations02, Polyreg01 , linestyle="-", color="black")
plt.title("Polynomial Regression method After Optimization (Output_signal_3)")
plt.xlabel('signals')
plt.ylabel('Delay (s)')
plt.show()


# plt.plot(iterations, mcrossCorr1)
plt.plot(iterations03, linreg02 , linestyle="-", color="black")
plt.title("Linear Regression method After Optimization (Output_signal_3)")
plt.xlabel('signals')
plt.ylabel('Delay (s)')
plt.show()


# plt.plot(iterations, mcrossCorr1)
plt.plot(iterations04, arxm03 , linestyle="-", color="black")
plt.title("ARX method After Optimization (Output_signal_3)")
plt.xlabel('signals')
plt.ylabel('Delay (s)')
plt.show()


# plt.plot(iterations, mcrossCorr1)
plt.plot(iterations05, lstm04 , linestyle="-", color="black")
plt.title("LSTM method After Optimization (Output_signal_3)")
plt.xlabel('signals')
plt.ylabel('Delay (s)')
plt.show()
"""
















































# import pandas as pd
# import matplotlib.pyplot as plt

# # Read the CSV file
# data = pd.read_csv("TimeDelayMethod1.csv")

# # Extract the relevant columns
# methods = data["BestMethod1"]
# delays = data["BestDelay1"]

# # Plotting the relationships between methods and delays
# plt.figure(figsize=(10, 6))
# plt.plot(methods, delays, marker="o", linestyle="-", color="b")
# plt.xlabel("Method")
# plt.ylabel("Time Delay")
# plt.title("Relationship between Methods and Time Delays")
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.show()







# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # Read the CSV file
# df = pd.read_csv('CrossCorr1_delays_Before_Optimization.csv')
# df2 = pd.read_csv('PolyRegre1_delays_Before_Optimization.csv')
# df3 = pd.read_csv('LinRegre1_delays_Before_Optimization.csv')
# df4 = pd.read_csv('ARXM1_delays_Before_Optimization.csv')
# df5 = pd.read_csv('LSTM1_delays_Before_Optimization.csv')


# # Assuming df is your DataFrame containing 1245 rows
# # iterations = df['Index'][:100]


# mcrossCorr1 = df['Delay']
# polyMethod1 = df2['Delay']
# linReg1 = df3['Delay']
# aRXM1 = df4['Delay']
# lSTM1 = df5['Delay']

# iterations = list(range(1, len(mcrossCorr1) + 1))
# iterations2 = list(range(1, len(polyMethod1) + 1))
# iterations3 = np.arange(1, len(linReg1) + 1)
# iterations4 = np.arange(1, len(aRXM1) + 1)
# iterations5 = np.arange(1, len(lSTM1) + 1)


# # plt.plot(iterations, mcrossCorr1)
# plt.plot(iterations, mcrossCorr1, color='black', linestyle='-', linewidth=1.5)
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()


# plt.plot(iterations2, polyMethod1 , color='black', linestyle='-', linewidth=1.5)
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()

# plt.plot(iterations3, linReg1, color='black', linestyle='-', linewidth=1.5)
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()

# plt.plot(iterations4, aRXM1, color='black', linestyle='-', linewidth=1.5)
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()

# plt.plot(iterations5, lSTM1, color='black', linestyle='-', linewidth=1.5)
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()





