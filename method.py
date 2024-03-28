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



# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # Read the CSV file
# df = pd.read_csv('TimeDelayMethod1.csv')


# # Assuming df is your DataFrame containing 1245 rows
# # iterations = df['Index'][:100]


# mcrossCorr1 = df['CrossCorr1'][1199:1240]
# polyMethod1 = df['PolyRegre1'][1199:1240]
# linReg1 = df['LinRegre1'][1199:1240]
# aRXM1 = df['ARXM1'][1199:1240]
# lSTM1 = df['LSTM1'][1199:1240]
# iterations = np.arange(1, len(mcrossCorr1) + 1)
# iterations2 = np.arange(1, len(polyMethod1) + 1)
# iterations3 = np.arange(1, len(linReg1) + 1)
# iterations4 = np.arange(1, len(aRXM1) + 1)
# iterations5 = np.arange(1, len(lSTM1) + 1)


# # plt.plot(iterations, mcrossCorr1)
# plt.plot(iterations, mcrossCorr1)
# plt.title('Cross Correlation Time Delay Plot')
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()

# plt.plot(iterations2, polyMethod1)
# plt.title('Polynomial Regression Method Time Delay Plot')
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()

# plt.plot(iterations3, linReg1)
# plt.title('Linear Regression Method Time Delay Plot')
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()

# plt.plot(iterations4, aRXM1)
# plt.title('ARX Method Time Delay Plot')
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()

# plt.plot(iterations5, lSTM1)
# plt.title('LSTM  Time Delay Plot')
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()






















































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







import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('CrossCorr2_best_delays.csv')
df2 = pd.read_csv('PolyRegre2_best_delays.csv')
df3 = pd.read_csv('LinRegre2_best_delays.csv')
df4 = pd.read_csv('ARXM2_best_delays.csv')
# df5 = pd.read_csv('LSTM2_best_delays.csv')


# Assuming df is your DataFrame containing 1245 rows
# iterations = df['Index'][:100]


# mcrossCorr1 = df['BestDelay']
# polyMethod1 = df2['BestDelay']
# linReg1 = df3['BestDelay']
# aRXM1 = df4['BestDelay']
# # lSTM1 = df5['BestDelay']

# iterations = list(range(1, len(mcrossCorr1) + 1))
# iterations2 = list(range(1, len(polyMethod1) + 1))
# iterations3 = np.arange(1, len(linReg1) + 1)
# iterations4 = np.arange(1, len(aRXM1) + 1)
# # iterations5 = np.arange(1, len(lSTM1) + 1)


# # plt.plot(iterations, mcrossCorr1)
# plt.plot(iterations, mcrossCorr1, color='blue', linestyle='-', linewidth=1.5)
# plt.title('Cross Correlation Time Delay Plot')
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()

# plt.plot(iterations2, polyMethod1)
# plt.title('Polynomial Regression Time Delay Plot')
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()

# plt.plot(iterations3, linReg1)
# plt.title('Linear Regression Time Delay Plot')
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()

# plt.plot(iterations4, aRXM1)
# plt.title('ARXM Time Delay Plot')
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()

# plt.plot(iterations5, lSTM1)
# plt.title('LSTM Time Delay Plot')
# plt.xlabel('signals')
# plt.ylabel('Delay (s)')
# plt.show()


# species = ("Method1", "Method2")
# percentage_delays = {
#     'Cross Corr': (25.1, 45.7),
#     'Poly Regre.': (39.3, 28.65),
#     'Linear Regre.': (18.7, 17.43),
#     'ARXM': (14.3, 8.77),
#     'LSTM': (2.6 ,0)
# }

# x = np.arange(len(species))  # the label locations
# width = 0.18  # the width of the bars
# multiplier = 0

# fig, ax = plt.subplots(layout='constrained')

# for attribute, measurement in percentage_delays.items():
#     offset = width * multiplier
#     rects = ax.bar(x + offset, measurement, width, label=attribute)
#     ax.bar_label(rects, padding=3)
#     multiplier += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Percentage (%)')
# # ax.set_title('Performance of each solution for accurate time delay estimation')
# ax.set_xticks(x + width, species)
# ax.legend(loc='upper left', ncols=2)
# ax.set_ylim(0, 100)

# plt.show()




