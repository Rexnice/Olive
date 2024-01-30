# import pandas as pd

# # create a dataframe with a column of large numbers
# df = pd.read_csv("ds4.csv")

# # display the dataframe with scientific notation
# print(df)


# column_to_convert = 'out1'
# # df = df.astype({column_to_convert:'float'})
# # df[column_to_convert] = pd.to_numeric(df[column_to_convert].str.replace(',', ''), errors='coerce')
# # df['out5'] = df['out5'] / 1000000000.0
# df[column_to_convert] = df[column_to_convert].round(2)


# # suppress scientific notation by setting float_format
# df_formatted = df.applymap(lambda x: '{:.3f}'.format(x) if pd.api.types.is_numeric_dtype(x) else x)

# # display the dataframe without scientific notation
# df.to_csv('ds4.csv', index=False)




import pandas as pd

# Read the dataset
data = pd.read_csv('ds5.csv')

# Check for patterns or regular intervals in the data
# This is a simplistic example and may need customization based on your data characteristics
if len(data) > 1:
    # Calculate the average time difference between consecutive samples
    time_diff = (data.index[1] - data.index[0]) / len(data)
    
    # Estimate sampling rate
    sampling_rate = 1 / time_diff
    print(f"Estimated Sampling Rate: {sampling_rate} samples per second")
else:
    print("Dataset is too small. Unable to estimate sampling rate.")





