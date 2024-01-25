import pandas as pd

# create a dataframe with a column of large numbers
df = pd.read_csv("ds7.csv")

# display the dataframe with scientific notation
print(df)


column_to_convert = 'out5'
df[column_to_convert] = pd.to_numeric(df[column_to_convert], errors='coerce')


# suppress scientific notation by setting float_format
df_formatted = df.applymap(lambda x: '{:.3f}'.format(x) if pd.api.types.is_numeric_dtype(x) else x)

# display the dataframe without scientific notation
df.to_csv('ds7.csv', index=False)