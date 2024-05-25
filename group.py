import pandas as pd

# # Read the CSV file
# data = pd.read_csv("TimeDelayMethod2.csv")

# # Group the data by the best method
# grouped_data = data.groupby("BestMethod2")

# # Iterate over the groups and create separate CSV files
# for method, group in grouped_data:
#     # Extract the method name and best delays
#     method_name = method
#     best_delays = group["BestDelay2"]
    
#     # Create a new DataFrame for the method and best delays
#     method_data = pd.DataFrame({"Method": method_name, "BestDelay": best_delays})
    
#     # Generate the file name for the method
#     file_name = f"{method_name}_best_delays.csv"
    
#     # Save the method data to a CSV file
#     method_data.to_csv(file_name, index=False)







"""
import csv
import pandas as pd

# Read the CSV file
df = pd.read_csv('Before&AfterOptimization.csv')

# Loop through each column
for col in df.columns:
    # Check if the column contains numeric floating values
    if df[col].dtype == 'float64':
        # Round the values to 3 decimal places
        df[col] = df[col].round(3)

# Save the modified DataFrame back to a CSV file
df.to_csv('Before&AfterOpt.csv', index=False)
"""














# This code Reads two data set and picks some selected columns among them
# Read the CSV file
data = pd.read_csv("Before&AfterOpt.csv")

# Specify the method you want to process
method_name = "Before&AfterOpti2"

# Create a new DataFrame for the method and best delays
method_data = pd.DataFrame({"Filename": data["Filename"], "Output": data["Output"], 
                            "CC1":data["CC1"], "PR1":data["PR1"], "LR1":data["LR1"],
                            "AR1":data["AR1"], "LM1":data["LM1"], "CC2":data["CC2"], "PR2":data["PR2"], "LR2":data["LR2"],
                            "AR2":data["AR2"], "LM2":data["LM2"] })

# Generate the file name for the method
file_name = f"{method_name}.csv"

# Save the method data to a CSV file
method_data.to_csv(file_name, index=False)
