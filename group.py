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






# Read the CSV file
data = pd.read_csv("TimeDelayMethod1.csv")

# Specify the method you want to process
method_name = "CrossCorr1"

# Create a new DataFrame for the method and best delays
method_data = pd.DataFrame({"Filename": data["Filename"], "Output": data["Output"], "Delay":data["CrossCorr1"]})

# Generate the file name for the method
file_name = f"{method_name}_delays_After_Optimization.csv"

# Save the method data to a CSV file
method_data.to_csv(file_name, index=False)