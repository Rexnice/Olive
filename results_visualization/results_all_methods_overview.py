import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Load the dataset
data = pd.read_csv('Modified_Before&AfterOpti3.csv')

# Function to extract delay from the 'Output' column
def extract_delay(output):
    match = re.search(r'(\d+sec)', output)
    return match.group(0) if match else None

# Convert delay from string to numerical format in seconds
data['Delay'] = data['Output'].apply(extract_delay)
data['Delay_Numeric'] = data['Delay'].str.extract(r'(\d+)').astype(float)  # Extract number and convert to float

# Function to categorize data based on presence of noise and/or filter
def categorize_data(output):
    if 'noise' in output and 'filt' in output:
        return 'Both Noise and Filter'
    elif 'noise' in output:
        return 'Noise Only'
    elif 'filt' in output:
        return 'Filtered Only'
    else:
        return 'None'

# Apply the categorization function
data['Condition'] = data['Output'].apply(categorize_data)

# Create a new dataframe to store the deviation of each method's result from the delay
deviation_data = data[['Filename', 'Output', 'Delay_Numeric', 'Condition']].copy()
for method in ['CC1', 'PR1', 'LR1', 'AR1', 'LM1', 'CC2', 'PR2', 'LR2', 'AR2', 'LM2']:
    deviation_data[f'{method}_Deviation'] = abs(data[method] - data['Delay_Numeric'])

# Melt the deviation data for visualization
melted_deviation_data = pd.melt(deviation_data, id_vars=['Filename', 'Output', 'Delay_Numeric', 'Condition'], 
                                value_vars=[f'{method}_Deviation' for method in ['CC1', 'PR1', 'LR1', 'AR1', 'LM1', 'CC2', 'PR2', 'LR2', 'AR2', 'LM2']],
                                var_name='Method', value_name='Deviation')

# Function to plot the deviation for each method across different conditions
def plot_deviation(condition):
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=melted_deviation_data[melted_deviation_data['Condition'] == condition], x='Method', y='Deviation', hue='Method')
    plt.title(f'Method Deviation from Delay under {condition}')
    plt.ylabel('Deviation from Delay (seconds)')
    plt.xlabel('Method')
    plt.xticks(rotation=45)
    plt.legend(title='Method', loc='upper right')
    plt.tight_layout()
    plt.show()

# Plot deviation for each condition
for cond in ['Noise Only', 'Filtered Only', 'Both Noise and Filter', 'None']:
    plot_deviation(cond)
