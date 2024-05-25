import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load your dataset
data = pd.read_csv('Modified_Before&AfterOpti2.csv')
# Extract delay and convert to numeric
data['Delay'] = data['Output'].apply(lambda x: re.search(r'(\d+sec)', x).group(0) if re.search(r'(\d+sec)', x) else None)
data['Delay_Numeric'] = data['Delay'].str.extract(r'(\d+)').astype(float)

# Categorize data by conditions
data['Condition'] = data['Output'].apply(lambda x: 'Both Noise and Filter' if 'noise' in x and 'filt' in x else
                                          ('Noise Only' if 'noise' in x else
                                           ('Filtered Only' if 'filt' in x else 'None')))

# Calculate deviations
method_pairs = [('CC1', 'CC2'), ('PR1', 'PR2'), ('LR1', 'LR2'), ('AR1', 'AR2'), ('LM1', 'LM2')]
for method1, method2 in method_pairs:
    data[f'{method1}_Deviation'] = abs(data[method1] - data['Delay_Numeric'])
    data[f'{method2}_Deviation'] = abs(data[method2] - data['Delay_Numeric'])

# Prepare data for plotting
melted_data = pd.melt(data, id_vars=['Filename', 'Output', 'Delay_Numeric', 'Condition'], 
                      value_vars=[f'{m1}_Deviation' for m1, m2 in method_pairs] + [f'{m2}_Deviation' for m1, m2 in method_pairs],
                      var_name='Method', value_name='Deviation')

# Plotting
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 18), sharey=True)
axes = axes.flatten()

for idx, (method1, method2) in enumerate(method_pairs):
    sns.boxplot(ax=axes[idx], data=melted_data[melted_data['Method'].isin([f'{method1}_Deviation', f'{method2}_Deviation'])],
                x='Condition', y='Deviation', hue='Method')
    axes[idx].set_title(f'{method1} vs {method2} Deviation Comparison')
    axes[idx].set_xlabel('Condition')
    axes[idx].set_ylabel('Deviation from Delay (seconds)')
    axes[idx].legend(title='Method')

plt.tight_layout()
plt.show()