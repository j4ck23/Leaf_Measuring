import pandas as pd
import os
file_path = "Block 2.csv"
df = pd.read_csv(file_path)
for plant in df['Plant'].unique():
    mask = df['Plant'] == plant
    df.loc[mask, 'Height (cm)'] = df.loc[mask, 'Height (cm)'].fillna(method='ffill')
    df.loc[mask, 'Width (cm)'] = df.loc[mask, 'Width (cm)'].fillna(method='ffill')

# Save output
output_path = 'Block2_filled.csv'
df.to_csv(output_path, index=False)

print('Filled missing Height and Width values and saved to Block2_filled.csv')
