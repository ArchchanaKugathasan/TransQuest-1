import pandas as pd

# Load your dataset
df = pd.read_excel('scaled_dataset.xlsx')

# Calculate the absolute difference between 'average' and 'z_mean'
df['deviation'] = abs(df['average'] - df['z_mean'])

# Sort the dataframe by 'deviation' in descending order and take the top 200 rows
top_200_deviation = df.sort_values(by='deviation', ascending=False).head(200)

# Save these rows to a new Excel file
top_200_deviation.to_excel('top200_deviated_dataset.xlsx', index=False)
