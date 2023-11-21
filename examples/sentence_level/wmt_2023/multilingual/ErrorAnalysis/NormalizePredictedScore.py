import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset from an Excel file
df = pd.read_excel('extracted_fr_ERR_Anlys/enta_for_err_analysis.xlsx')

# Find the min and max from the True z_mean column
z_mean_min = df['z_mean'].min()
z_mean_max = df['z_mean'].max()

# Initialize the MinMaxScaler with the found min and max
scaler = MinMaxScaler(feature_range=(z_mean_min, z_mean_max))

# Columns to scale
columns_to_scale = ['infoxlm_pred', 'xlmr_large', 'xlmv_base','average']

# Fit the scaler on the data
scaler.fit(df[columns_to_scale])

# Transform the data and update the dataframe
df[columns_to_scale] = scaler.transform(df[columns_to_scale])

# Save the new dataframe to an Excel file
df.to_excel('scaled_dataset.xlsx', index=False)
