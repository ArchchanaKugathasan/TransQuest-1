import pandas as pd

# Load the datasets
df_error_analysis = pd.read_excel('errorAnalysis.xlsx')
df_test_short = pd.read_csv('test.enta.df.short.tsv', sep='\t')  # Adjust the separator if necessary

# Initialize an empty list to store the matched rows
matched_rows = []

# Iterate over the rows in the error analysis dataframe
for index, row in df_error_analysis.iterrows():
    original_text = row['original']
    
    # Find the matching row in the test short dataframe
    matching_row = df_test_short[df_test_short['original'] == original_text]
    
    # If there is a matching row, combine the data
    if not matching_row.empty:
        # Assuming there is only one match, we take the first
        matching_row = matching_row.iloc[0]
        
        # Create a new row combining data from both dataframes
        new_row = {**row, 'z_mean': matching_row['z_mean']}
        
        # Append the new row dictionary to the matched_rows list
        matched_rows.append(new_row)

# Convert the list of dictionaries to a dataframe
matched_df = pd.DataFrame(matched_rows)

# Save the combined data to a new Excel file
matched_df.to_excel('extracted_fr_ERR_Anlys/enta_for_err_analysis.xlsx', index=False, engine='openpyxl')

print("Data extraction and file save completed.")
