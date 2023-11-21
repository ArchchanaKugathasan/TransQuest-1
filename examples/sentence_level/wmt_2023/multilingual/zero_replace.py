# import pandas as pd

# # Read the dataset
# df = pd.read_csv('data/few_shot/few_shot_data/test.enmr.df.short.tsv')

# # Convert non-numeric values in the 'z_mean' column to NaN and then replace NaN with 0
# df['z_mean'] = pd.to_numeric(df['z_mean'], errors='coerce').fillna(0)

# # Save the updated dataframe back to csv (if needed)
# df.to_csv('data/few_shot/few_shot_data/test.enmr.df.short.tsv', index=False)

filename = 'data/our_test_data_with_gold_labels/test.enmr.df.short.tsv'

with open(filename, 'r') as f:
    lines = f.readlines()
    
    # Check header for the expected number of fields
    expected_fields = len(lines[0].split('\t'))
    
    for line_num, line in enumerate(lines):
        fields = len(line.split('\t'))
        
        if fields != expected_fields:
            print(f"Line {line_num + 1} has {fields} fields instead of {expected_fields}.")
            print(line)

# Once you've fixed the issues or if there are none:
import pandas as pd

df = pd.read_csv(filename, delimiter='\t')

# Check and replace non-numeric z_mean values
def replace_non_numeric(val):
    try:
        float(val)
        return val
    except ValueError:
        return 0

df['z_mean'] = df['z_mean'].apply(replace_non_numeric)

# Save the corrected DataFrame if you want
df.to_csv('corrected_data.tsv', sep='\t', index=False)
