import pandas as pd
import os

# List of file paths
file_paths = [
    "data/WMT23/train.enmr.df.short.tsv",
    "data/WMT23/dev.enmr.df.short.tsv",
    "data/WMT23/test.enmr.df.short.tsv"
]

# Directory where you want to save the new files
output_dir = "data/few_shot/fewshot_200"  # Replace with your desired output directory

# Ensure the directory exists or create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the train data outside the loop as we'll need it for the test file
train_df = pd.read_csv("data/WMT23/train.enmr.df.short.tsv", sep="\t", index_col=0)

for file_path in file_paths:
    # Load the data file
    df = pd.read_csv(file_path, sep="\t", index_col=0)
    
    # If it's the test file, extract the last 50 rows from the train data
    if "test.enmr.df.short.tsv" in file_path:
        df_subset = train_df.tail(200)
    else:
        # For other files, extract the first 50 rows
        df_subset = df.head(200)
    
    # Save to the new directory with the same file name
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    df_subset.to_csv(output_path, sep="\t", index=True)

print("Files have been saved in the new directory!")
