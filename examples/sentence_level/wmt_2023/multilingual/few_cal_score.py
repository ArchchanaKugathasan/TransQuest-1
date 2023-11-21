####################CALCULATING SCORES ONE BY ONE FOR eACH FILE######################
# import pandas as pd
# import numpy as np
# from scipy.stats import pearsonr, spearmanr

# # Provided methods
# def pearson_corr(preds, labels):
#     return pearsonr(preds, labels)[0]

# def spearman_corr(preds, labels):
#     return spearmanr(preds, labels)[0]

# def rmse(preds, labels):
#     return np.sqrt(((np.asarray(preds, dtype=np.float32) - np.asarray(labels, dtype=np.float32)) ** 2).mean())

# # Load predicted values from predictions_EN-GU.txt
# with open("temp/data/predictions_EN-TA.txt", "r") as f:
#     preds = [float(line.strip().split()[3]) for line in f.readlines()]

# # Load labels from test.enta.df.short.tsv, skip the header and fetch the 7th column values
# labels_df = pd.read_csv("data/few_shot/fewshot_200/test.enta.df.short.tsv", sep="\t", skiprows=1, header=None)
# #labels_df = pd.read_csv("data/few_shot/fewshot_100/enta_corrected_data.tsv", sep="\t", skiprows=1, header=None)
# labels = labels_df.iloc[:, 6].tolist()



# # Ensure lengths match (not strictly necessary but a good sanity check)
# assert len(preds) == len(labels), "Length of predictions and labels do not match!"

# # Calculate scores
# pearson_score = pearson_corr(preds, labels)
# spearman_score = spearman_corr(preds, labels)
# rmse_score = rmse(preds, labels)

# # Write scores to a text file
# with open("data/predicted_scores_full/enta_scores.txt", "w") as f:
#     f.write(f"Pearson Correlation: {pearson_score}\n")
#     f.write(f"Spearman Correlation: {spearman_score}\n")
#     #f.write(f"RMSE: {rmse_score}\n")

# print("Scores written to scores.txt")

# # Save preds and labels to a file
# with open("data/predicted_scores_full/enta_preds_and_labels.txt", "w") as f:
#     f.write("Predicted\tTrue_Label\n")
#     for p, l in zip(preds, labels):
#         f.write(f"{p}\t{l}\n")

# print("Predicted values and labels written to preds_and_labels.txt")



####################CALCULATING SCORES FOR ALL FILES AT ONCE######################
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Provided methods
def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]

def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]

def rmse(preds, labels):
    return np.sqrt(((np.asarray(preds, dtype=np.float32) - np.asarray(labels, dtype=np.float32)) ** 2).mean())

def process_language_pair(language_pair):
    # Load predicted values
    with open(f"temp/data/predictions_EN-{language_pair.upper()}.txt", "r") as f:
        preds = [float(line.strip().split()[3]) for line in f.readlines()]

    # Load labels
    labels_df = pd.read_csv(f"data/few_shot/fewshot_200/test.en{language_pair}.df.short.tsv", sep="\t", skiprows=1, header=None)
    labels = labels_df.iloc[:, 6].tolist()

    # Ensure lengths match
    assert len(preds) == len(labels), f"Length of predictions and labels do not match for {language_pair.upper()}!"

    # Calculate scores
    pearson_score = pearson_corr(preds, labels)
    spearman_score = spearman_corr(preds, labels)
    rmse_score = rmse(preds, labels)

    # Write scores to a text file
    with open(f"data/predicted_scores_full/en{language_pair}_scores.txt", "w") as f:
        f.write(f"Pearson Correlation: {pearson_score}\n")
        f.write(f"Spearman Correlation: {spearman_score}\n")
        f.write(f"RMSE: {rmse_score}\n")

    print(f"Scores written to en{language_pair}_scores.txt")

    # Save preds and labels to a file
    with open(f"data/predicted_scores_full/en{language_pair}_preds_and_labels.txt", "w") as f:
        f.write("Predicted\tTrue_Label\n")
        for p, l in zip(preds, labels):
            f.write(f"{p}\t{l}\n")

    print(f"Predicted values and labels written to en{language_pair}_preds_and_labels.txt")

# List of language pairs
language_pairs = ['gu', 'hi', 'mr', 'ta', 'te']

# Process each language pair
for pair in language_pairs:
    process_language_pair(pair)
