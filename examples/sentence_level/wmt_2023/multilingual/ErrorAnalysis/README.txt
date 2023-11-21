extract_similar_rows.py
Used to extract the similar rows from the predicted and the gold label test dataset 

NormalizePredictedScore.py
scale the predicted value from MonoTransQuest predicted files, to the min and max value of the true z_mean value.
So we can do a comparison for error analysis. creates the scaled file 'scaled_dataset.xlsx'

extract_most_deviated_values.py
extracts the 200 most deviated values(from scaled_dataset.xlsx) on average column(ensembled score) comparing to z_mean value. 
This is to find the most error containing rows.