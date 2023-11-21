import os
import shutil
os.environ['TRANSFORMERS_CACHE'] = '/vol/research/Archchana/.cache'
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from examples.sentence_level.wmt_2023.common.util.draw import draw_scatterplot, print_stat
from examples.sentence_level.wmt_2023.common.util.normalizer import fit, un_fit
from examples.sentence_level.wmt_2023.common.util.postprocess import format_submission
from examples.sentence_level.wmt_2023.common.util.reader import read_annotated_file, read_test_file
from examples.sentence_level.wmt_2023.multilingual.monotransquest_config import TEMP_DIRECTORY, GOOGLE_DRIVE, DRIVE_FILE_ID, MODEL_NAME, \
    monotransquest_config, MODEL_TYPE, SEED, RESULT_FILE, RESULT_IMAGE, SUBMISSION_FILE
from transquest.algo.sentence_level.monotransquest.evaluation import pearson_corr, spearman_corr
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Your saved model path
#SAVED_MODEL_PATH = 'temp/outputs'
SAVED_MODEL_PATH = '/user/HS501/as04746/Archchana/Experiments/WMT23/result_10_Aug/xlmv/temp/outputs'

# Load the trained model
model = MonoTransQuestModel(MODEL_TYPE, SAVED_MODEL_PATH, num_labels=1, use_cuda=torch.cuda.is_available())

def infer_z_mean(original_text, translated_text):
    # Prepare the input for the model
    test_sentence_pairs = [[original_text, translated_text]]
    
    # Predict using the model
    predictions, _ = model.predict(test_sentence_pairs)
    
    # Check if predictions is an array and extract the first element, otherwise use it directly
    if isinstance(predictions, np.ndarray) and predictions.ndim > 0:
        z_mean_predicted = predictions[0]
    else:
        z_mean_predicted = predictions

    return z_mean_predicted



# Example usage
original_text = " My hat’s off to so many others doing the same. "
translated_text =  "என் தொப்பியைப்போல இன்னும் பலர் இதைச் செய்கிறார்கள்."
predicted_z_mean = infer_z_mean(original_text, translated_text)
print(f"Predicted z_mean before inverse_transform: {predicted_z_mean}")


##### NORMALIZE THE PREDICTED VALUE using the min max of the true value column##### 
####NOTE:even after normalizing the same score is provided, means the value outputed from the predicted model
# is already normalized on ####################

# Load the dataset from an Excel file, to filter the true z_mean column
df = pd.read_excel('ErrorAnalysis/extracted_fr_ERR_Anlys/enta_for_err_analysis.xlsx')

# Find the min and max from the True z_mean column
z_mean_min = df['z_mean'].min()
z_mean_max = df['z_mean'].max()

# Initialize the MinMaxScaler with the found min and max
scaler = MinMaxScaler(feature_range=(z_mean_min, z_mean_max))

# Fit the scaler on the z_mean column
scaler.fit(df[['z_mean']])

# Scale a single value
def scale_value(value):
    return scaler.transform([[value]])[0][0]

# Example of scaling a single value
value_to_scale = predicted_z_mean  # Replace with your value
scaled_value = scale_value(value_to_scale)
print("Scaled value:", scaled_value)