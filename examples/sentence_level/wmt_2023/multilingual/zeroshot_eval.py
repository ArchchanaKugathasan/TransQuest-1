#this file evaluates from the trained model

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

# Your saved model path
SAVED_MODEL_PATH = 'temp/outputs'

# Zero-shot language pair for evaluation
zero_shot_language = {
    "EN-MR": [
        "data/WMT23/dev.enmr.df.short.tsv",
        #"data/our_test_data_with_gold_labels/test.enmr.df.short.tsv"
        #for marati only
        "data/our_test_data_with_gold_labels/enmr_corrected_data.tsv"
    ]
}

# Load the trained model
model = MonoTransQuestModel(MODEL_TYPE, SAVED_MODEL_PATH, num_labels=1, use_cuda=torch.cuda.is_available())

# Evaluation on the zero-shot language pair
for language, paths in zero_shot_language.items():
    dev_path, test_path = paths

    dev = read_annotated_file(dev_path)
    test = read_test_file(test_path)

    dev = dev[['original', 'translation', 'z_mean']]
    test = test[['index', 'original', 'translation']]

    index = test['index'].to_list()
    dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()
    test = test.rename(columns={'original': 'text_a', 'translation': 'text_b'}).dropna()

    test_sentance_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

    dev = fit(dev, 'labels')

    # Evaluation on dev set
    result, model_outputs, wrong_predictions = model.eval_model(dev)
    dev['predictions'] = model_outputs

    # Predictions on test set
    predictions, _ = model.predict(test_sentance_pairs)
    test['predictions'] = predictions

    # Post-processing and result visualization
    dev = un_fit(dev, 'labels')
    dev = un_fit(dev, 'predictions')
    dev.to_csv(os.path.join(TEMP_DIRECTORY, f"result_{language}.tsv"), header=True, sep='\t', index=False, encoding='utf-8')
    test = un_fit(test, 'predictions')
    draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, f"scatterplot_{language}.png"), language)
    print_stat(dev, 'labels', 'predictions')
    format_submission(df=test, index=index, language_pair=language.lower(), method="TransQuest", path=os.path.join(TEMP_DIRECTORY, f"submission_{language}.tsv"))

#   format_submission(df=test, index=index, language_pair=language.lower(), method="TransQuest",
#                   path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE.split(".")[0] + "_" + language + "." + SUBMISSION_FILE.split(".")[1]))

print("Evaluation done!")
