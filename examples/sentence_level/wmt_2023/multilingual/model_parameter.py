import os
import sys
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

import torch

# Load the parameters from the 'pytorch_model.bin' file
state_dict = torch.load("/user/HS501/as04746/Archchana/Experiments/WMT23/result_10_Aug/infoxlm/temp/outputs/pytorch_model.bin")


def count_parameters(state_dict):
    #return sum(p.numel() for p in state_dict.values() if p.requires_grad)
    return sum(p.numel() for p in state_dict.values())

total_params = count_parameters(state_dict)
print(f"Total trainable parameters: {total_params}")

state_dict_size = sys.getsizeof(state_dict)
print(f"Total size: {state_dict_size}")







