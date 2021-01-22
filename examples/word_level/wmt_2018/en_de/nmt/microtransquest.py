import os
import shutil

from sklearn.model_selection import train_test_split

from examples.word_level.common.util import reader
from examples.word_level.wmt_2018.en_de.nmt.microtransquest_config import TRAIN_PATH, TRAIN_SOURCE_FILE, \
    TRAIN_SOURCE_TAGS_FILE, \
    TRAIN_TARGET_FILE, \
    TRAIN_TARGET_TAGS_FLE, MODEL_TYPE, MODEL_NAME, microtransquest_config, TEST_PATH, TEST_SOURCE_FILE, \
    TEST_TARGET_FILE, TEMP_DIRECTORY, TEST_SOURCE_TAGS_FILE, SEED, TEST_TARGET_TAGS_FILE, TEST_TARGET_GAPS_FILE
from transquest.algo.word_level.microtransquest.format import prepare_data, prepare_testdata, post_process
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

raw_train_df = reader(TRAIN_PATH, microtransquest_config, TRAIN_SOURCE_FILE, TRAIN_TARGET_FILE, TRAIN_SOURCE_TAGS_FILE,
                      TRAIN_TARGET_TAGS_FLE)
raw_test_df = reader(TEST_PATH, microtransquest_config, TEST_SOURCE_FILE, TEST_TARGET_FILE)

test_sentences = prepare_testdata(raw_test_df, args=microtransquest_config)

fold_sources_tags = []
fold_targets_tags = []

for i in range(microtransquest_config["n_fold"]):

    if os.path.exists(microtransquest_config['output_dir']) and os.path.isdir(microtransquest_config['output_dir']):
        shutil.rmtree(microtransquest_config['output_dir'])

    if microtransquest_config["evaluate_during_training"]:
        raw_train, raw_eval = train_test_split(raw_train_df, test_size=0.1, random_state=SEED * i)
        train_df = prepare_data(raw_train, args=microtransquest_config)
        eval_df = prepare_data(raw_eval, args=microtransquest_config)
        tags = train_df['labels'].unique().tolist()
        model = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=microtransquest_config)
        model.train_model(train_df, eval_df=eval_df)
        model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=tags,
                                     args=microtransquest_config)

    else:
        train_df = prepare_data(raw_train_df, args=microtransquest_config)
        tags = train_df['labels'].unique().tolist()
        model = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=microtransquest_config)
        model.train_model(train_df)

    predicted_labels, raw_predictions = model.predict(test_sentences, split_on_space=True)
    sources_tags, targets_tags = post_process(predicted_labels, test_sentences, args=microtransquest_config)
    fold_sources_tags.append(sources_tags)
    fold_targets_tags.append(targets_tags)

source_predictions = []
for sentence_id in range(len(test_sentences)):
    majority_prediction = []
    predictions = []
    for fold_prediction in fold_sources_tags:
        predictions.append(fold_prediction[sentence_id])

    sentence_length = len(predictions[0])

    for word_id in range(sentence_length):
        word_prediction = []
        for prediction in predictions:
            word_prediction.append(prediction[word_id])
        majority_prediction.append(max(set(word_prediction), key=word_prediction.count))
    source_predictions.append(majority_prediction)

target_predictions = []
for sentence_id in range(len(test_sentences)):
    majority_prediction = []
    predictions = []
    for fold_prediction in fold_targets_tags:
        predictions.append(fold_prediction[sentence_id])

    sentence_length = len(predictions[0])

    for word_id in range(sentence_length):
        word_prediction = []
        for prediction in predictions:
            word_prediction.append(prediction[word_id])
        majority_prediction.append(max(set(word_prediction), key=word_prediction.count))
    target_predictions.append(majority_prediction)

test_source_sentences = raw_test_df[microtransquest_config["source_column"]].tolist()
test_target_sentences = raw_test_df[microtransquest_config["target_column"]].tolist()

with open(os.path.join(TEMP_DIRECTORY, TEST_SOURCE_TAGS_FILE), 'w') as f:
    for sentence_id, (test_source_sentence, source_prediction) in enumerate(
            zip(test_source_sentences, source_predictions)):
        words = test_source_sentence.split()
        word_predictions = source_prediction.split()
        for word_id, (word, word_prediction) in enumerate(zip(words, word_predictions)):
            f.write("MicroTransQuest" + "\t" + "source" + "\t" +
                    str(sentence_id) + "\t" + str(word_id) + "\t"
                    + word + "\t" + word_prediction + '\n')

with open(os.path.join(TEMP_DIRECTORY, TEST_TARGET_TAGS_FILE), 'w') as target_f, open(
        os.path.join(TEMP_DIRECTORY, TEST_TARGET_GAPS_FILE), 'w') as gap_f:
    for sentence_id, (test_sentence, target_prediction) in enumerate(zip(test_sentences, target_predictions)):
        target_sentence = test_sentence.split("[SEP]")[1]
        words = target_sentence.split()
        word_predictions = target_prediction.split()
        gap_index = 0
        word_index = 0
        for word, word_prediction in zip(words, word_predictions):
            if word == microtransquest_config["tag"]:
                gap_f.write("MicroTransQuest" + "\t" + "gap" + "\t" +
                            str(sentence_id) + "\t" + str(gap_index) + "\t"
                            + "gap" + "\t" + word_prediction + '\n')
                gap_index += 1
            else:
                target_f.write("MicroTransQuest" + "\t" + "mt" + "\t" +
                               str(sentence_id) + "\t" + str(word_index) + "\t"
                               + word + "\t" + word_prediction + '\n')
                word_index += 1
