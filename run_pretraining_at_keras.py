# coding=utf-8
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import backpropagation.loss
from backpropagation import optimization
from config.config import Config
from nn.bert import BertModel
from metric.metric import WordMetric, SentenceMetric


def _decode_record(record, name_to_features,vocab_size):
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    features = dict()
    labels = dict()
    for name in list(example.keys()):
        if name=='masked_lm_ids':
            word_label_ids = tf.reshape(example['masked_lm_ids'], [-1])
            word_one_hot_labels = tf.one_hot(word_label_ids, depth=vocab_size, dtype=tf.float32)
            labels[name]=word_one_hot_labels
        elif name=='next_sentence_labels':
            sentence_labels = tf.reshape(example['next_sentence_labels'], [-1])
            sentence_one_hot_labels = tf.one_hot(sentence_labels, depth=2, dtype=tf.float32)
            labels[name]=sentence_one_hot_labels

        # if name in {'masked_lm_ids', 'next_sentence_labels'}:
        #     labels[name] = example[name]
        else:
            features[name] = example[name]



    return (features, labels)


def train_input_fn(input_files, batch_size, max_seq_length, max_predictions_per_seq,vocab_size, num_cpu_threads=4):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights": tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
    }

    raw_dataset = tf.data.TFRecordDataset(input_files)
    parsed_dataset = raw_dataset.map(map_func=lambda record: _decode_record(record, name_to_features,vocab_size)).repeat().shuffle(
        buffer_size=len(input_files)).batch(batch_size=batch_size, drop_remainder=True)
    return parsed_dataset


def eval_input_fn(input_files, batch_size, max_seq_length, max_predictions_per_seq, vocab_size,num_cpu_threads=4):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights": tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
    }

    raw_dataset = tf.data.TFRecordDataset(input_files)
    parsed_dataset = raw_dataset.map(map_func=lambda record: _decode_record(record, name_to_features,vocab_size)).repeat().shuffle(
        buffer_size=len(input_files)).batch(batch_size=batch_size, drop_remainder=True)
    return parsed_dataset


def model_fn(bert_config):
    return BertModel(config=bert_config)


def main():
    config = Config('./config/config.yml')
    bert_config = config.model_config

    running_config = config.running_config
    io_config = running_config.io_config
    device_config = running_config.device_config
    training_config = running_config.training_config
    eval_config = running_config.eval_config

    if not bert_config.do_train and not bert_config.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(io_config.model_dir):
        os.mkdir(io_config.model_dir)

    # =========Train&Eval data========
    input_files = []
    for input_pattern in running_config.io_config.input_files.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    print("*** Input Files ***")
    for input_file in input_files:
        print(" {} ".format(input_file))

    train_spec = tf.estimator.TrainSpec(input_fn=lambda :train_input_fn(input_files=input_files,
                                                                batch_size=training_config.batch_size,
                                                                max_seq_length=training_config.max_seq_length,
                                                                max_predictions_per_seq=training_config.max_predictions_per_seq,
                                                                vocab_size=bert_config.embedding_layer_config.word_piece_embedding_layer_config.vocab_size,
                                                                num_cpu_threads=4), max_steps=training_config.num_train_steps)
    eval_spec = tf.estimator.TrainSpec(input_fn=lambda :eval_input_fn(input_files=input_files,
                                                              batch_size=eval_config.batch_size,
                                                              max_seq_length=eval_config.max_seq_length,
                                                              max_predictions_per_seq=eval_config.max_predictions_per_seq,
                                                              vocab_size=bert_config.embedding_layer_config.word_piece_embedding_layer_config.vocab_size,
                                                              num_cpu_threads=4), max_steps=training_config.num_train_steps)

    # =========model========

    print("*** Features ***")
    # for name in sorted(features.keys()):
    #     print("  name = {}, shape = {}".format(name, features[name].shape))
    #
    # input_ids = features["input_ids"]
    # input_mask = features["input_mask"]
    # segment_ids = features["segment_ids"]
    # masked_lm_positions = features["masked_lm_positions"]
    # masked_lm_ids = features["masked_lm_ids"]
    # masked_lm_weights = features["masked_lm_weights"]
    # next_sentence_labels = features["next_sentence_labels"]
    #
    # is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    # is_eval = (mode == tf.estimator.ModeKeys.EVAL)

    # nn = BertModel(config=bert_config, is_training=is_training, input_ids=input_ids, input_mask=input_mask, token_type_ids=segment_ids, use_one_hot_embeddings=use_one_hot_embeddings)
    model = BertModel(config=bert_config)

    # ========================================================

    # todo::optimizer修改
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'masked_lm_log_probs': backpropagation.loss.WordLoss(),
            'next_sentence_log_probs': backpropagation.loss.SentenceLoss()
        },
        metrics={
            'masked_lm_log_probs': WordMetric(),
            'next_sentence_log_probs': SentenceMetric()
        })
        # loss_weights={
        #     'masked_lm_log_probs': word_label_weights,
        #     'next_sentence_log_probs': sentence_label_weigths
        # })

    estimator = tf.keras.estimator.model_to_estimator(keras_model=model)
    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


if __name__ == "__main__":
    main()
