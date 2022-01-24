# coding=utf-8
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from backpropagation import loss
from backpropagation import optimization
from config.config import Config
from nn.bert import BertModel
from metric import metric
import numpy as np


def _decode_record(record, name_to_features, vocab_size):
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
        if name == 'masked_lm_ids':
            # word_label_ids = tf.reshape(example['masked_lm_ids'], [-1])
            # word_one_hot_labels = tf.one_hot(word_label_ids, depth=vocab_size, dtype=tf.float32)
            # labels[name]=word_one_hot_labels
            labels[name] = example[name]
        elif name == 'next_sentence_labels':
            # sentence_labels = tf.reshape(example['next_sentence_labels'], [-1])
            # sentence_one_hot_labels = tf.one_hot(sentence_labels, depth=2, dtype=tf.float32)
            # labels[name]=sentence_one_hot_labels
            labels[name] = example[name]

        # if name in {'masked_lm_ids', 'next_sentence_labels'}:
        #     labels[name] = example[name]
        else:
            features[name] = example[name]

    return (features, labels)


def train_input_fn(input_files, batch_size, max_seq_length, max_predictions_per_seq, vocab_size, num_cpu_threads=4):
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
    parsed_dataset = raw_dataset.map(map_func=lambda record: _decode_record(record, name_to_features, vocab_size)).repeat().shuffle(
        buffer_size=len(input_files)).batch(batch_size=batch_size, drop_remainder=True)
    return parsed_dataset


def eval_input_fn(input_files, batch_size, max_seq_length, max_predictions_per_seq, vocab_size, num_cpu_threads=4):
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
    parsed_dataset = raw_dataset.map(map_func=lambda record: _decode_record(record, name_to_features, vocab_size)).repeat().shuffle(
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

    train_dataset = train_input_fn(input_files=input_files, batch_size=training_config.batch_size,
                                   max_seq_length=training_config.max_seq_length,
                                   max_predictions_per_seq=training_config.max_predictions_per_seq,
                                   vocab_size=bert_config.embedding_layer_config.word_piece_embedding_layer_config.vocab_size,
                                   num_cpu_threads=4)
    # eval_dataset = eval_input_fn(input_files=input_files, batch_size=eval_config.batch_size,
    #                              max_seq_length=eval_config.max_seq_length,
    #                              max_predictions_per_seq=eval_config.max_predictions_per_seq,
    #                              vocab_size=bert_config.embedding_layer_config.word_piece_embedding_layer_config.vocab_size,
    #                              num_cpu_threads=4)

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

    # model.build(input_shape= {
    #     "input_ids": [training_config.batch_size,training_config.max_seq_length],
    #     "input_mask": [training_config.batch_size,training_config.max_seq_length],
    #     "segment_ids": [training_config.batch_size,training_config.max_seq_length],
    #     "masked_lm_positions": [training_config.batch_size,training_config.max_predictions_per_seq],
    #     "masked_lm_weights": [training_config.batch_size,training_config.max_predictions_per_seq]
    # })
    toy_data = {
        "input_ids": np.ones(shape=[training_config.batch_size, training_config.max_seq_length], dtype=np.int32),
        "input_mask": np.ones(shape=[training_config.batch_size, training_config.max_seq_length], dtype=np.float32),
        "segment_ids": np.ones(shape=[training_config.batch_size, training_config.max_seq_length], dtype=np.int32),
        "masked_lm_positions": np.ones(shape=[training_config.batch_size, training_config.max_predictions_per_seq], dtype=np.int32),
        "masked_lm_weights": np.ones(shape=[training_config.batch_size, training_config.max_predictions_per_seq], dtype=np.float32)
    }

    model(toy_data)
    print(model.summary())

    # print(model.embedding_layer.wordpiece_embedding_layer.embedding_table[:5,:5])

    print('original')
    model.restore_from_checkpoint(running_config.io_config.init_checkpoint)
    # print(model.embedding_layer.wordpiece_embedding_layer.embedding_table[:5,:5])


    '''
    print('restore')
    model.restore_from_checkpoint(running_config.io_config.init_checkpoint)
    model.restore_from_checkpoint('./files/checkpoints/my/bert.ckpt')
    print(model.embedding_layer.wordpiece_embedding_layer.embedding_table[:5,:5])

    assert 1==2

    # model.save_weights_as_checkpoint()
    # assert 1==2

    print('=-' * 25)
    print('Random Init')
    tvars = model.trainable_variables
    for var in tvars:
        print(var.name, var.shape)

    print('=-' * 25)
    print('Official')

    vars = tf.train.list_variables('./files/checkpoints/tiny/bert_model.ckpt')
    for var in vars:
        name = var[0]
        shape = var[1]
        print(var)

    print('Saved')
    vars = tf.train.list_variables('./files/checkpoints/my/bert.ckpt')
    for var in vars:
        name = var[0]
        shape = var[1]
        print(var)
    assert 1==2
    assert 1==2


    print('=-' * 25)
    print('Official')
    print(model.layers)

    print('=-' * 25)
    print('Layers...')

    layer_name_to_weight_name=dict()
    for layer in model.layers:
        print(layer.name)
        layer_name_to_weight_name[layer.name]=list()

        t_weights=layer.trainable_weights
        for w in t_weights:
            print(w.name)
            layer_name_to_weight_name[layer.name].append(w.name)
        print()
    print(str(layer_name_to_weight_name))
    assert 1 == 2
    '''

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    loss_obj_masked_lm = loss.WordLoss()
    loss_obj_next_sentence = loss.SentenceLoss()

    metrics_obj_masked_lm = metric.WordMetric()
    metrics_obj_next_sentence = metric.SentenceMetric()

    is_train = bert_config.do_train == True
    is_eval = bert_config.do_eval == True

    if is_train:
        for iter_index,(features, labels) in enumerate(train_dataset):
            labels_masked_lm = labels['masked_lm_ids']
            word_label_ids = tf.reshape(labels_masked_lm, [-1])
            word_one_hot_labels = tf.one_hot(word_label_ids, depth=bert_config.embedding_layer_config.word_piece_embedding_layer_config.vocab_size, dtype=tf.float32)

            labels_next_sentence = labels['next_sentence_labels']
            sentence_labels = tf.reshape(labels_next_sentence, [-1])
            sentence_one_hot_labels = tf.one_hot(sentence_labels, depth=2, dtype=tf.float32)

            with tf.GradientTape() as tape:
                masked_lm_log_probs, next_sentence_log_probs = model(features, training=True)

                loss_masked_lm = loss_obj_masked_lm(y_true=word_one_hot_labels, y_pred=masked_lm_log_probs)
                loss_next_sentence = loss_obj_next_sentence(y_true=sentence_one_hot_labels, y_pred=next_sentence_log_probs)

                total_loss = loss_masked_lm + loss_next_sentence

            tvars = model.trainable_variables
            gradients = tape.gradient(total_loss, sources=tvars)
            optimizer.apply_gradients(grads_and_vars=zip(gradients, tvars))

            metrics_obj_masked_lm.update_state(y_true=labels_masked_lm, y_pred=masked_lm_log_probs)
            metrics_obj_next_sentence.update_state(y_true=labels_next_sentence, y_pred=next_sentence_log_probs)

            if iter_index%2==0:
                print('iter_index:{}, masked_lm_metric: {:2f},next_sentence_metric: {:>2f}'.format(iter_index,metrics_obj_masked_lm.result(),metrics_obj_next_sentence.result()))



if __name__ == "__main__":
    main()
