# -*- coding: utf-8 -*-
# @Time    : 22:10 2021/4/24 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : write_TFRecord.py
import tensorflow as tf
import collections
# import  b_data.a_tokenization as tokenization
from data import tokenization

# 函数功能混杂，拆解
def write_instance_to_example_files(instances, tokenizer, max_seq_length, max_predictions_per_seq, output_files):
    writers = create_writers(output_files)

    writer_index = 0
    for (instance_index, instance) in enumerate(instances):
        features, tf_example = create_tf_examples(instance, max_predictions_per_seq, max_seq_length, tokenizer)
        writers[writer_index].write(tf_example.SerializeToString())

        writer_index = instance_index % len(writers)  # 轮转均匀写文件

        if instance_index < 5:
            print_instance(features, instance, instance_index)

    close_writers(writers)

    print("Wrote {} total instances".format(len(instances)))

def create_writers(output_files):
    writers = [tf.io.TFRecordWriter(output_file) for output_file in output_files]
    return writers


def close_writers(writers):
    for writer in writers:
        writer.close()


def create_tf_examples(instance, max_predictions_per_seq, max_seq_length, tokenizer):
    input_ids, input_mask, segment_ids = pad_raw_input(instance, max_seq_length, tokenizer)

    masked_lm_ids, masked_lm_positions, masked_lm_weights = pad_raw_masked_lm(instance, max_predictions_per_seq, tokenizer)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    return features, tf_example


def pad_raw_input(instance, max_seq_length, tokenizer):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)

    assert len(input_ids) <= max_seq_length
    # padding
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def pad_raw_masked_lm(instance, max_predictions_per_seq, tokenizer):
    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)
    # padding
    while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
        masked_lm_weights.append(0.0)

    return masked_lm_ids, masked_lm_positions, masked_lm_weights


def print_instance(features, instance, instance_index):
    print("*** Example{} ***".format(instance_index))
    print("tokens: %s" % " ".join([tokenization.printable_text(x) for x in instance.tokens]))
    for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
            values = feature.int64_list.value
        elif feature.float_list.value:
            values = feature.float_list.value
        print("%s: %s" % (feature_name, " ".join([str(x) for x in values])))
    print()


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

