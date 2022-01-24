# -*- coding: utf-8 -*-
# @Time    : 15:42 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : utils.py
import collections
import re

import numpy as np
import six
import tensorflow as tf


def assert_rank(tensor, expected_rank, name=None):
    # if name is None:
    #     name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        # scope_name = tf.get_variable_scope().name
        # raise ValueError("For the tensor `%s` in scope `%s`, the actual rank " "`%d` (shape = %s) is not equal to the expected rank `%s`" % (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
        raise ValueError("For the tensor `%s`, the actual rank " "`%d` (shape = %s) is not equal to the expected rank `%s`" % (name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    # if name is None: name = tensor.name
    if expected_rank is not None: assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    # 无dynamic shape直接返回,todo::命名non-static -> dynamic更直观
    dynamic_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            dynamic_indexes.append(index)

    if not dynamic_indexes:
        return shape

    # 有dynamic shape，将动态的转换为对应的tensor表示，即None
    dyn_shape = tf.shape(tensor)
    for index in dynamic_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" % (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


# def gelu(x):
#     cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
#     return x * cdf


def get_activation(activation_string):
    if not isinstance(activation_string, six.string_types): return activation_string
    if not activation_string: return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return tf.nn.gelu
        # return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    # Model trainable variables
    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None: name = m.group(1)

        name_to_variable[name] = var

    #checkpoint variables
    init_vars = tf.train.list_variables(init_checkpoint)
    assignment_map = collections.OrderedDict()
    initialized_variable_names = {}
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable: continue

        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def create_initializer(initializer_range=0.02):
    return tf.initializers.TruncatedNormal(stddev=initializer_range)


'''
def get_masked_lm_output(bert_config, input_tensor, output_weights, positions, label_ids, label_weights):
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(input_tensor, units=bert_config.hidden_size, activation=modeling.get_activation(bert_config.hidden_act), kernel_initializer=modeling.create_initializer(bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        output_bias = tf.get_variable("output_bias", shape=[bert_config.vocab_size], initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable("output_weights", shape=[2, bert_config.hidden_size], initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable("output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)
'''

# def gather_indexes(sequence_tensor, positions):
#     sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
#     batch_size = sequence_shape[0]
#     seq_length = sequence_shape[1]
#     width = sequence_shape[2]
#
#     flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
#     flat_positions = tf.reshape(positions + flat_offsets, [-1])
#     flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
#     output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
#     return output_tensor

def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

    return output_tensor
