# -*- coding: utf-8 -*-
# @Time    : 10:01 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : metri.py
import tensorflow as tf


class WordMetric(tf.keras.metrics.Metric):
    def __init__(self, name="word_metric", **kwargs):
        super(WordMetric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        masked_lm_ids = y_true
        masked_lm_log_probs = y_pred
        masked_lm_weights = sample_weight

        masked_lm_log_probs = tf.reshape(masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
        # masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)

        # masked_lm_accuracy = tf.compat.v1.metrics.accuracy(labels=masked_lm_ids, predictions=masked_lm_predictions,
        #                                                    weights=masked_lm_weights)

        masked_lm_predictions = tf.reshape(tf.argmax(masked_lm_log_probs, axis=1), shape=(-1, 1))
        values = tf.cast(masked_lm_ids, "int32") == tf.cast(masked_lm_predictions, "int32")
        values = tf.cast(values, "float32")
        if masked_lm_weights is not None:
            sample_weight = tf.cast(masked_lm_weights, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)


class SentenceMetric(tf.keras.metrics.Metric):
    def __init__(self, name="sentence_metric", **kwargs):
        super(SentenceMetric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        next_sentence_labels = y_true
        next_sentence_log_probs = y_pred
        masked_lm_weights = sample_weight

        next_sentence_log_probs = tf.reshape(next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(next_sentence_log_probs, axis=-1, output_type=tf.int32)

        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])

        values = tf.cast(next_sentence_predictions, "int32") == tf.cast(next_sentence_labels, "int32")
        values = tf.cast(values, "float32")
        if masked_lm_weights is not None:
            sample_weight = tf.cast(masked_lm_weights, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)
