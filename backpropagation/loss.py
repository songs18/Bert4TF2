# -*- coding: utf-8 -*-
# @Time    : 10:01 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : loss.py
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

class WordLoss(tf.keras.losses.Loss):
    def __init__(self,reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE):
        super(WordLoss, self).__init__(reduction=reduction)

    def call(self,y_true,y_pred):
        assert y_true.shape==y_pred.shape

        one_hot_labels=y_true
        log_probs=y_pred

        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        return per_example_loss


class SentenceLoss(tf.keras.losses.Loss):
    def __init__(self,reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE):
        super(SentenceLoss, self).__init__(reduction=reduction)

    def call(self,y_true,y_pred):
        assert y_true.shape==y_pred.shape

        one_hot_labels=y_true
        log_probs=y_pred

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

        return per_example_loss
