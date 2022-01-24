# -*- coding: utf-8 -*-
# @Time    : 10:46 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : pooling_layer.py
import tensorflow as tf


class PoolingLayer(tf.keras.layers.Layer):
    def __init__(self, config=None):
        super(PoolingLayer, self).__init__(name=config.name)

        kernel_init = tf.initializers.TruncatedNormal(stddev=config.initializer_range)
        self.pool_layer = tf.keras.layers.Dense(units=config.hidden_size, activation=config.act_str,
                                                kernel_initializer=kernel_init)

    def call(self, inputs):
        first_token_tensor = tf.squeeze(inputs[:, 0:1, :], axis=1)
        first_token_tensor = self.pool_layer(first_token_tensor)

        return first_token_tensor
