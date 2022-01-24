# -*- coding: utf-8 -*-
# @Time    : 21:05 2021/4/28 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : next_sentence_layer.py
import tensorflow as tf


class NextSentenceLayer(tf.keras.layers.Layer):
    def __init__(self,  config=None):
        super(NextSentenceLayer, self).__init__(name=config.name)

        self.config=config
    def build(self,input_shape):
        kernel_init = tf.initializers.TruncatedNormal(stddev=self.config.initializer_range)

        self.w=self.add_weight(name='output_weights',shape=[2,self.config.hidden_size],
                               dtype=tf.float32,initializer=kernel_init)
        self.b=self.add_weight(name='output_bias',shape=[2],
                               dtype=tf.float32,initializer=tf.keras.initializers.Zeros())

    def call(self, inputs):
        logits=tf.matmul(inputs,self.w,transpose_b=True)
        logits=tf.nn.bias_add(logits,self.b)

        log_probs = tf.nn.log_softmax(logits, axis=-1)

        return log_probs

