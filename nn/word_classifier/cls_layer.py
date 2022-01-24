# -*- coding: utf-8 -*-
# @Time    : 20:41 2021/4/27 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : cls_layer.py

import tensorflow as tf
from nn import utils


class ClassifierLayer(tf.keras.layers.Layer):
    def __init__(self, config=None):
        super(ClassifierLayer, self).__init__(name=config.name)

        #todo::这里name_scope不起作用,最后的output_bias没有layer的name前缀
        # with tf.name_scope('predictions'):
        kernel_init = tf.initializers.TruncatedNormal(stddev=config.initializer_range)
        self.dense_layer = tf.keras.layers.Dense(units=config.hidden_size,
                                                 activation=utils.get_activation(config.act_str),
                                                 kernel_initializer=kernel_init, name='transform')
        self.norm_layer = tf.keras.layers.LayerNormalization(name='LayerNorm')

        self.config=config

    def build(self,input_shape):
        #在__init__中写tf.Variable变量名没有前缀，是output_bias:0
        # self.output_bias = tf.Variable(initial_value=tf.zeros(shape=[config.vocab_size]), trainable=True, name='output_bias')
        #在__init__中写self.add_weight变量名没有前缀，还是output_bias:0
        # self.output_bias = self.add_weight(name='output_bias',shape=[config.vocab_size],trainable=True)
        # 在build中写add_weight有前缀，是bert/cls/predictions/output_bias
        # 或者写tf.keras.layer可以自动跟踪
        self.output_bias = self.add_weight(name='output_bias',shape=[self.config.vocab_size],trainable=True)

    def call(self, x,mask=None):
        input_tensors=x[0]
        output_weigths=x[1]

        input_tensors = self.dense_layer(input_tensors)
        input_tensors = self.norm_layer(input_tensors)

        logits = tf.matmul(input_tensors, output_weigths, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)

        return logits