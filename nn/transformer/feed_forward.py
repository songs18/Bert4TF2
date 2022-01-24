# -*- coding: utf-8 -*-
# @Time    : 10:36 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : feed_forward.py
import tensorflow as tf


class IntermediateSubLayer(tf.keras.layers.Layer):
    def __init__(self, config=None):
        super(IntermediateSubLayer, self).__init__(name=config.name)

        intermediate_kernel_init = tf.initializers.TruncatedNormal(stddev=config.initializer_range)
        self.intermediate_layer = tf.keras.layers.Dense(units=config.intermediate_size,
                                                        activation=config.act_str,
                                                        kernel_initializer=intermediate_kernel_init,name='dense')

    def call(self, inputs):
        intermediate_output = self.intermediate_layer(inputs)
        return intermediate_output

class OutputSubLayer(tf.keras.layers.Layer):
    def __init__(self, config=None):
        super(OutputSubLayer, self).__init__(name=config.name)

        output_kernel_init = tf.initializers.TruncatedNormal(stddev=config.initializer_range)
        self.output_dense = tf.keras.layers.Dense(units=config.hidden_size,
                                                  kernel_initializer=output_kernel_init,name='dense')
        self.output_dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

        self.output_norm = tf.keras.layers.LayerNormalization(axis=-1,name='LayerNorm')

    def call(self, x):
        inputs=x[0]
        intermediate_output =x[1]

        output = self.output_dense(intermediate_output)
        output = self.output_dropout(output)

        # output = self.output_norm(intermediate_output + output)
        output = self.output_norm(inputs + output)

        return output

'''
class FeedForawrdSubLayer(tf.keras.layers.Layer):
    def __init__(self, config=None):
        super(FeedForawrdSubLayer, self).__init__(name=config.name)

        intermediate_config = config.intermediate_config
        #todo::name_scope不起作用，这里使用layer组合解决
        with tf.name_scope(name=intermediate_config.name):
            intermediate_kernel_init = tf.initializers.TruncatedNormal(stddev=intermediate_config.initializer_range)
            self.intermediate_layer = tf.keras.layers.Dense(units=intermediate_config.intermediate_size,
                                                            activation=intermediate_config.act_str,
                                                            kernel_initializer=intermediate_kernel_init)

        output_config = config.output_config
        with tf.name_scope(name=output_config.name):
            output_kernel_init = tf.initializers.TruncatedNormal(stddev=output_config.initializer_range)
            self.output_dense = tf.keras.layers.Dense(units=output_config.hidden_size,
                                                      kernel_initializer=output_kernel_init)
            self.output_dropout = tf.keras.layers.Dropout(rate=output_config.hidden_dropout_prob)
            self.output_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, inputs):
        intermediate_output = self.intermediate_layer(inputs)

        output = self.output_dense(intermediate_output)
        output = self.output_dropout(output)

        # output = self.output_norm(intermediate_output + output)
        output = self.output_norm(inputs + output)

        return output
'''