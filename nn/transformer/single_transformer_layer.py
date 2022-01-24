# -*- coding: utf-8 -*-
# @Time    : 10:37 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : transformer.py
import tensorflow as tf

# from nn.transformer.feed_forward import FeedForawrdSubLayer
from nn.transformer.feed_forward import IntermediateSubLayer,OutputSubLayer
from nn.transformer.multi_head_attention import SelfAttentionSubLayer


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, name='layer_0', config=None):
        super(TransformerLayer, self).__init__(name=name)

        self.attention_layer = SelfAttentionSubLayer(config=config.selfattention_sublayer_config)

        # self.attention_dropout_layer = tf.keras.layers.Dropout(1.0 - config.dropout_prob)
        # self.attention_norm_layer = tf.keras.layers.LayerNormalization(axis=-1)

        # =============================================================

        # self.feedforward_sublayer = FeedForawrdSubLayer(config=config.feeforward_sublayer_config)
        self.intermediate_sublayer=IntermediateSubLayer(config.feeforward_sublayer_config.intermediate_config)
        self.output_sublayer=OutputSubLayer(config.feeforward_sublayer_config.output_config)

        # self.intermediate_dropout_layer = tf.keras.layers.Dropout(1.0 - config.dropout_prob)
        # self.intermediate_norm_layer = tf.keras.layers.LayerNormalization(axis=-1)

        # self.config = config

    def call(self, inputs):
        attention_output = self.attention_layer(inputs)
        # attention_output = self.attention_dropout_layer(attention_output)
        # attention_output = self.attention_norm_layer(attention_output + inputs)  # residual

        # output = self.feedforward_sublayer(attention_output)
        intermediate_output=self.intermediate_sublayer(attention_output)
        output=self.output_sublayer((attention_output,intermediate_output))
        # intermediate_output = self.intermediate_layer(attention_output)
        # intermediate_output = self.intermediate_output_layer(intermediate_output)
        # output = self.intermediate_dropout_layer(intermediate_output)
        # output = self.intermediate_norm_layer(output + attention_output)  # residual

        return output
