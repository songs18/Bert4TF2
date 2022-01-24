# -*- coding: utf-8 -*-
# @Time    : 10:45 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : z_multi_transformer_layers.py
import tensorflow as tf

from nn import utils
from nn.transformer.single_transformer_layer import TransformerLayer


class TransformerLayers(tf.keras.layers.Layer):
    def __init__(self, config=None):
        super(TransformerLayers, self).__init__(name=config.name)

        self.layers = [TransformerLayer(name='layer_{}'.format(i), config=config.single_layer_config) for i in
                       range(config.num_hidden_layers)]

        # todo::字典查询会降低速度
        self.config = config

    def call(self, inputs):
        # if self.config.hidden_size % self.config.num_attention_heads != 0:
        #     raise ValueError("The hidden size (%d) is not a multiple of the number of attention " "heads (%d)" % (self.config.hidden_size,
        #                                                                                                           self.config.num_attention_heads))
        input_tensor = inputs[0]
        attention_mask=inputs[1]

        # attention_head_size = int(self.config.hidden_size / self.config.num_attention_heads)
        input_shape = utils.get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]

        # if input_width != self.config.hidden_size:
        #     raise ValueError("The width of the input tensor (%d) != hidden size (%d)" % (input_width, self.config.hidden_size))

        prev_output = utils.reshape_to_matrix(input_tensor)

        layer_outputs = []
        for layer in self.layers:
            prev_output = layer(inputs=(prev_output,attention_mask,batch_size,seq_length,input_width))
            layer_outputs.append(prev_output)

        if self.config.do_return_all_layers:
            final_outputs = []
            for layer_output in layer_outputs:
                final_output = utils.reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = utils.reshape_from_matrix(prev_output, input_shape)
            return final_output
