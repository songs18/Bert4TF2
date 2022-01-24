# -*- coding: utf-8 -*-
# @Time    : 10:35 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : multi_head_attention.py
import math

import tensorflow as tf

from nn import utils


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, config=None):
        super(SelfAttention, self).__init__(name=config.name)

        self.size_per_head = int(config.hidden_size / config.num_attention_heads)
        # `query_layer` = [B*F, N*H]
        kernel_init = tf.initializers.TruncatedNormal(stddev=config.initializer_range)

        self.query_layer = tf.keras.layers.Dense(units=config.num_attention_heads * self.size_per_head,
                                                 activation=utils.get_activation(config.act_str),
                                                 name="query", kernel_initializer=kernel_init)

        self.key_layer = tf.keras.layers.Dense(units=config.num_attention_heads * self.size_per_head,
                                               activation=utils.get_activation(config.act_str),
                                               name="key", kernel_initializer=kernel_init)

        self.value_layer = tf.keras.layers.Dense(units=config.num_attention_heads * self.size_per_head,
                                                 activation=utils.get_activation(config.act_str),
                                                 name="value", kernel_initializer=kernel_init)

        #根据training判断dropout
        self.dropout_layer = tf.keras.layers.Dropout(rate=1.0 - config.hidden_dropout_prob)

        self.config=config


    def transpose_for_scores(self, input_tensor, batch_size, num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    def call(self, inputs):
        from_tensor = inputs[0]
        to_tensor = inputs[1]
        attention_mask = inputs[2]
        batch_size = inputs[3]  # optional
        from_seq_length = inputs[4]
        to_seq_length = inputs[5]

        from_shape = utils.get_shape_list(from_tensor, expected_rank=[2, 3])
        to_shape = utils.get_shape_list(to_tensor, expected_rank=[2, 3])

        if len(from_shape) != len(to_shape):
            raise ValueError("The rank of `from_tensor` must match the rank of `to_tensor`.")

        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        elif len(from_shape) == 2:
            if (batch_size is None or from_seq_length is None or to_seq_length is None):
                raise ValueError("When passing in rank 2 tensors to attention_layer, the values " "for `batch_size`, `from_seq_length`, and `to_seq_length` " "must all be specified.")

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        from_tensor_2d = utils.reshape_to_matrix(from_tensor)
        to_tensor_2d = utils.reshape_to_matrix(to_tensor)

        query = self.query_layer(from_tensor_2d)
        # `query_layer` = [B, N, F, H]
        query = self.transpose_for_scores(query, batch_size, self.config.num_attention_heads, from_seq_length, self.size_per_head)

        # `key_layer` = [B*T, N*H]
        key = self.key_layer(to_tensor_2d)
        # `key_layer` = [B, N, T, H]
        key = self.transpose_for_scores(key, batch_size, self.config.num_attention_heads, to_seq_length, self.size_per_head)

        # `value_layer` = [B*T, N*H]
        value = self.value_layer(to_tensor_2d)
        # `value_layer` = [B, T, N, H]
        value = tf.reshape(value, [batch_size, to_seq_length, self.config.num_attention_heads, self.size_per_head])
        # `value_layer` = [B, N, T, H]
        value = tf.transpose(value, [0, 2, 1, 3])

        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self.size_per_head)))

        if attention_mask is not None:
            # `attention_mask` = [B, 1, F, T]
            attention_mask = tf.expand_dims(attention_mask, axis=[1])
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
            attention_scores += adder

        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.softmax(attention_scores)

        attention_probs = self.dropout_layer(attention_probs)

        # `context_layer` = [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value)
        # `context_layer` = [B, F, N, H]
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if self.config.do_return_2d_tensor:
            # `context_layer` = [B*F, N*H]
            context_layer = tf.reshape(context_layer, [batch_size * from_seq_length, self.config.num_attention_heads * self.size_per_head])
        else:
            # `context_layer` = [B, F, N*H]
            context_layer = tf.reshape(context_layer, [batch_size, from_seq_length, self.config.num_attention_heads * self.size_per_head])

        return context_layer


class SelfAttentionOutput(tf.keras.layers.Layer):
    def __init__(self,  config=None):
        super(SelfAttentionOutput, self).__init__(name=config.name)

        kernel_init = tf.initializers.TruncatedNormal(stddev=config.initializer_range)

        self.dense_layer = tf.keras.layers.Dense(units=config.hidden_size, kernel_initializer=kernel_init,name='dense')
        self.dropout_layer = tf.keras.layers.Dropout(1.0 - config.hidden_dropout_prob)
        self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1,name='LayerNorm')

    def call(self, inputs):
        attention_output = self.dense_layer(inputs)
        attention_output = self.dropout_layer(attention_output)
        attention_output=self.norm_layer(inputs+attention_output)

        return attention_output


class SelfAttentionSubLayer(tf.keras.layers.Layer):
    def __init__(self, config=None):
        # python 函数参数默认值
        super(SelfAttentionSubLayer, self).__init__(name=config.name)

        self.self_attention_layer = SelfAttention(config.self_config)
        self.self_attention_output_layer = SelfAttentionOutput(config.output_config)


    def call(self, inputs):
        from_tensor = inputs[0]
        to_tensor = inputs[0]
        attention_mask = inputs[1]

        batch_size = inputs[2]
        seq_length = inputs[3]
        input_width = inputs[4]

        attention_heads = []
        attention_head = self.self_attention_layer(inputs=(from_tensor, to_tensor, attention_mask,batch_size,seq_length,input_width))
        attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
            attention_output = attention_heads[0]
        else:
            attention_output = tf.concat(attention_heads, axis=-1)

        attention_output = self.self_attention_output_layer(attention_output)
        # attention_output = self.norm_layer(attention_output + from_tensor)  # residual

        return attention_output
