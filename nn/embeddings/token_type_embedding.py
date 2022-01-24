# -*- coding: utf-8 -*-
# @Time    : 10:41 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : b_token_type_embedding.py
import tensorflow as tf

from nn import utils


class TokenTypeEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(TokenTypeEmbeddingLayer, self).__init__(name='token_type_embeddings', dtype=tf.float32)

        self.config = config

    def build(self, input_shape):
        width = input_shape[0][-1]
        token_type_table_init = tf.initializers.TruncatedNormal(stddev=self.config.initializer_range)
        self.token_type_table = self.add_weight(name=self.config.name, shape=[self.config.vocab_size, width],
                                                dtype=tf.float32, initializer=token_type_table_init, trainable=True)
        # self.token_type_table = tf.Variable(initial_value=token_type_table_init(shape=[self.vocab_size, width], dtype='float32'),
        #                                     name=self.name,
        #                                     trainable=True)

    def call(self, inputs):
        previous_embedding = inputs[0]
        token_type_ids = inputs[1]

        input_shape = utils.get_shape_list(previous_embedding, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if" "`use_token_type` is True.")
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])

        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.config.vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_table)

        token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])

        output = previous_embedding + token_type_embeddings

        return output
