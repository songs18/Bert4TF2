# -*- coding: utf-8 -*-
# @Time    : 10:42 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : c_position_embedding.py
import tensorflow as tf

from nn import utils


class PositionEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(PositionEmbeddingLayer, self).__init__(name='position_embeddings', dtype=tf.float32)

        self.config = config

    def build(self, input_shape):
        width = input_shape[-1]
        full_position_embeddings_init = tf.initializers.TruncatedNormal(stddev=self.config.initializer_range)
        self.token_type_table = self.add_weight(name=self.config.name,
                                                shape=[self.config.max_position_embeddings, width],
                                                dtype=tf.float32, initializer=full_position_embeddings_init)
        # self.token_type_table = tf.Variable(initial_value=full_position_embeddings_init(shape=[self.max_position_embeddings, width], dtype='float32'),
        #                                     name=self.name,
        #                                     trainable=True)

    def call(self, inputs):
        previous_embedding = inputs

        input_shape = utils.get_shape_list(previous_embedding, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        if seq_length > self.config.max_position_embeddings:
            raise ValueError('seq_length is greater than max_position_embeddings')

        position_embeddings = tf.slice(self.token_type_table, [0, 0], [seq_length, -1])

        num_dims = len(previous_embedding.shape.as_list())
        position_broadcast_shape = [1 for _ in range(num_dims - 2)]
        position_broadcast_shape.extend([seq_length, width])

        position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)

        output = previous_embedding + position_embeddings

        return output
