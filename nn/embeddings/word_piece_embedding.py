# -*- coding: utf-8 -*-
# @Time    : 10:40 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : a_word_piece_embedding.py
import tensorflow as tf

from nn import utils


class WordPieceEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(WordPieceEmbeddingLayer, self).__init__(name='word_embeddings', dtype=tf.float32)

        self.confg=config

        # self.get_ouput = None
        # if config.use_one_hot_embeddings:
        #     self.get_ouput = self.get_output_from_one_hot
        # else:
        #     self.get_ouput = self.get_output_from_gather

    def build(self, input_shape):
        embedding_table_init = tf.initializers.TruncatedNormal(stddev=self.confg.initializer_range)
        self.embedding_table=self.add_weight(name=self.confg.name,shape=[self.confg.vocab_size,self.confg.embedding_size],
                                             dtype=tf.float32,initializer=embedding_table_init,trainable=True)
        # self.embedding_table = tf.Variable(initial_value=embedding_table_init(shape=[self.vocab, self.embedding_size], dtype='float32'),
        #                                    name=self.name, trainable=True)


    def call(self, inputs):
        # print(inputs)
        input_ids = inputs

        if input_ids.shape.ndims == 2:
            input_ids = tf.expand_dims(input_ids, axis=[-1])

        flat_input_ids = tf.reshape(input_ids, [-1])

        if self.confg.use_one_hot_embeddings:
            one_hot_input_ids = tf.one_hot(flat_input_ids, depth=self.vocab_size)
            output = tf.matmul(one_hot_input_ids, self.embedding_table)
        else:
            output = tf.gather(self.embedding_table, flat_input_ids)

        input_shape = utils.get_shape_list(input_ids)

        output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * self.confg.embedding_size])

        return output
