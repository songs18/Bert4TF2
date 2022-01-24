# -*- coding: utf-8 -*-
# @Time    : 10:30 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : embeddings.py
import tensorflow as tf

from nn.embeddings.word_piece_embedding import WordPieceEmbeddingLayer
from nn.embeddings.token_type_embedding import TokenTypeEmbeddingLayer
from nn.embeddings.position_embedding import PositionEmbeddingLayer


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(EmbeddingLayer, self).__init__(name='embeddings')

        self.wordpiece_embedding_layer = WordPieceEmbeddingLayer(config.word_piece_embedding_layer_config)
        self.token_type_embedding_layer = TokenTypeEmbeddingLayer(config.tokentype_embedding_layer_config)
        self.position_embedding_layer = PositionEmbeddingLayer(config.position_embedding_layer_config)

        self.dropout_layer = tf.keras.layers.Dropout(1.0 - config.dropout_prob)
        self.norm_layer = tf.keras.layers.LayerNormalization(axis=-1)

        self.config=config

        # for k, v in config.items():
        #     setattr(self, k, v)

    def call(self, x,mask=None):
        # input_tensor = inputs[0],
        # token_type_ids = inputs[1],

        #todo::从x中索引切片为python变量错误：索引生成tuple形式，直接索引type是tensor，暂且使用直接索引，后续查明原因
        embeddings = self.wordpiece_embedding_layer(x[0])

        if self.config.use_token_type:
            embeddings = self.token_type_embedding_layer((embeddings, x[1]))

        if self.config.use_position_embeddings:
            embeddings = self.position_embedding_layer(embeddings)

        output = self.norm_layer(embeddings)
        output = self.dropout_layer(output)

        return output
