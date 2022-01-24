# coding=utf-8
"""The main BERT nn and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nn import utils
import warnings
# from nn.embeddings.z_embedding_layer import EmbeddingLayer
# from c_dnn.a_modeling.b_transformer.z_multi_transformer_layers import TransformerLayers
# from c_dnn.a_modeling.c_pooling.pooling_layer import PoolingLayer
#
# from c_dnn.a_modeling.d_word_classifier.cls_layer import ClassifierLayer
# from c_dnn.a_modeling.e_next_sentence.next_sentence_layer import NextSentenceLayer
from nn.embeddings.embedding_layer import EmbeddingLayer
from nn.transformer.multi_transformer_layers import TransformerLayers
from nn.pooling.pooling_layer import PoolingLayer
from nn.word_classifier.cls_layer import ClassifierLayer
from nn.next_sentence.next_sentence_layer import NextSentenceLayer
import json


# from nn.word_classifier.cls_layer import ClassifierLayer
# from nn.next_sentence.next_sentence_layer import NextSentenceLayer


class AttentionMask(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionMask, self).__init__(name='attention_mask', dtype=tf.float32)

    # todo::call函数签名是（x,?），后续修改
    def call(self, inputs):
        from_tensor = inputs[0]
        to_mask = inputs[1]

        from_shape = utils.get_shape_list(from_tensor, expected_rank=[2, 3])
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

        to_shape = utils.get_shape_list(to_mask, expected_rank=2)
        to_seq_length = to_shape[1]
        to_mask = tf.cast(tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
        mask = broadcast_ones * to_mask

        return mask


class BertModel(tf.keras.Model):
    def __init__(self, config):
        super(BertModel, self).__init__(name=config.name)

        self.embedding_layer = EmbeddingLayer(config.embedding_layer_config)
        self.attention_mask_layer = AttentionMask()
        self.transformer_layers = TransformerLayers(config=config.transformer_layers_config)
        self.pooling_layer = PoolingLayer(config=config.pooling_layer_config)

        self.output_cls_layer = ClassifierLayer(config=config.word_cls_layer_config)
        self.output_next_sentence_layer = NextSentenceLayer(config=config.next_sentence_layer_config)

    def compute_sequence_output(self, input_ids, input_mask, segment_ids):
        # input_ids = input_ids
        # input_mask = input_mask
        # token_type_ids = segment_ids

        input_shape = utils.get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if segment_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        embeddings = self.embedding_layer(x=(input_ids, segment_ids))
        attention_mask = self.attention_mask_layer(inputs=(input_ids, input_mask))

        all_encoder_layers = self.transformer_layers((embeddings, attention_mask))
        sequence_output = all_encoder_layers[-1]

        return sequence_output

    def compute_pool_output(self, sequence_output):

        pool_output = self.pooling_layer(sequence_output)

        return pool_output

    def call(self, inputs, training=False):
        # for k,v in inputs.items():
        #     print(k,type(v),v.shape)
        # print(inputs)
        # assert 1==2
        # todo::全局变量名统一
        input_ids = inputs["input_ids"]
        input_mask = inputs["input_mask"]
        segment_ids = inputs["segment_ids"]

        masked_lm_positions = inputs["masked_lm_positions"]
        # masked_lm_ids = inputs["masked_lm_ids"]
        masked_lm_weights = inputs["masked_lm_weights"]
        # next_sentence_labels = inputs["next_sentence_labels"]

        sequence_output = self.compute_sequence_output(input_ids, input_mask, segment_ids)

        embedding_table = self.embedding_layer.wordpiece_embedding_layer.embedding_table
        masked_lm_log_probs = self.get_masked_lm_output(sequence_output, embedding_table, masked_lm_positions, masked_lm_weights)

        pool_output = self.compute_pool_output(sequence_output)
        next_sentence_log_probs = self.get_next_sentence_output(pool_output)

        return masked_lm_log_probs, next_sentence_log_probs

    def get_masked_lm_output(self, input_tensor, embedding_table, positions, masked_lm_weights):
        input_tensor = utils.gather_indexes(input_tensor, positions)
        # print(input_tensor.shape)
        log_probs = self.output_cls_layer((input_tensor, embedding_table))
        # print('wetwet',log_probs.shape)
        # print(log_probs.shape)
        # assert 1==2

        word_label_weights = tf.reshape(masked_lm_weights, [-1, 1])

        weighted_log_probs = log_probs * word_label_weights

        denominator = tf.reduce_sum(word_label_weights) + 1e-5

        weighted_log_probs = weighted_log_probs / denominator

        return weighted_log_probs

    def get_next_sentence_output(self, input_tensor):
        """Get loss and log probs for the next sentence prediction."""

        # Simple binary classification. Note that 0 is "next sentence" and 1 is
        # "random sentence". This weight matrix is not used after pre-training.
        logits = self.output_next_sentence_layer(input_tensor)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        '''
        # sentence_label_weigths = 1 / tf.shape(sentence_labels)[0]
        w=1.0 / tf.shape(input_tensor)[0]
        sentence_label_weigths = tf.cast(x=w,dtype=tf.float32)

        weighted_log_probs = log_probs * sentence_label_weigths
        '''
        # todo::使用weight方法报错：TypeError: Cannot convert 1.0 to EagerTensor of dtype int32，暂且直接类型转换
        weighted_log_probs = log_probs / tf.cast(tf.shape(input_tensor)[0], tf.float32)

        return weighted_log_probs

    # todo::build和get不同，先build，再get
    # todo::增加shape一致性检查
    def build_variable_mapping(self):
        #todo::dict更新函数
        # mapping.update({})
        with open('./config/variable_names_to_pretrained_names_table.json','r',encoding='utf8') as fr:
            variable_names_to_pretrained_names_table=json.load(fr)

        return variable_names_to_pretrained_names_table

    def load_variable_value_from_checkpoint(self,checkpoint,var_name):
        return tf.train.load_variable(checkpoint,var_name)

    def restore_from_checkpoint(self, checkpoint):
        raw_weights=self.trainable_weights

        raw_weight_names_to_pretrained_weight_names=self.build_variable_mapping()

        raw_weights_to_pretrained_weigths=list()
        for raw_weight_name,pretrained_weight_name in raw_weight_names_to_pretrained_weight_names.items():
            raw_weight=None
            for weight in raw_weights:
                if weight.name==raw_weight_name:
                    raw_weight=weight
                    break
            if raw_weight is None:
                raise Exception('{} not found'.format(raw_weight_name))

            pretrained_weight=self.load_variable_value_from_checkpoint(checkpoint=checkpoint,var_name=pretrained_weight_name)

            raw_weights_to_pretrained_weigths.append((raw_weight,pretrained_weight))

        tf.keras.backend.batch_set_value(raw_weights_to_pretrained_weigths)
        return None

        #todo::设置值方法3扩展：model继承自layer，module。layer有trainable_weigths参数，model也有。利用该参数实现映射
        #todo:: print(self.embedding_layer.wordpiece_embedding_layer.trainable_weights)
        #todo:: t_vars=self.trainable_weights
        #todo:: for v in t_vars:
        #todo::     print(v.name,v.shape,v.dtype)
        print('-------'*20)
        for v in self.trainable_weights:
            print(v.name)

        assert 1==2
        # print(self.layers)
        # print('bert/embeddings/word_embeddings/word_embeddings' in self.layers)

        #todo:: 设置值方法1：先检索出layer，然后使用set_weights函数设置权重值,Cons:使用位置对应，不是名字对应，适合参数少的情况
        #todo:: example self.get_layer('embeddings').set_weights()
        # layer_weigths=self.get_layer('embeddings').get_weights()

        #todo:: 设置值方法2：model.load_weigths(file_path,by_name=False,skip_mismatch=False,options=None)
        #todo:: example

        #todo::设置值方法3：tf.keras.backend.batch_set_value()
        #todo:: raw_value=self.embedding_layer.wordpiece_embedding_layer.embedding_table
        #todo:: pre_trained_value=self.load_variable_value_from_checkpoint(checkpoint,'bert/embeddings/word_embeddings')
        #todo:: tf.keras.backend.batch_set_value([(raw_value,pre_trained_value)])


        assert 1==2

        layer_name_to_weight_names = self.build_variable_mapping()
        layer_name_to_weight_names = {k: v if k in self.layers else warnings.warn('{} not found'.format(k)) for k, v in layer_name_to_weight_names.items()}

        print(layer_name_to_weight_names)
        assert 1==2

        raw_value_to_pretrained_value_pairs = []
        for layer_name, pre_trained_weight_name in layer_name_to_weight_names.items():
            layer = self.layers[layer_name]
            raw_weight_value = layer.trainable_weights  # List of variables to be included in backprop.

            pre_trained_weight_value = [self.load_variable_value_from_checkpoint(checkpoint, v) for v in pre_trained_weight_name]

            raw_value_to_pretrained_value_pairs.extend(zip(raw_weight_value, pre_trained_weight_value))

        tf.keras.backend.batch_set_value(raw_value_to_pretrained_value_pairs)
    def create_tfv1_tensor(self,name,value,dtype):
        #todo:: RuntimeError: Attempting to capture an EagerTensor without building a function.
        # return tf.compat.v1.keras.backend.variable(value=value,dtype=dtype,name=name)
        return tf.compat.v1.keras.backend.variable(value=tf.ones(shape=value.shape,dtype=dtype),dtype=dtype,name=name)
        # return tf.Variable(initial_value=tf.ones(shape=value.shape,dtype=dtype),dtype=dtype,name=name)

    def save_weights_as_checkpoint(self):
        bert24_weights=self.trainable_weights
        bert24_weight_names_to_official_weight_names=self.build_variable_mapping()

        #========================================

        all_variables, all_values,all_types = [], [],[]
        for bert24_weight_name, official_weight_name in bert24_weight_names_to_official_weight_names.items():
            value = None
            for weight in bert24_weights:
                if weight.name == bert24_weight_name:
                    value = weight
                    break
            if value is None:
                raise ValueError('{} not found'.format(bert24_weight_names_to_official_weight_names))

            # value=tf.keras.backend.get_value(bert24_weight_name)
            all_variables.append(official_weight_name)
            all_values.append(value.numpy())
            all_types.append(value.dtype)
        #========================================

        with tf.compat.v1.Graph().as_default():
            all_session_tensor=list()
            for var_name,value,type in zip(all_variables,all_values,all_types):
                session_tensor = self.create_tfv1_tensor(var_name, value, type)
                all_session_tensor.append(session_tensor)

            with tf.compat.v1.Session() as sess:
                # for varabel,value in zip(all_session_tensor,all_values):
                #     tf.compat.v1.assign(varabel,value)
                tf.compat.v1.keras.backend.batch_set_value(zip(all_session_tensor, all_values))
                saver = tf.compat.v1.train.Saver()
                saver.save(sess, './files/checkpoints/my/bert.ckpt')

    def save_weights_as_checkpoint_v1(self):

        class ForSaveModel(object):
            def __init__(self):
                super(ForSaveModel, self).__init__()


            def set_weights_by_pretrained_value(self,name_value_dtype_tuple,checkpoint_dir):
                #创建变量名
                for name,value,dtype in name_value_dtype_tuple:
                    print(name)
                    self.name=tf.Variable(initial_value=tf.ones(shape=value.shape,dtype=dtype),trainable=True,name=name,dtype=dtype)

                #赋值变量
                raw_weights = self.trainable_weights
                print('yet',raw_weights)

                raw_value_to_pretraiend_value_pairs=list()

                for name,pretrained_value,_ in name_value_dtype_tuple:
                    raw_value=None
                    for weight in raw_weights:
                        if weight.name==name:
                            raw_value=weight
                            break
                    if raw_value is None:
                        raise Exception('{} not found'.format(name))

                    raw_value_to_pretraiend_value_pairs.append((raw_value,pretrained_value))

                tf.keras.backend.batch_set_value(raw_value_to_pretraiend_value_pairs)

                #保存checkpoint
                tf.train.Checkpoint(self).write(checkpoint_dir)
                print('success')
            def call(self,x):
                pass

        bert24_weights=self.trainable_weights

        bert24_weight_names_to_official_weight_names=self.build_variable_mapping()

        official_weight_names_to_bert24_weight =list()
        mapping=dict()
        for bert24_weight_name,official_weight_name in bert24_weight_names_to_official_weight_names.items():
            print(official_weight_name)
            bert24_weight=None
            for weight in bert24_weights:
                if weight.name==bert24_weight_name:
                    bert24_weight=weight
                    break
            if bert24_weight is None:
                raise Exception('{} not found'.format(bert24_weight_name))

            official_weight_names_to_bert24_weight.append((official_weight_name,bert24_weight,bert24_weight.dtype))

            #todo::使用value()获取tensor值，tf文档没有
            mapping[official_weight_name]=bert24_weight.value()
            # tmp=bert24_weight.value()
            # print(type(tmp),tmp.shape,tmp.dtype)

        # assert 1==2
        #todo::保存方法1：直接使用list放在listed位置
        checkpoint_saver=tf.train.Checkpoint()
        checkpoint_saver.mapped=mapping
        # checkpoint_saver.listed=bert24_weights
        #todo::必须加/表示文件夹
        checkpoint_saver.save('./files/checkpoints/my/bert.ckpt')
        print('success')

        # for_checkpoint_model=ForSaveModel()
        # for_checkpoint_model.set_weights_by_pretrained_value(official_weight_names_to_bert24_weight,checkpoint_dir='./files/checkpoints/my')

        # mapping = mapping or self.variable_mapping()
        # mapping = {self.prefixed(k): v for k, v in mapping.items()}
        # mapping = {k: v for k, v in mapping.items() if k in self.layers}
        #
        # with tf.Graph().as_default():
        #     all_variables, all_values = [], []
        #     for layer, variables in mapping.items():
        #         layer = self.layers[layer]
        #         values = K.batch_get_value(layer.trainable_weights)
        #         for name, value in zip(variables, values):
        #             variable, value = self.create_variable(name, value, dtype)
        #             all_variables.append(variable)
        #             all_values.append(value)
        #
        #     with tf.Session() as sess:
        #         K.batch_set_value(zip(all_variables, all_values))
        #         saver = tf.train.Saver()
        #         saver.save(sess, filename)

    # todo::tensorflow的keras文档没有batch_set_value函数介绍，直接写函数然后查看函数签名
    # tf.keras.backend.batch_set_value()

    # def get_config(self):
    #     return None
    #
    # @classmethod
    # def from_config(self, config, custom_objects=None):
    #     return None

    '''
    # deprecated::
    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table
    '''
