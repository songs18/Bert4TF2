# -*- coding: utf-8 -*-
# @Time    : 10:16 2021/4/26 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : Config.py
import yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf8') as fr:
        config = yaml.load(fr, Loader=yaml.FullLoader)
    return config


class DataConfig:
    def __init__(self, config):
        if not config:
            config=dict()

        for k, v in config.items():
            setattr(self, k, v)


# ==============================================================================================================

class WordEmbeddingConfig:
    def __init__(self, config):
        if not config:
            config=dict()
        for k, v in config.items():
            setattr(self, k, v)


class TokenTypeEmbeddingConfig:
    def __init__(self, config):
        if not config:
            config=dict()
        for k, v in config.items():
            setattr(self, k, v)


class PositionEmbeddingConfig:
    def __init__(self, config):
        if not config:
            config=dict()
        for k, v in config.items():
            setattr(self, k, v)


class EmbeddingLayerConfig:
    def __init__(self, config):
        self.word_piece_embedding_layer_config = WordEmbeddingConfig(config.pop('word_piece_embedding_layer_config', {}))
        self.tokentype_embedding_layer_config = TokenTypeEmbeddingConfig(config.pop('token_type_embedding_layer_config', {}))
        self.position_embedding_layer_config = PositionEmbeddingConfig(config.pop('position_embedding_layer_config', {}))

        for k, v in config.items():
            setattr(self, k, v)


# ==============================================================================================================
class AttentionConfig:
    def __init__(self,config):
        if not config:
            config=dict()
        for k,v in config.items():
            setattr(self,k,v)

class AttentionOutputConfig:
    def __init__(self,config):
        if not config:
            config=dict()
        for k,v in config.items():
            setattr(self,k,v)

class AttentionSubLayerConfig:
    def __init__(self,config):
        if not config:
            config=dict()
        self.self_config=AttentionConfig(config.pop('self',{}))
        self.output_config=AttentionOutputConfig(config.pop('output',{}))

        for k,v in config.items():
            setattr(self,k,v)


#=====================================================================================

class IntermediateConfig:
    def __init__(self, config):
        if not config:
            config = dict()
        for k, v in config.items():
            setattr(self, k, v)


class IntermediateOutputConfig:
    def __init__(self, config):
        if not config:
            config = dict()
        for k, v in config.items():
            setattr(self, k, v)


class FeedForwardSubLayerConfig:
    def __init__(self, config):
        if not config:
            config = dict()
        self.intermediate_config = IntermediateConfig(config.pop('intermediate', {}))
        self.output_config = IntermediateOutputConfig(config.pop('output', {}))

        for k,v in config.items():
            setattr(self,k,v)

#=====================================================================================
class SingleLayerConfig:
    def __init__(self, config):
        if not config:
            config = dict()
        self.selfattention_sublayer_config = AttentionSubLayerConfig(config.pop('selfattention_sublayer_config', {}))
        self.feeforward_sublayer_config = FeedForwardSubLayerConfig(config.pop('feedforward_sublayer_config', {}))

        for k, v in config.items():
            setattr(self, k, v)



class TransformerLayersConfig:
    def __init__(self, config):
        if not config:
            config=dict()
        self.single_layer_config = SingleLayerConfig(config.pop('single_layer_config', {}))

        for k, v in config.items():
            setattr(self, k, v)


# ==============================================================================================================

class PoolingLayerConfig:
    def __init__(self, config):
        if not config:
            config=dict()
        for k, v in config.items():
            setattr(self, k, v)


# ==============================================================================================================
class WordClassifierLayerConfig:
    def __init__(self, config):
        if not config:
            config=dict()
        for k, v in config.items():
            setattr(self, k, v)

# ==============================================================================================================

class NextSentenceLayerConfig:
    def __init__(self, config):
        if not config:
            config=dict()
        for k, v in config.items():
            setattr(self, k, v)
# ==============================================================================================================

class ModelConfig:
    def __init__(self, config):
        self.embedding_layer_config = EmbeddingLayerConfig(config.pop('embedding_layer_config',{}))
        self.transformer_layers_config = TransformerLayersConfig(config.pop('transformer_layers_config', {}))
        self.pooling_layer_config = PoolingLayerConfig(config.pop('pooling_layer_config', {}))

        self.word_cls_layer_config = WordClassifierLayerConfig(config.pop('word_cls_layer_config', {}))
        self.next_sentence_layer_config = NextSentenceLayerConfig(config.pop('next_sentence_layer_config', {}))


        for k, v in config.items():
            setattr(self, k, v)


# ==============================================================================================================

class IOConfig:
    def __init__(self, config):
        if not config:
            config=dict()
        for k, v in config.items():
            setattr(self, k, v)


class DeviceConfig:
    def __init__(self, config):
        if not config:
            config=dict()
        for k, v in config.items():
            setattr(self, k, v)


class TrainingConfig:
    def __init__(self, config):
        if not config:
            config=dict()
        for k, v in config.items():
            setattr(self, k, v)


class EvalConfig:
    def __init__(self, config):
        if not config:
            config=dict()
        for k, v in config.items():
            setattr(self, k, v)


class RunningConfig:
    def __init__(self, config):
        self.io_config = IOConfig(config.pop('IO_config', {}))
        self.device_config = DeviceConfig(config.pop('device_config', {}))
        self.training_config = TrainingConfig(config.pop('training_config', {}))
        self.eval_config = EvalConfig(config.pop('eval_config', {}))

        for k, v in config.items():
            setattr(self, k, v)


class Config:
    def __init__(self, path):
        config = load_yaml(path)

        self.data_config=DataConfig(config.pop('data_config', {}))
        self.model_config = ModelConfig(config.pop('model_config', {}))
        self.running_config = RunningConfig(config.pop('running_config', {}))

        for k, v in config.items():
            setattr(self, k, v)


# 设置为{}是软性的
# if not config: config = {}

'''
# 3种embedding混合，这里分开
EmbeddingLayerConfig = collections.namedtuple('EmbeddingConfig', ['name', 'word_embedding_config', 'tokentype_embedding_config', 'position_embedding_config', 'dropout_prob'])

WordEmbeddingConfig = collections.namedtuple('WordEmbeddingConfig', ['name', 'vocab_size', 'embedding_size', 'initializer_range', 'use_one_hot_embeddings'])
TokenTypeEmbeddingConfig = collections.namedtuple('TokenTypeEmbeddingConfig', ['name', 'vocab_size', 'initializer_range'])
PositionEmbeddingConfig = collections.namedtuple('PositionEmbeddingConfig', ['name', 'max_position_embeddings', 'initializer_range'])

# PostProcessConfig=collections.namedtuple('PostProcessConfig',['use_token_type','token_type_ids','token_type_vocab_size',
#                                                               'token_type_embedding_name','use_position_embeddings',
#                                                               'position_embedding_name','initializer_range',
#                                                               'max_position_embeddings','dropout_prob'])
word_embedding_config = WordEmbeddingConfig('word_embedding', 1, 1, 1, False)
tokentype_embedding_config = TokenTypeEmbeddingConfig('token_type_embeddings', 16, 0.02)
position_embedding_config = PositionEmbeddingConfig('position_embeddings', 512, 0.02)
# post_process_config=PostProcessConfig(use_token_type=False, token_type_ids=None, token_type_vocab_size=16,
#                                 token_type_embedding_name="token_type_embeddings", use_position_embeddings=True,
#                                 position_embedding_name="position_embeddings", initializer_range=0.02,
#                                 max_position_embeddings=512, dropout_prob=0.1)
embeddinglayer_config = EmbeddingLayerConfig('embeddings', word_embedding_config, tokentype_embedding_config, position_embedding_config)
'''

'''
class DecoderConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.beam_width = config.pop("beam_width", 0)
        self.blank_at_zero = config.pop("blank_at_zero", True)
        self.norm_score = config.pop("norm_score", True)
        self.lm_config = config.pop("lm_config", {})

        self.vocabulary = preprocess_paths(config.pop("vocabulary", None))
        self.target_vocab_size = config.pop("target_vocab_size", 1024)
        self.max_subword_length = config.pop("max_subword_length", 4)
        self.output_path_prefix = preprocess_paths(config.pop("output_path_prefix", None))
        self.model_type = config.pop("model_type", None)
        self.corpus_files = preprocess_paths(config.pop("corpus_files", []))
        self.max_corpus_chars = config.pop("max_corpus_chars", None)
        self.reserved_tokens = config.pop("reserved_tokens", None)

        for k, v in config.items(): setattr(self, k, v)


class DatasetConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.stage = config.pop("stage", None)
        self.data_paths = preprocess_paths(config.pop("data_paths", None))
        self.tfrecords_dir = preprocess_paths(config.pop("tfrecords_dir", None))
        self.tfrecords_shards = config.pop("tfrecords_shards", 16)
        self.shuffle = config.pop("shuffle", False)
        self.cache = config.pop("cache", False)
        self.drop_remainder = config.pop("drop_remainder", True)
        self.buffer_size = config.pop("buffer_size", 100)
        self.use_tf = config.pop("use_tf", False)
        self.augmentations = Augmentation(config.pop("augmentation_config", {}), use_tf=self.use_tf)
        for k, v in config.items(): setattr(self, k, v)


class RunningConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.batch_size = config.pop("batch_size", 1)
        self.accumulation_steps = config.pop("accumulation_steps", 1)
        self.num_epochs = config.pop("num_epochs", 20)
        self.outdir = preprocess_paths(config.pop("outdir", None))
        self.log_interval_steps = config.pop("log_interval_steps", 500)
        self.save_interval_steps = config.pop("save_interval_steps", 500)
        self.eval_interval_steps = config.pop("eval_interval_steps", 1000)
        for k, v in config.items(): setattr(self, k, v)


class LearningConfig:
    def __init__(self, config: dict = None):
        if not config: config = {}
        self.train_dataset_config = DatasetConfig(config.pop("train_dataset_config", {}))
        self.eval_dataset_config = DatasetConfig(config.pop("eval_dataset_config", {}))
        self.test_dataset_config = DatasetConfig(config.pop("test_dataset_config", {}))
        self.optimizer_config = config.pop("optimizer_config", {})
        self.running_config = RunningConfig(config.pop("running_config", {}))
        for k, v in config.items(): setattr(self, k, v)


class Config:
    """ User config class for training, testing or infering """

    def __init__(self, path: str):
        config = load_yaml(preprocess_paths(path))

        self.speech_config = config.pop("speech_config", {})
        self.decoder_config = config.pop("decoder_config", {})
        self.model_config = config.pop("model_config", {})
        self.learning_config = LearningConfig(config.pop("learning_config", {}))

        for k, v in config.items():
            setattr(self, k, v)
'''
