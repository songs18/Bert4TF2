# -*- coding: utf-8 -*-
# @Time    : 22:27 2021/4/28 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : test.py
from config.config import Config
from nn.bert import BertModel

config = Config('./config/config.yml')
bert_config = config.model_config

running_config = config.running_config
io_config = running_config.io_config
device_config = running_config.device_config
training_config = running_config.training_config
eval_config = running_config.eval_config

if not bert_config.do_train and not bert_config.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

model = BertModel(config=bert_config)
print(model.trainable_variables)
