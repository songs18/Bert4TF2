# -*- coding: utf-8 -*-
# @Time    : 22:05 2021/4/24 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : read_files.py
import tensorflow as tf
# from b_data import a_tokenization as tokenization
from data import tokenization

def read_files(input_patterns):
    input_files = []
    for input_pattern in input_patterns:
        input_files.extend(tf.io.gfile.glob(input_pattern))

    print("*** Reading from input files ***")
    for input_file in input_files:
        print("  %s", input_file)

    return input_files

def read_documents(input_files,tokenizer,rng):
    all_documents = [[]]  # [documents[document[sentence][sentence]...]]

    for input_file in input_files:
        with tf.io.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line: break

                line = line.strip()

                if not line: all_documents.append([])

                tokens = tokenizer.tokenize(line)
                if tokens: all_documents[-1].append(tokens)

    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    return all_documents

# def create_training_instances(input_files, tokenizer, data_stadard, rng):

