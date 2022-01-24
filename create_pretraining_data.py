# coding=utf-8
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random

from data import tokenization
from data import read_text
from data.create_instance import create_instances_from_document
from data.write_TFRecord import write_instance_to_example_files
from config.config import Config


def main(_):
    config = Config('./config/config.yml')
    tokenizer = tokenization.FullTokenizer(vocab_file=config.data_config.vocab_file, do_lower_case=config.data_config.do_lower_case)
    rng = random.Random(config.data_config.random_seed)

    # step1
    input_patterns = config.data_config.input_file.split(",")
    input_files = read_text.read_files(input_patterns)
    all_documents = read_text.read_documents(input_files, tokenizer, rng)

    # step2
    DataStandard = collections.namedtuple('DataStandard', ['max_seq_length',
                                                           'dupe_factor',
                                                           'short_seq_prob',
                                                           'masked_lm_prob',
                                                           'max_predictions_per_seq',
                                                           'do_whole_word_mask'])
    data_standard = DataStandard(config.data_config.max_seq_length,
                                 config.data_config.dupe_factor,
                                 config.data_config.short_seq_prob,
                                 config.data_config.masked_lm_prob,
                                 config.data_config.max_predictions_per_seq,
                                 config.data_config.do_whole_word_mask)
    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(data_standard.dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(create_instances_from_document(all_documents, document_index, data_standard, vocab_words, rng))
    rng.shuffle(instances)

    # step3
    output_files = config.data_config.output_file.split(",")
    print("*** Writing to output files ***")
    for output_file in output_files:
        print("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, config.data_config.max_seq_length, config.data_config.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    # flags.mark_flag_as_required("input_file")
    # flags.mark_flag_as_required("output_file")
    # flags.mark_flag_as_required("vocab_file")
    # tf.app.run()
    main('')
