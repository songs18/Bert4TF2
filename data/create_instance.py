# -*- coding: utf-8 -*-
# @Time    : 22:03 2021/4/24 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : create_instances.py
import collections

TrainingInstance = collections.namedtuple('TrainingInstance', ['tokens',
                                                               'segment_ids',
                                                               'is_random_next',
                                                               'masked_lm_positions',
                                                               'masked_lm_labels'])

# =============================================================================

def create_instances_from_document(all_documents, document_index, data_standard, vocab_words, rng):
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = data_standard.max_seq_length - 3

    # 10% short_sequences
    target_seq_length = get_seq_length(data_standard, max_num_tokens, rng)

    instances = []
    segments = []
    num_tokens = 0
    segment_index = 0
    while segment_index < len(document):
        segment = document[segment_index]  # segment==sentece

        segments.append(segment)
        num_tokens+= len(segment)

        if segment_index == len(document) - 1 or num_tokens >= target_seq_length:
            if segments:
                a_end = 1
                if len(segments) >= 2: a_end = rng.randint(1, len(segments) - 1)

                # ========================================================
                a_tokens = []
                for j in range(a_end): a_tokens.extend(segments[j])

                # ========================================================
                b_tokens = []
                is_random_next = False
                if len(segments) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    b_length = target_seq_length - len(a_tokens)

                    b_tokens=random_sentence(all_documents, document_index, rng, b_length, b_tokens)

                    num_unused_segments = len(segments) - a_end
                    segment_index -= num_unused_segments
                else:
                    is_random_next = False
                    for j in range(a_end, len(segments)): b_tokens.extend(segments[j])

                # ========================================================
                truncate_seq_pair(a_tokens, b_tokens, max_num_tokens, rng)

                assert len(a_tokens) >= 1
                assert len(b_tokens) >= 1

                # ========================================================
                segment_ids, tokens = get_segment_tokens(a_tokens, b_tokens)
                # ========================================================

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(tokens, data_standard, vocab_words, rng)

                instance = TrainingInstance(tokens, segment_ids, is_random_next, masked_lm_positions, masked_lm_labels)
                #TrainingInstance 5 attributes:

                #token:       [CLS] my(id) dog(id) is(id) cute(id) [SEP] he(id) likes(id) play(id) ##ing(id) [SEP] 0 todo::token转id
                #segment_ids:   0     0     0   0    0    0    1    1      1    1     1   0 todo::segment转id; padding的也是0！！
                #input_mask:    1     1     1    1    1    1    1    1    1     1     1   0

                #mask_lm_position: [1, 3, 0]
                #mask_lm_lables: [my(id), is(id), 0]
                #mask_lm_weigths: [1.0,1.0]

                #is_random_next: 1,否则0
                instances.append(instance)

            segments = []
            num_tokens = 0

        segment_index += 1

    return instances


def get_segment_tokens(a_tokens, b_tokens):
    tokens = ["[CLS]"]
    segment_ids = [0]
    for token in a_tokens:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in b_tokens:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    return segment_ids, tokens


def get_seq_length(data_standard, max_num_tokens, rng):
    target_seq_length = max_num_tokens
    if rng.random() < data_standard.short_seq_prob: target_seq_length = rng.randint(2, max_num_tokens)
    return target_seq_length


def random_sentence(all_documents, document_index, rng, b_length, b_tokens):
    random_document_index = -1
    for _ in range(10):
        random_document_index = rng.randint(0, len(all_documents) - 1)
        if random_document_index != document_index: break

    random_document = all_documents[random_document_index]
    random_start = rng.randint(0, len(random_document) - 1)
    for j in range(random_start, len(random_document)):
        b_tokens.extend(random_document[j])
        if len(b_tokens) >= b_length:
            break
    return b_tokens



MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens, data_standard, vocab_words, rng):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]": continue

        if (data_standard.do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)


    num_to_predict = min(data_standard.max_predictions_per_seq, max(1, int(round(len(tokens) * data_standard.masked_lm_prob))))

    output_tokens = list(tokens)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict: break
        if len(masked_lms) + len(index_set) > num_to_predict: continue

        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered: continue

        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            #todo::一个sequence中进行多次mask的Cons：损害sequence的语义；-> 分多个sequence分别训练
            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens: break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()
