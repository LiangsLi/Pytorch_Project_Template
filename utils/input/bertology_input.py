# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     bert_input
   Description :
   Author :       Liangs
   date：          2019/7/30
-------------------------------------------------
   Change Activity:
                   2019/11/7:
-------------------------------------------------
"""
import os
import csv
import pathlib
import sys
import pandas
import torch
from pytorch_transformers import BertTokenizer, RobertaTokenizer, XLMTokenizer, XLNetTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from utils.input.bertology_data import SequenceClassificationProcessor, InputFeatures

BERT_TOKENIZER = {
    'bert': BertTokenizer,
    'xlnet': XLNetTokenizer,
    'xlm': XLMTokenizer,
    'roberta': RobertaTokenizer,
}


def load_bert_tokenizer(model_path, model_type, do_lower_case=True):
    # 必须把 unused 添加到 additional_special_tokens 上，否则 unused （用来表示ROOT）可能无法正确切分！
    return BERT_TOKENIZER[model_type].from_pretrained(model_path, do_lower_case=do_lower_case,
                                                      additional_special_tokens=['[unused1]', '[unused2]', '[unused3]'])


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode='classification'):
    """Loads a data file into a list of `InputBatch`s."""
    if label_list:
        label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     print("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if label_list:
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
        else:
            label_id = None

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def convert_features_to_dataloader(features, batch_size, random_sample=True):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if random_sample:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def load_data_for_nlu_task(args, train=True, dev=False, test=False):
    processor = SequenceClassificationProcessor(args.class_num, args.is_complex)
    label_list = processor.get_labels()
    assert pathlib.Path(args.saved_model_path).exists()
    tokenizer = load_bert_tokenizer(args.saved_model_path, args.bertology_type)
    if train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_len, tokenizer)
        train_dataloader = convert_features_to_dataloader(train_features, batch_size=args.train_batch_size,
                                                          random_sample=True)
    else:
        train_dataloader = None
    if dev:
        dev_examples = processor.get_dev_examples(args.data_dir)
        dev_features = convert_examples_to_features(dev_examples, label_list, args.max_seq_len, tokenizer)
        dev_dataloader = convert_features_to_dataloader(dev_features, batch_size=args.eval_batch_size,
                                                        random_sample=False)
    else:
        dev_dataloader = None
    return train_dataloader, dev_dataloader, None
