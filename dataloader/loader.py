import os
import itertools
import collections
import json
from collections import defaultdict
from copy import copy

import numpy as np
import pandas as pd
import torch
from torchtext.vocab import Vocab, Vectors

from embedding.avg import AVG
from embedding.cxtebd import CXTEBD
from embedding.wordebd import WORDEBD
import dataset.stats as stats
from dataset.utils import tprint

from transformers import BertTokenizer, AutoTokenizer
# from pytorch_transformers import BertModel

def _get_Liu_classes(args):

    train_classes = list(range(20))
    val_classes = list(range(20, 30))
    test_classes = list(range(30, 54))

    return train_classes, val_classes, test_classes


def _get_COVID_19_classes(args):
    '''
        @return list of classes associated with each split
    '''
    train_classes = list(range(36))
    val_classes = list(range(36, 46))
    test_classes = list(range(46, 81))

    return train_classes, val_classes, test_classes


def _get_hwu64_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(23))
    val_classes = list(range(23, 39))
    test_classes = list(range(39, 64))

    return train_classes, val_classes, test_classes


def _get_clinic150_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(50))
    val_classes = list(range(50, 100))
    test_classes = list(range(100, 150))

    return train_classes, val_classes, test_classes


def _get_banking77_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(25))
    val_classes = list(range(25, 50))
    test_classes = list(range(50, 77))

    return train_classes, val_classes, test_classes

def _get_huffpost_classes(args):

    train_classes = [0, 1, 3, 4, 9, 10, 12, 14, 15, 17, 19, 20, 21, 23, 29, 30, 32, 33, 35, 37]
    val_classes = [2, 13, 18, 22, 34]
    test_classes = [5, 6, 7, 8, 11, 16, 24, 25, 26, 27, 28, 31, 36, 38, 39, 40]

    return train_classes, val_classes, test_classes

def _get_cyberbully_classes(args):
    train_classes = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50,
                     52, 54, 56, 58]
    val_classes = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51,
                   53, 55, 57, 59]
    # train_classes = list(range(0, 30))
    # val_classes = list(range(30, 60))
    test_classes = list(range(60, 90))

    return train_classes, val_classes, test_classes


def _get_acid_classes(args):
    train_classes = list(range(0, 50))
    val_classes = list(range(50, 70))
    # train_classes = list(range(0, 30))
    # val_classes = list(range(30, 60))
    test_classes = list(range(70, 116))

    return train_classes, val_classes, test_classes

def _get_minicyberbully_classes(args):

    train_classes = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    val_classes = [11, 13, 15, 17, 19]
    test_classes = [1, 3, 5, 7, 9, 21, 23, 25, 27, 29, 20, 22, 24, 26, 28]

    return train_classes, val_classes, test_classes

def _load_csv(path):
    label = {}
    text_len = []

    # tokenizer = BertTokenizer.from_pretrained("../module/bert-base-uncased")

    dataset = pd.read_csv(path, encoding='UTF-8')
    texts, intents = dataset['content'], dataset['label']

    data = []
    for i, line in enumerate(texts):

        if int(intents[i]) not in label:
            label[int(intents[i])] = 1
        else:
            label[int(intents[i])] += 1

        # ids = tokenizer(line, return_tensors="pt")['input_ids'][0][1:-1]
        # line = tokenizer.convert_ids_to_tokens(ids)

        item = {
            'label': int(intents[i]),
            'text': line,
        }

        text_len.append(len(line))
        data.append(item)

    tprint('Class balance:')

    print(label)

    tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

    return data


def _load_json(path):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            item = {
                'label': int(row['label']),
                'text': row['text'][:500]  # truncate the text to 500 tokens
            }

            text_len.append(len(row['text']))

            data.append(item)

        tprint('Class balance:')

        print(label)

        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data

def _read_words(data):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    for example in data:
        words += example['text']
    return words


def _meta_split(all_data, train_classes, val_classes, test_classes):
    '''
        Split the dataset according to the specified train_classes, val_classes and test_classes

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    for example in all_data:
        if example['label'] in train_classes:
            train_data.append(example)
        if example['label'] in val_classes:
            val_data.append(example)
        if example['label'] in test_classes:
            test_data.append(example)

    return train_data, val_data, test_data


def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list


def _data_to_nparray(data, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)

    # tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    for e in data:
        tokenize = tokenizer(e['text'], return_tensors="pt")
        e['bert_id'], e['attn_mask'] = tokenize['input_ids'][0].numpy(), tokenize['attention_mask'][0].numpy()

    text_len = np.array([len(e['bert_id']) for e in data])
    max_text_len = max(text_len)

    text = np.zeros([len(data), max_text_len], dtype=np.int64)
    text_mask = np.zeros([len(data), max_text_len], dtype=np.int64)

    del_idx = []
    # convert each token to its corresponding id
    for i in range(len(data)):
        text[i, :len(data[i]['bert_id'])] = data[i]['bert_id']
        text_mask[i, :len(data[i]['attn_mask'])] = data[i]['attn_mask']

        # filter out document with only special tokens
        # unk (100), cls (101), sep (102), pad (0)
        if np.max(text[i]) < 103:
            del_idx.append(i)

    text_len, text, doc_label, raw = _del_by_idx([text_len, text, doc_label, raw], del_idx, 0)

    new_data = {
        'text': text,
        'text_len': text_len,
        'attn_mask': text_mask,
        'label': doc_label,
        'raw': raw,
    }

    return new_data

# def _data_to_nparray(data, vocab, args):
#     '''
#         Convert the data into a dictionary of np arrays for speed.
#     '''
#     doc_label = np.array([x['label'] for x in data], dtype=np.int64)
#
#     raw = np.array([e['text'] for e in data], dtype=object)
#
#
#     # compute the max text length
#     text_len = np.array([len(e['text']) for e in data])
#     max_text_len = max(text_len)
#
#     # initialize the big numpy array by <pad>
#     text = vocab.stoi['<pad>'] * np.ones([len(data), max_text_len], dtype=np.int64)
#
#     del_idx = []
#     # convert each token to its corresponding id
#     for i in range(len(data)):
#         text[i, :len(data[i]['text'])] = [
#                 vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>']
#                 for x in data[i]['text']]
#
#         # filter out document with only unk and pad
#         if np.max(text[i]) < 2:
#             del_idx.append(i)
#
#     vocab_size = vocab.vectors.size()[0]
#
#     text_len, text, doc_label, raw = _del_by_idx([text_len, text, doc_label, raw], del_idx, 0)
#
#     new_data = {
#         'text': text,
#         'text_len': text_len,
#         'label': doc_label,
#         'raw': raw,
#         'vocab_size': vocab_size,
#     }
#
#     return new_data

def load_dataset(args):
    """
        判断 dataset 属于哪个数据集，然后调用对应函数获取数据
    """
    # 1.获取类别数
    if args.dataset == 'liu':
        train_classes, val_classes, test_classes = _get_Liu_classes(args)
    elif args.dataset == 'covid':
        train_classes, val_classes, test_classes = _get_COVID_19_classes(args)
    elif args.dataset == 'hwu64':
        train_classes, val_classes, test_classes = _get_hwu64_classes(args)
    elif args.dataset == 'clinic150':
        train_classes, val_classes, test_classes = _get_clinic150_classes(args)
    elif args.dataset == 'banking77':
        train_classes, val_classes, test_classes = _get_banking77_classes(args)
    elif args.dataset == 'acid':
        train_classes, val_classes, test_classes = _get_acid_classes(args)
    elif args.dataset == 'cyberbully':
        train_classes, val_classes, test_classes = _get_cyberbully_classes(args)
    elif args.dataset == 'huffpost':
        train_classes, val_classes, test_classes = _get_huffpost_classes(args)
    elif args.dataset == 'mini_en':
        train_classes, val_classes, test_classes = _get_minicyberbully_classes(args)
    elif args.dataset == 'mini_cn':
        train_classes, val_classes, test_classes = _get_minicyberbully_classes(args)
    else:
        raise ValueError(
            'args.dataset should be one of'
            '[20newsgroup, amazon, fewrel, huffpost, reuters, rcv1]')

    assert(len(train_classes) == args.n_train_class) # 当条件为假时执行
    assert(len(val_classes) == args.n_val_class)
    assert(len(test_classes) == args.n_test_class)

    if args.mode == 'finetune':
        # in finetune, we combine train and val for training the base classifier
        train_classes = train_classes + val_classes
        args.n_train_class = args.n_train_class + args.n_val_class
        args.n_val_class = args.n_train_class

    tprint('Loading data: {} way: {} shot: {}'.format(args.data_path, args.way, args.shot))
    all_data = _load_csv(args.data_path)

    vocab = None

    # tprint('Loading word vectors')
    #
    # vectors = Vectors(args.word_vector, cache=args.wv_path)
    # vocab = Vocab(collections.Counter(_read_words(all_data)), vectors=vectors, specials=['<pad>', '<unk>'], min_freq=5)
    #
    # # print word embedding statistics
    # wv_size = vocab.vectors.size()
    # tprint('Total num. of words: {}, word vector dimension: {}'.format(wv_size[0], wv_size[1]))
    #
    # num_oov = wv_size[0] - torch.nonzero(torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    # tprint('Num. of out-of-vocabulary words (they are initialized to zeros): {}'.format(num_oov))

    # Split into meta-train, meta-val, meta-test data
    train_data, val_data, test_data = _meta_split(all_data, train_classes, val_classes, test_classes)
    tprint('#train {}, #val {}, #test {}'.format(len(train_data), len(val_data), len(test_data)))

    # Convert everything into np array for fast data loading
    train_data = _data_to_nparray(train_data, vocab, args)
    val_data = _data_to_nparray(val_data, vocab, args)
    test_data = _data_to_nparray(test_data, vocab, args)

    train_data['is_train'] = True

    return train_data, val_data, test_data, vocab


def load_dataset_pretrain(args):

    # filename = ["clinic150", "banking77", "hwu64", "Liu"]

    data = []
    for name in args.filename:
        label = {}
        path = os.path.join("../data", name + '.csv')

        dataset = pd.read_csv(path, encoding='UTF-8')
        texts, intents = dataset['content'], dataset['label']

        for i, line in enumerate(texts):

            if int(intents[i]) not in label:
                label[int(intents[i])] = 1
            else:
                label[int(intents[i])] += 1

            item = {'label': int(intents[i]), 'text': line}
            data.append(item)

    all_data = _data_to_nparray(data, None, args)

    all_data['is_train'] = True

    return all_data


class RTR_Augments:
    def __init__(self, args):
        self.tokenizer = BertTokenizer.from_pretrained("../module/bert-base-uncased")
        self.rtr_prob = 0.25

    def random_token_replace(self, ids, mlm_probability):
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        masked_ids, labels = self.mask_tokens(ids.clone(), self.tokenizer, mlm_probability=mlm_probability)
        aug_ids = masked_ids.clone()
        random_words = torch.randint(len(self.tokenizer), aug_ids.shape, dtype=torch.long)
        indices_replaced = torch.where(aug_ids == mask_id)
        aug_ids[indices_replaced] = random_words[indices_replaced].cuda()
        return aug_ids, masked_ids, labels

    def mask_tokens(self, inputs, tokenizer, special_tokens_mask=None, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs == 0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random].cuda()

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
