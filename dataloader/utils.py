import numpy as np
import torch
import datetime

from transformers import BertTokenizer


def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), s),
          flush=True)


def to_tensor(data, cuda, exclude_keys=[]):
    '''
        Convert all values in the data into torch.tensor
    '''
    for key in data.keys():
        if key in exclude_keys:
            continue

        data[key] = torch.from_numpy(data[key])
        if cuda != -1:
            data[key] = data[key].cuda(cuda)

    return data


def select_subset(old_data, new_data, keys, idx, max_len=None):
    '''
        modifies new_data

        @param old_data target dict
        @param new_data source dict
        @param keys list of keys to transfer
        @param idx list of indices to select
        @param max_len (optional) select first max_len entries along dim 1
    '''

    for k in keys:
        new_data[k] = old_data[k][idx]
        if max_len is not None and len(new_data[k].shape) > 1:
            new_data[k] = new_data[k][:,:max_len]

    return new_data


class Stop_Augment(object):
    def __init__(self, args):
        self.args = args
        stop_file = open('../data/stop_word.txt', 'r', encoding='utf-8')
        self.stop_word = []
        for word in stop_file:
            self.stop_word.append(word.rstrip('\n').lstrip('\ufeff'))
        self.stop_word = np.array(self.stop_word)
        stop_file.close()

        self.tokenizer = BertTokenizer.from_pretrained("../module/bert-base-uncased")

    def data_augment(self, data):
        #  query sample was acquired by data augmentation
        query_data = []
        for line in data:
            line_list = line.split(' ')
            sent = []
            for i, word in enumerate(line_list):
                if word in self.stop_word:
                    word = np.random.permutation(self.stop_word)[0]
                sent.append(word)
            item = {'text': ' '.join(sent)}
            query_data.append(item)

        return query_data

    def token_to_idx(self, data):

        for e in data:
            tokenize = self.tokenizer(e['text'], return_tensors="pt")
            e['bert_id'], e['attn_mask'] = tokenize['input_ids'][0].numpy(), tokenize['attention_mask'][0].numpy()

        text_len = np.array([len(e['bert_id']) for e in data])
        max_text_len = max(text_len)

        text, text_mask = self.get_data(data, max_text_len)

        new_data = {
            'text': text,
            'attn_mask': text_mask
        }

        return new_data

    def get_data(self, data, max_text_len):
        text = np.zeros([len(data), max_text_len], dtype=np.int64)
        text_mask = np.zeros([len(data), max_text_len], dtype=np.int64)

        del_idx = []
        # convert each token to its corresponding id
        for i in range(len(data)):
            text[i, :len(data[i]['bert_id'])] = data[i]['bert_id']
            text_mask[i, :len(data[i]['attn_mask'])] = data[i]['attn_mask']

            if np.max(text[i]) < 103:
                del_idx.append(i)

        text = self._del_by_idx([text], del_idx, 0)
        return text, text_mask

    def _del_by_idx(self, array_list, idx, axis):
        if type(array_list) is not list:
            array_list = [array_list]

        # modified to perform operations in place
        for i, array in enumerate(array_list):
            array_list[i] = np.delete(array, idx, axis)

        if len(array_list) == 1:
            return array_list[0]
        else:
            return array_list

