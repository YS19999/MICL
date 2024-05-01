import time
import datetime
from copy import deepcopy, copy
from multiprocessing import Process, Queue, cpu_count

import torch
import numpy as np
from dataset.loader import RTR_Augments
# from pytorch_transformers import BertModel
from transformers import BertModel

import dataset.utils as utils
import dataset.stats as stats

class Sampler_PT():
    def __init__(self, data, args, num_episodes=None):
        self.data = data # 训练数据集
        self.args = args
        self.num_episodes = num_episodes # 训练任务 episodes

        self.idx_list = np.arange(0, len(self.data['raw']))
        self.rtr_aug = RTR_Augments(self.args)
        self.stop_aug = utils.Stop_Augment(self.args)

        self.count = 0
        self.done_queue = Queue()

        # self.num_cores = cpu_count() if args.n_workers is 0 else args.n_workers
        self.num_cores = 1

        self.p_list = []
        for i in range(self.num_cores):
            self.p_list.append(
                Process(target=self.worker, args=(self.done_queue,)))

        for i in range(self.num_cores):
            self.p_list[i].start()

    def get_epoch(self):

        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support = self.done_queue.get()
            query = deepcopy(support)
            query_raw = self.stop_aug.data_augment(copy(query['raw']))
            stop_text = self.stop_aug.token_to_idx(query_raw)
            query['text_aug1'], query['text_aug1_mask'] = stop_text['text'], stop_text['attn_mask']

            support = utils.to_tensor(support, self.args.cuda, ['raw'])
            query = utils.to_tensor(query, self.args.cuda, ['raw'])

            query['text_aug2'], support['mask_ids'], support['mask_label'] = self.rtr_aug.random_token_replace(support['text'], 0.35)

            support['is_support'] = True
            query['is_support'] = False

            yield support, query

    def worker(self, done_queue):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        while True:
            if done_queue.qsize() > 100:
                time.sleep(1)
                continue

            # getting support set
            tmp = np.random.permutation(self.idx_list)
            support_idx = tmp[:self.args.pre_way]

            # aggregate examples
            max_support_len = np.max(self.data['text_len'][support_idx])
            support = utils.select_subset(self.data, {}, ['text', 'attn_mask'],
                                          support_idx, max_support_len)
            support['raw'] = self.data['raw'][support_idx]

            done_queue.put(support)

    def __del__(self):
        '''
            Need to terminate the processes when deleting the object
        '''
        for i in range(self.num_cores):
            self.p_list[i].terminate()

        del self.done_queue


class ParallelSampler():
    def __init__(self, data, args, num_episodes=None):
        self.data = data
        self.args = args
        self.num_episodes = num_episodes

        self.all_classes = np.unique(self.data['label'])
        self.num_classes = len(self.all_classes)
        if self.num_classes < self.args.way:
            raise ValueError("Total number of classes is less than #way.")

        self.idx_list = []
        for y in self.all_classes:
            self.idx_list.append(np.squeeze(np.argwhere(self.data['label'] == y)))

        if self.args.classifier == "mn" or self.args.classifier == "contrastnet":
            self.augment = utils.Stop_Augment(self.args)

        self.count = 0
        self.done_queue = Queue()

        # self.num_cores = cpu_count() if args.n_workers is 0 else args.n_workers
        self.num_cores = 1

        self.p_list = []
        for i in range(self.num_cores):
            self.p_list.append(Process(target=self.worker, args=(self.done_queue,)))

        for i in range(self.num_cores):
            self.p_list[i].start()

    def get_epoch(self):
        for _ in range(self.num_episodes):
            # wait until self.thread finishes
            support, query = self.done_queue.get()

            # convert to torch.tensor
            support = utils.to_tensor(support, self.args.cuda, ['raw'])
            query = utils.to_tensor(query, self.args.cuda, ['raw'])

            support['is_support'] = True
            query['is_support'] = False

            yield support, query

    def worker(self, done_queue):
        '''
            Generate one task (support and query).
            Store into self.support[self.cur] and self.query[self.cur]
        '''
        while True:
            if done_queue.qsize() > 100:
                time.sleep(1)
                continue
            # sample ways
            sampled_classes = np.random.permutation(self.num_classes)[:self.args.way]

            source_classes = []
            for j in range(self.num_classes):
                if j not in sampled_classes:
                    source_classes.append(self.all_classes[j])
            source_classes = sorted(source_classes)

            # sample examples
            support_idx, query_idx = [], []
            for y in sampled_classes:
                tmp = np.random.permutation(len(self.idx_list[y]))
                support_idx.append(
                        self.idx_list[y][tmp[:self.args.shot]])
                query_idx.append(
                        self.idx_list[y][
                            tmp[self.args.shot:self.args.shot+self.args.query]])

            support_idx = np.concatenate(support_idx)
            query_idx = np.concatenate(query_idx)
            if self.args.mode == 'finetune' and len(query_idx) == 0:
                query_idx = support_idx

            # aggregate examples
            max_support_len = np.max(self.data['text_len'][support_idx])
            max_query_len = np.max(self.data['text_len'][query_idx])

            support = utils.select_subset(self.data, {}, ['text', 'text_len', 'attn_mask', 'label'],
                                          support_idx, max_support_len)
            query = utils.select_subset(self.data, {}, ['text', 'text_len', 'attn_mask', 'label'],
                                        query_idx, max_query_len)

            if self.args.classifier == "mn" or self.args.classifier == "contrastnet":
                data_s = self.data['raw'][support_idx]
                data_q = self.data['raw'][query_idx]

                aug_data_s = self.augment.data_augment(data_s)
                aug_data_q = self.augment.data_augment(data_q)

                aug_ids_s = self.augment.token_to_idx(aug_data_s)
                aug_ids_q = self.augment.token_to_idx(aug_data_q)
                support['aug_text'], support['aug_mask'] = aug_ids_s['text'], aug_ids_s['attn_mask']
                query['aug_text'], query['aug_mask'] = aug_ids_q['text'], aug_ids_q['attn_mask']

            # support = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
            #                               support_idx, max_support_len)
            # query = utils.select_subset(self.data, {}, ['text', 'text_len', 'label'],
            #                             query_idx, max_query_len)

            if 'pos' in self.args.auxiliary:
                support = utils.select_subset(self.data, support, ['head', 'tail'], support_idx)
                query = utils.select_subset(self.data, query, ['head', 'tail'], query_idx)

            done_queue.put((support, query))

    def __del__(self):
        """
            Need to terminate the processes when deleting the object
        """
        for i in range(self.num_cores):
            self.p_list[i].terminate()

        del self.done_queue
