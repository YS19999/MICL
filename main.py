import json
import os
import sys
import pickle
import signal
import argparse
import traceback

import torch
import numpy as np

import embedding.factory as ebd
import classifier.factory as clf
import dataset.loader as loader
import train.factory as train_utils

def parse_args():
    parser = argparse.ArgumentParser(
            description="Few Shot intent detection with mutual information and contrastive learning")

    # data configuration
    parser.add_argument("--data_path", type=str,
                        default="../data/clinic150.csv",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="clinic150",
                        help="name of the dataset. "
                        "Options: [20newsgroup, amazon, huffpost, "
                        "reuters, rcv1, fewrel]")
  
    parser.add_argument("--n_train_class", type=int, default=60,
                        help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=30,
                        help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=60,
                        help="number of meta-test classes")

    # load bert embeddings for sent-level datasets (optional)
    parser.add_argument("--n_workers", type=int, default=10,
                        help="Num. of cores used for loading data. Set this "
                        "to zero if you want to use all the cpus.")
    parser.add_argument("--pretrained_bert", default="bert-base-uncased", type=str,
                        help=("path to the pre-trained bert embeddings."))

    # task configuration
    parser.add_argument("--way", type=int, default=5,
                        help="#classes for each task")
    parser.add_argument("--shot", type=int, default=1,
                        help="#support examples for each class for each task")
    parser.add_argument("--query", type=int, default=20,
                        help="#query examples for each class for each task")

    # train/test configuration
    parser.add_argument("--train_epochs", type=int, default=50,
                        help="max num of training epochs")
    parser.add_argument("--train_episodes", type=int, default=100,
                        help="#tasks sampled during each training epoch")
    parser.add_argument("--val_episodes", type=int, default=100,
                        help="#asks sampled during each validation epoch")
    parser.add_argument("--test_episodes", type=int, default=1000,
                        help="#tasks sampled during each testing epoch")

    # nn configuration
    parser.add_argument("--nn_distance", type=str, default="l2",
                        help=("distance for nearest neighbour. "
                              "Options: l2, cos [default: l2]"))

    # proto configuration
    parser.add_argument("--proto_hidden", nargs="+", type=int,
                        default=[300, 300],
                        help=("hidden dimension of the proto-net"))

    # training options
    parser.add_argument("--seed", type=int, default=123, help="seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="drop rate")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--patience", type=int, default=20, help="patience")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--cuda", type=int, default=0,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--mode", type=str, default="train",
                        help=("Running mode."
                              "Options: [train, test, finetune]"
                              "[Default: test]"))
    parser.add_argument("--save", action="store_true", default=False,
                        help="train the model")
    parser.add_argument("--notqdm", action="store_true", default=False,
                        help="disable tqdm")
    parser.add_argument("--result_path", type=str, default="results.json")
    parser.add_argument("--snapshot", type=str, default="",
                        help="path to the pretraiend weights")

    return parser.parse_args()


def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def main(args):
  
    # load data
    train_data, val_data, test_data, vocab = loader.load_dataset(args)

    # initialize model
    model = {}
    model["ebd"] = ebd.get_embedding(vocab, args)
    model["clf"] = clf.get_classifier(model["ebd"].ebd_dim, args)

    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        train_utils.train(train_data, val_data, model, args)

    test_acc, test_std = train_utils.test(test_data, model, args, args.test_episodes)

    if args.result_path:
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdir(directory)

        result = {
            "test_acc": test_acc,
            "test_std": test_std,
        }

        name = ['classifier', 'embedding', 'way', 'shot', 'query', 'data_path', 'dataset', 'lr', 'mode', 'pretrain_name']
        for attr, value in sorted(args.__dict__.items()):
            if attr in name:
                result[attr] = value

        with open(args.result_path, "a", encoding='UTF-8') as f:
            result = json.dumps(result, ensure_ascii=False)
            f.writelines(result + '\n')


if __name__ == "__main__":
    args = parse_args()
    main(args)
