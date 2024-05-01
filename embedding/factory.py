import torch
import datetime

from embedding.wordebd import WORDEBD
from embedding.cxtebd import CXTEBD

from embedding.avg import AVG
from embedding.cnn import CNN
from embedding.idf import IDF
from embedding.meta import META
from embedding.lstmatt import LSTMAtt


def get_embedding(vocab, args):
    print("{}, Building embedding: {}".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), args.embedding), flush=True)

    # check if loading pre-trained embeddings
    # ebd = WORDEBD(vocab, args)
    ebd = CXTEBD(args, return_seq=(args.embedding != 'ebd'))

    if args.embedding in ['meta', 'meta_mlp']:
        model = META(ebd, args)
    elif args.embedding == 'avg' and args.bert:
        model = AVG(ebd, args)  # using bert representation directly
    elif args.embedding == 'ebd' and args.bert:
        model = ebd  # using bert representation directly
    elif args.embedding == 'cnn' and args.bert:
        model = CNN(ebd, args)  # using bert representation directly
    elif args.embedding == 'lstmatt' and args.bert:
        model = LSTMAtt(ebd, args)  # using bert representation directly
    else:
        model = None

    print("{}, Building embedding".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)

    if args.snapshot != '':
        # load pretrained models
        print("{}, Loading pretrained embedding from {}".format(
            datetime.datetime.now().strftime('%0y/%0m/%0d %H:%M:%S'),
            args.snapshot + '.ebd'
            ))
        model.load_state_dict(torch.load(args.snapshot + '.ebd'))

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
