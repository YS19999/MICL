import torch
from classifier.mn import MN
from classifier.contrastnet import ContrastNet

from dataset.utils import tprint


def get_classifier(ebd_dim, args):
    tprint("Building classifier: {}".format(args.classifier))

    model = MN(ebd_dim, args)
    if args.snapshot != '':
        # load pretrained models
        tprint("Loading pretrained classifier from {}".format(
            args.snapshot + '.clf'
            ))
        model.load_state_dict(torch.load(args.snapshot + '.clf'))

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
