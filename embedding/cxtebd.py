import datetime
import os.path

import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
# from pytorch_transformers import BertModel
import dataset.stats as stats

class CXTEBD(nn.Module):
    """
        An embedding layer directly returns precomputed BERT
        embeddings.
    """
    def __init__(self, args, return_seq=False):
        """
            pretrained_model_name_or_path, cache_dir: check huggingface's codebase for details
            finetune_ebd: finetuning bert representation or not during
            meta-training
            return_seq: return a sequence of bert representations, or [cls]
        """
        super(CXTEBD, self).__init__()

        self.args = args
        self.return_seq = return_seq

        print("{}, Loading pretrainedModel {}".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), args.pretrained_model), flush=True)

        # self.model = BertModel.from_pretrained("../module/bert-base-uncased")
        self.model = AutoModel.from_pretrained(args.pretrained_model, ignore_mismatched_sizes=True)
        # self.unfreeze_layers = ['layer.11']
        # self.unfreeze_layers = ['layer.11', 'pooler.']
        # for name, param in self.model.named_parameters():
        #     param.requires_grad = False
        #     for ele in self.unfreeze_layers:
        #         if ele in name:
        #             param.requires_grad = True

        self.embedding_dim = self.model.config.hidden_size
        self.ebd_dim = self.model.config.hidden_size

    def get_bert(self, bert_id, mask, data):
        """
            Return the last layer of bert's representation
            @param: bert_id: batch_size * max_text_len+2
            @param: text_len: text_len

            @return: last_layer: batch_size * max_text_len
        """

        # need to use smaller batches
        out = self.model(input_ids=bert_id, attention_mask=mask)

        # return seq of bert ebd, dim: batch, text_len, ebd_dim
        # return last_layer[:,1:-1,:]

        if self.args.meta_w_target:
            # count length excluding <pad> and <unk>.
            is_zero = (torch.sum(torch.abs(out[0]), dim=2) > 1e-8).float()
            soft_len = torch.sum(is_zero, dim=1, keepdim=True)

            soft_len[soft_len < 1] = 1
            # # don't need to mask out the <pad> tokens, as the embeddings are zero
            ebd = torch.sum(out[0], dim=1)

            ebd = ebd / soft_len

            w = stats.get_w_target_rr(data, ebd, self.args.meta_w_target_lam)

            return out[0], w.detach()

        if self.return_seq:
            return out[0]
        else:
            return out[0][:, 0, :]

    def forward(self, data, weight=None, flag=None):

        with torch.no_grad():
            x = self.get_bert(data['text'], data['attn_mask'], data)
            if (self.args.classifier == "mn" or self.args.classifier == "contrastnet") and self.args.embedding == "ebd":
                x_aug = self.get_bert(data['aug_text'], data['aug_mask'], data)
                return x, x_aug
            return x

