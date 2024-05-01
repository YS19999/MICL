import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from classifier.base import BASE


def dot_similarity(x1, x2):
    return torch.matmul(x1, x2.t())

class Contrast_loss(nn.Module):
    def __init__(self, args, tau=0.7):
        super(Contrast_loss, self).__init__()
        self.args = args
        self.tau = tau

    def similarity(self, x1, x2):
        # # Gaussian Kernel
        # M = euclidean_dist(x1, x2)
        # s = torch.exp(-M/self.tau)

        # dot product
        M = dot_similarity(x1, x2) / self.tau
        s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        return s

    def forward(self, batch_label, *x):
        X = torch.cat(x, 0)
        len_ = batch_label.size()[0]

        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)

        # computing masks for contrastive loss
        mask_i = 1. - torch.eye(len_).to(
            batch_label.device)  # 获得一个对角线全为0，其余全为1的矩阵，去除自己与自己  # sum over items in the numerator
        label_matrix = batch_label.unsqueeze(0).repeat(len_, 1)
        mask_j = (batch_label.unsqueeze(
            1) - label_matrix == 0).float() * mask_i  # sum over items in the denominator, 标签句子，属于相同类的为1（不包括自身），不同的为0，
        pos_num = torch.sum(mask_j, 1)

        # weighted NLL loss
        s_i = torch.clamp(torch.sum(s * mask_i, 1), min=1e-10)  # 控制最小值大于1e-10
        s_j = torch.clamp(s * mask_j, min=1e-10)
        log_p = torch.sum(-torch.log(s_j / s_i) * mask_j, 1) / pos_num
        loss = torch.mean(log_p)

        return loss


class MN(BASE):
    def __init__(self, ebd_dim, args):
        super(MN, self).__init__(args)
        self.args = args
        self.ebd_dim = ebd_dim

        self.instance_loss = Contrast_loss(args)
        self.task_loss = Contrast_loss(args)

        # for mutual information
        self.fc = nn.Sequential(
            nn.Linear(2 * self.ebd_dim, 2 * self.ebd_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.ebd_dim, 1)
        )

        if self.args.classifier == "mn" and self.args.embedding == "ebd":
            print("{}, Loading my model".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')))

        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def get_distance(self, XS, XQ):

        dot_product = XQ.mm(XS.t()) / 0.1

        return dot_product

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot


    def get_sorted(self, XS, YS, XQ, YQ, XS_aug, XQ_aug):

        sorted_YS, indices_YS = torch.sort(YS)
        sorted_YQ, indices_YQ = torch.sort(YQ)

        XS = XS[indices_YS]
        XQ = XQ[indices_YQ]

        if XS_aug is not None:
            XS_aug = XS_aug[indices_YS]
            XQ_aug = XQ_aug[indices_YQ]
            return XS, sorted_YS, XQ, sorted_YQ, XS_aug, XQ_aug

        return XS, sorted_YS, XQ, sorted_YQ, None, None

    def _compute_mean(self, XS):
        '''
            Compute the prototype for each class by averaging over the ebd.

            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size

            @return prototype: way x ebd_dim
        '''
        # sort YS to make sure classes of the same labels are clustered together

        mean_ = []
        for i in range(self.args.way):
            mean_.append(torch.mean(
                XS[i * self.args.shot:(i + 1) * self.args.shot], dim=0,
                keepdim=True))

        mean_ = torch.cat(mean_, dim=0)

        return mean_

    def get_miloss(self, x_aug, x):
        batch_size = x.size(0)
        tiled_x = torch.cat([x, x], dim=0)
        idx = torch.randperm(batch_size)

        shuffled_y = x_aug[idx]
        concat_y = torch.cat([x_aug, shuffled_y], dim=0)
        inputs = torch.cat([tiled_x, concat_y], dim=1)
        logits = self.layers(inputs)

        pred_xy = logits[:batch_size] # 联合概率分布
        pred_x_y = logits[batch_size:] # 边缘分布的乘积
        # max mi  下面是将互信息计算转换成了对偶KL散度计算损失
        loss = - torch.log2(torch.exp(torch.tensor(1))) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        return loss


    def forward(self, XS, YS, XQ, YQ, XS_aug=None, XQ_aug=None):

        YS, YQ = self.reidx_y(YS, YQ)
        XS, YS, XQ, YQ, XS_aug, XQ_aug = self.get_sorted(XS, YS, XQ, YQ, XS_aug, XQ_aug)

        if self.args.classifier == "mn" and self.args.embedding == "ebd":

            all_x = torch.cat([XS, XQ], dim=0)
            all_x_aug = torch.cat([XS_aug, XQ_aug], dim=0)
            inst_label = torch.arange(0, all_x.shape[0], dtype=torch.long, device=all_x.device)
            inst_label = torch.cat([inst_label, inst_label], dim=0)
            instance_loss = self.instance_loss(inst_label, all_x, all_x_aug)

            samples_mean = self._compute_mean(XS)
            samples_mean_aug = self._compute_mean(XS_aug)
            task_label = torch.arange(0, self.args.way, dtype=torch.long, device=samples_mean.device)
            task_label = torch.cat([task_label, task_label], dim=0)
            task_loss = self.task_loss(task_label, samples_mean, samples_mean_aug)

            similar = self.get_distance(XS, XQ)
            YS_onehot = self._label2onehot(YS)
            pred = similar.mm(YS_onehot.float())

            mi_loss1 = self.get_miloss(XS_aug, XS)
            mi_loss2 = self.get_miloss(XQ_aug, XQ)

            loss = F.cross_entropy(pred, YQ)

            acc = BASE.compute_acc(pred, YQ)

            loss += 0.7 * instance_loss + 0.7 * task_loss + mi_loss1 + mi_loss2

        else:

            similar = self.get_distance(XS, XQ)
            YS_onehot = self._label2onehot(YS)

            pred = similar.mm(YS_onehot.float())

            loss = F.cross_entropy(pred, YQ)

            acc = BASE.compute_acc(pred, YQ)

        return acc, loss


