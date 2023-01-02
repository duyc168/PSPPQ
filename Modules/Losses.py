import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import pdb

class ArcfaceCombLoss(nn.Module):
    def __init__(self, s=30.0, m=0.50, easy_margin=False):
        super(ArcfaceCombLoss, self).__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self, f, train_set, labels, lab_word2vec, indexs):
        cos_sim = torch.cosine_similarity(f.unsqueeze(1), lab_word2vec.unsqueeze(0), dim=2)  # cos(θ)
        sine = torch.sqrt((1.0 - torch.pow(cos_sim, 2)).clamp(0, 1))
        phi = cos_sim*self.cos_m - sine*self.sin_m  # cos(θ+m)
        if self.easy_margin:
            phi = torch.where(cos_sim > 0, phi, cos_sim)
        else:
            phi = torch.where(cos_sim > self.th, phi, cos_sim - self.mm)

        outputs = labels*phi + (1.0 - labels)*cos_sim
        outputs *= self.s

        loss = 0
        for i, ind in enumerate(indexs):
            label = torch.Tensor(train_set.lab_pinds[ind]).long().cuda()
            # one_hot = torch.zeros([len(label), cos_sim.size(1)], device='cuda')
            # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = outputs[i].view(1, outputs.size(1))
            output = output.expand(len(label), outputs.size(1))
            loss += self.CE_loss(output, label)/len(label)

        return loss/indexs.size(0)
