import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import math
import pdb
import copy
from torchvision import models



class MulPQNet(nn.Module):
    def __init__(self, args, n_class):
        super(MulPQNet, self).__init__()
        backbone = models.alexnet(pretrained=True)
        self.feature = backbone.features
        cl1 = nn.Linear(256*6*6, 4096)
        cl1.weight = backbone.classifier[1].weight
        cl1.bias = backbone.classifier[1].bias
        cl2 = nn.Linear(4096, 4096)
        cl2.weight = backbone.classifier[4].weight
        cl2.bias = backbone.classifier[4].bias
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2, 
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(4096, args.d_model)

        self.stackLevel = int(args.mbits / np.log2(args.subcenter_num))
        assert(args.d_model % self.stackLevel == 0)
        self.sub_dim = int(args.d_model/self.stackLevel)

        fcR = nn.Sequential(
                nn.Linear(args.d_model, 300, bias=False), 
                nn.LeakyReLU(inplace=True),
                )
        fcL = nn.Linear(args.d_model, n_class)
        
        self.CodeBook = nn.ModuleList([nn.Embedding(args.subcenter_num, self.sub_dim) for i in range(self.stackLevel)])
        for i in range(self.stackLevel):
            nn.init.xavier_normal_(self.CodeBook[i].weight, gain=1)
        self.gama = args.Qgama
        self.fcR = fcR
        
    
    def forward(self, x):
        # HardCodes = []
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        x = self.fc1(x)

        x = x.reshape(x.size(0), self.stackLevel, -1)     
        soft_feats = torch.zeros((x.size(0), 0)).cuda()
        hard_feats = torch.zeros((x.size(0), 0)).cuda()

        for level in range(self.stackLevel):
            C = self.CodeBook[level]
            distance = torch.cosine_similarity(x[:, level, :].unsqueeze(1), C.weight.unsqueeze(0), dim=2)
            soft = torch.matmul(F.softmax(self.gama*distance, dim=1), C.weight)
            soft = F.normalize(soft, p=2, dim=1)
            hardcode = torch.argmax(distance, axis=1)
            # HardCodes.append(hardcode)
            hard = C.weight[hardcode]
            hard = F.normalize(hard, p=2, dim=1)

            soft_feats = torch.cat([soft_feats, soft], dim=1)
            hard_feats = torch.cat([hard_feats, hard], dim=1)

        recon_embs = self.fcR(soft_feats)
        
        return x.reshape(x.size(0), -1), hard_feats, soft_feats, recon_embs