
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import pdb
import pickle
import copy
import argparse
import time

from datetime import datetime
from torchvision import transforms
from Datasets.datasets import DatasetProcessing_Imagenet, DatasetProcessing_NUS81, DatasetProcessing_MSCOCO
from Modules.models import MulPQNet
from Modules.warmup_scheduler import GradualWarmupScheduler
from Modules.Losses import ArcfaceCombLoss
from Utils.utils import CalcMapk



def setup_seed(seed=8):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def GenerateCode(model, data_loader, num_data, n_class, args, datatype):
    Q_Code, S_Code = [], []
    Q_Code = np.zeros([num_data, args.d_model], dtype=np.float32)
    S_Code = np.zeros([num_data, args.d_model], dtype=np.float32)
    all_labels = np.zeros([num_data, n_class], dtype=np.float32)
    for i, data in enumerate(data_loader):
        try:
            imgs, _, labels, indexs = data
        except:
            imgs, labels, indexs = data
        imgs = imgs.cuda()
        x, hard_feats, soft_feats, _ = model(imgs)

        if datatype == 'query' and args.Simode == 'AQD': 
            Q_Code[indexs.numpy(), :] = x.cpu().data.numpy()
            S_Code[indexs.numpy(), :] = soft_feats.cpu().data.numpy()
        else:
            Q_Code[indexs.numpy(), :] = hard_feats.cpu().data.numpy()
            S_Code[indexs.numpy(), :] = soft_feats.cpu().data.numpy()
        all_labels[indexs.numpy(), :] = labels.data.numpy()
    return Q_Code, S_Code, all_labels


def pairwise_loss(distances, labels, alpha):
    similarity = (torch.mm(labels.data.float(), labels.data.float().t()) > 0).float()

    mask_positive = (similarity.data > 0).float()
    mask_negative = (similarity.data <= 0).float()
    exp_loss = torch.log(1+torch.exp(alpha*distances)) - alpha*similarity*distances
    
    #weight
    S1 = torch.sum(mask_positive)
    S0 = torch.sum(mask_negative)
    S = S0 + S1
    exp_loss[similarity.data > 0] = exp_loss[similarity.data > 0] * (S / S1)
    exp_loss[similarity.data <= 0] = exp_loss[similarity.data <= 0] * (S / S0)

    loss = torch.sum(exp_loss) / S

    return loss


def init_optimizer(model, args):
    optimizer = torch.optim.SGD([
        {'params': model.feature.parameters(), 'lr': args.multi_lr*args.lr, 'momentum':0.9, 'weight_decay':args.weight_decay},
        {'params': model.avgpool.parameters(), 'lr': args.multi_lr*args.lr, 'momentum':0.9, 'weight_decay':args.weight_decay},
        {'params': model.classifier.parameters(), 'lr': args.multi_lr*args.lr, 'momentum':0.9, 'weight_decay':args.weight_decay},
        {'params': model.fc1.parameters()},
        {'params': model.CodeBook.parameters()},
        {'params': model.fcR.parameters()},
    ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    return optimizer



def SimPreserve(args):
    test_transformations = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transformations = [transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomCrop(227),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) for _ in range(2)]


    if args.dataset == 'NUS81':
        root = 'data root'
        data_path = 'data path'
        train_set = DatasetProcessing_NUS81(root, train_transformations, use='train')
        database_set = DatasetProcessing_NUS81(root, test_transformations, use='database')
        query_set = DatasetProcessing_NUS81(root, test_transformations, use='test')
        n_class = 81
        topK = 5000
        lab_word2vec = np.loadtxt(os.path.join(data_path, 'nuswide_81_wordvec.txt'))
        lab_word2vec = torch.Tensor(lab_word2vec).cuda()
    elif args.dataset == 'Imagenet':
        root = 'data root'
        data_path = 'data path'
        train_set = DatasetProcessing_Imagenet(root, train_transformations, use='train')
        database_set = DatasetProcessing_Imagenet(root, test_transformations, use='database')
        query_set = DatasetProcessing_Imagenet(root, test_transformations, use='test')
        n_class = 100
        topK = 1000
        lab_word2vec = np.loadtxt(os.path.join(data_path, 'Imagenet_wordvec.txt'))
        lab_word2vec = torch.Tensor(lab_word2vec).cuda()
    elif args.dataset == 'MSCOCO':
        root = 'data root'
        data_path = 'data path'
        train_set = DatasetProcessing_MSCOCO(root, train_transformations, use='train')
        database_set = DatasetProcessing_MSCOCO(root, test_transformations, use='database')
        query_set = DatasetProcessing_MSCOCO(root, test_transformations, use='test')
        n_class = 80
        topK = 5000
        lab_word2vec = np.loadtxt(os.path.join(data_path, 'MSCOCO_word2vec.txt'))
        lab_word2vec = torch.Tensor(lab_word2vec).cuda()
    else:
        raise NameError("unknown dataset")

    num_train = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_sz,
                            shuffle=True, num_workers=args.num_workers)
    num_database = len(database_set)
    database_loader = torch.utils.data.DataLoader(database_set, batch_size=args.batch_sz,
                                shuffle=False, num_workers=args.num_workers)
    num_query = len(query_set)
    query_loader = torch.utils.data.DataLoader(query_set, batch_size=args.batch_sz,
                            shuffle=False, num_workers=args.num_workers)

    model = MulPQNet(args, n_class).cuda()
    print(model)

    optimizer = init_optimizer(model, args)

    if args.warmup:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup, after_scheduler=None)

    arcomb_loss = ArcfaceCombLoss(s=args.s, m=args.margin, easy_margin=False)
    
    final_map = dict()
    stackLevel = int(args.mbits / np.log2(args.subcenter_num))
    for j in range(stackLevel):
        final_map["{}bits".format(int((j+1)*args.mbits/stackLevel))] = 0

    start = time.time()
    for epoch in range(args.epochs):
        # training stage
        epoch_loss, L1_eploss, L2_eploss, L3_eploss = 0, 0, 0, 0
        model.train()
        for i, train_data in enumerate(train_loader):
            imgs1, imgs2, labels, indexs = train_data
            imgs1, imgs2, labels = imgs1.cuda(), imgs2.cuda(), labels.cuda()
            imgs = torch.cat((imgs1, imgs2), dim=0)
            N = len(imgs1)
            L1, Loss, distances = 0, 0, torch.zeros(N, N).cuda()
    
            x, hard_feats, soft_feats, recon_embs = model(imgs)
            x = x.reshape(len(imgs), stackLevel, -1)
            soft_feats = soft_feats.reshape(len(imgs), stackLevel, -1)
            hard_feats = hard_feats.reshape(len(imgs), stackLevel, -1)

            "quantization loss"
            # quan_loss = (F.normalize(x, p=2, dim=2) - soft_feats).square().sum() / len(imgs)
            quan_loss = (hard_feats - soft_feats).square().sum() / len(imgs)          

            "semantic loss"
            sem_loss = (arcomb_loss(recon_embs.narrow(0, 0, int(N)), train_set, labels, lab_word2vec, indexs) + \
                        arcomb_loss(recon_embs.narrow(0, int(N), int(N)), train_set, labels, lab_word2vec, indexs)) / 2
            
            "pair wise loss"
            for j in range(stackLevel):
                outputs1 = soft_feats[:, j, :].narrow(0, 0, int(N))
                outputs2 = soft_feats[:, j, :].narrow(0, int(N), int(N))

                distances = distances.detach()
                distances += torch.cosine_similarity(outputs1.unsqueeze(1), outputs2.unsqueeze(0), dim=2)
                similarity_loss = pairwise_loss(distances, labels, args.alpha)
                L1 += similarity_loss*(args.w**j)
            
            Loss = args.lam1*L1/stackLevel + args.lam2*sem_loss + args.lam3*quan_loss
                
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            epoch_loss += Loss.item()
            L1_eploss += L1.item() / stackLevel
            L2_eploss += sem_loss
            L3_eploss += quan_loss.item()

        print('[Train Phase][Epoch: %3d/%3d][lr:%3.5f/%3.5f][Loss: %3.5f, L1: %3.5f, L2: %3.5f, L3: %3.5f]' % (
            epoch+1, args.epochs, optimizer.state_dict()['param_groups'][0]['lr'], optimizer.state_dict()['param_groups'][-1]['lr'], 
            epoch_loss/len(train_loader), L1_eploss/len(train_loader), L2_eploss/len(train_loader), L3_eploss/len(train_loader)))

        if args.warmup:
            scheduler_warmup.step()

        # validation stage
        model.eval()
        with torch.no_grad():
            qmB, _, q_labels = GenerateCode(model, query_loader, num_query, n_class, args, 'query')
            tmB, _, t_labels = GenerateCode(model, train_loader, num_train, n_class, args, 'database')
            qmB, tmB = qmB.reshape(num_query, stackLevel, -1), tmB.reshape(num_train, stackLevel, -1)
            pre_smat = 0
            for j in range(stackLevel):
                qB, tB = qmB[:, j, :], tmB[:, j, :]
                map, pre_smat = CalcMapk(qB, tB, q_labels, t_labels, topK, pre_smat)
                if (j+1)%2:
                    print('[Validation Phase][Epoch: %3d/%3d][bit: %3d, map: %3.5f]' % (
                        epoch+1, args.epochs, int((j+1)*args.mbits/stackLevel), map))

            # test stage
            if (epoch+1) % args.test_period == 0:
                qmB, _, q_labels = GenerateCode(model, query_loader, num_query, n_class, args, 'query')
                dmB, _, d_labels = GenerateCode(model, database_loader, num_database, n_class, args, 'database')
                qmB, dmB = qmB.reshape(num_query, stackLevel, -1), dmB.reshape(num_database, stackLevel, -1)
                pre_smat = 0
                for j in range(stackLevel):
                    qB, dB = qmB[:, j, :], dmB[:, j, :]
                    map, pre_smat = CalcMapk(qB, dB, q_labels, d_labels, topK, pre_smat)
                    if (j+1)%2: 
                        print('[Test Phase][Epoch: %3d/%3d][bit: %3d, map: %3.5f]' % (
                            epoch+1, args.epochs, int((j+1)*args.mbits/stackLevel), map))
                        final_map['{}bits'.format(int((j+1)*args.mbits/stackLevel))] = map
    
    end = time.time()
    print("total times:{}s".format(end - start))


    file_path = 'results/{}_{}_'.format(args.script_method, args.dataset) + datetime.now().strftime("%y-%m-%d-%H-%M-%S")    
    result = {}
    result['map'] = final_map
    result['model_dict'] = model.cpu().state_dict()
    result['args'] = args

    return result, model, file_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser("SimPreserve")
    parser.add_argument('--script_method', type=str, default="PSPPQ")
    parser.add_argument('--dataset', type=str, required=True, help="choose dataset")
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--mbits', type=int, default=64)
    parser.add_argument('--subcenter_num', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=150, help="choose epoch")
    parser.add_argument('--batch_sz', type=int, default=50, help="choose batch size")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-3, help="choose lr")
    parser.add_argument('--multi_lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=8.0)
    parser.add_argument('--w', type=float, default=0.8)
    parser.add_argument('--Qgama', type=float, default=10)
    parser.add_argument('--lam1', type=float, default=1)
    parser.add_argument('--lam2', type=float, default=0.7)
    parser.add_argument('--lam3', type=float, default=0)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--margin', type=float, default=0.7)
    parser.add_argument('--s', type=float, default=3)
    parser.add_argument('--test_period', type=int, default=3, help="choose test period")
    parser.add_argument('--Simode', type=str, default='AQD')
    
    args = parser.parse_args()
    
    filename = 'record.pkl'
    setup_seed(args.seed)
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    result, model, file_path = SimPreserve(args)

    fp = open(os.path.join(file_path, filename), 'wb')
    pickle.dump(result, fp)
    fp.close()

    with open(os.path.join(file_path, 'record.txt'), 'a', encoding='utf-8') as ft:
        ft.writelines("parameter settings ---------------- \n")
        for k in args.__dict__:
            print(k + ": " + str(args.__dict__[k]))
            ft.writelines("{}:{} \n".format(k, str(args.__dict__[k])))
        
        ft.writelines("map results ---------------- \n")
        for k in result['map']:
            print(k + ": " + str(result['map'][k]))
            ft.writelines("{}:{} \n".format(k, str(result['map'][k])))

        print("model ---------------- \n")
        print(model, file=ft)
    print("save:{}".format(file_path))