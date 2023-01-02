import torch
import os
import numpy as np
import random
import pdb
import copy
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data.dataset import Dataset


class DatasetProcessing_NUS81(Dataset):
    def __init__(self, root, transformations, use):
        self.root = root
        self.data_path = 'path to dataset'
        self.transforms = transformations

        if use == 'train':
            name_file = 'train.txt'
        elif use == 'database':
            name_file = 'database.txt'
        else:
            name_file = 'test.txt'
        
        with open(os.path.join(self.data_path, name_file), 'r') as fr:
            image_list = fr.readlines()
            images = [val.split()[0] for val in image_list]
            all_lab = [[int(la) for la in val.split()[1:]] for val in image_list]
        
        lab_pinds, lab_ninds = [], []
        for label in all_lab:
            lab_pind, lab_nind = [], []
            for i in range(len(label)):
                if label[i] == 1:
                    lab_pind.append(i)
                else:
                    lab_nind.append(i)
            lab_pinds.append(lab_pind)
            lab_ninds.append(lab_nind)
            
        print("data:{}, images:{}".format(name_file, len(images)))
        self.imgs = images
        self.all_lab = np.array(all_lab, dtype=np.float32)
        self.lab_pinds = lab_pinds
        self.lab_ninds = lab_ninds

        
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.imgs[index]))
        img = img.convert('RGB')
        target = torch.Tensor(self.all_lab[index])

        if isinstance(self.transforms, list):
            img1 = self.transforms[0](img)
            img2 = self.transforms[1](img)
            # self.img_show(img, img1)
            return img1, img2, target, index
        else:
            img = self.transforms(img)
            return img, target, index
    

    def img_show(self, img):
        _, fig = plt.subplots(1, 1)
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std*img + mean
        img = np.clip(img,0,1)
        fig.imshow(img)
        plt.show()


    def __len__(self):
        return len(self.imgs)




class DatasetProcessing_Imagenet(Dataset):
    def __init__(self, root, transformations, use):
        self.root = root
        self.data_path = 'path to dataset'
        self.transforms = transformations

        if use == 'train':
            name_file = 'train.txt'
        elif use == 'database':
            name_file = 'database.txt'
        else:
            name_file = 'my_test.txt'
        
        with open(os.path.join(self.data_path, name_file), 'r') as fr:
            image_list = fr.readlines()
            if len(image_list[0].split()) > 2:
                # images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
                images = [val.split()[0] for val in image_list]
                all_lab = [[int(la) for la in val.split()[1:]] for val in image_list]
            else:
                # images = [(val.split()[0], int(val.split()[1])) for val in image_list]
                images = [val.split()[0] for val in image_list]
                all_lab = [int(val.split()[1]) for val in image_list]
        print("data:{}, images:{}".format(name_file, len(images)))
        self.imgs = images
        self.all_lab = np.array(all_lab, dtype=np.float32)

        lab_pinds, lab_ninds = [], []
        for label in all_lab:
            lab_pind, lab_nind = [], []
            for i in range(len(label)):
                if label[i] == 1:
                    lab_pind.append(i)
                else:
                    lab_nind.append(i)
            lab_pinds.append(lab_pind)
            lab_ninds.append(lab_nind)
            
        self.imgs = images
        self.all_lab = np.array(all_lab, dtype=np.float32)
        self.lab_pinds = lab_pinds
        self.lab_ninds = lab_ninds

    
    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.all_lab[index]
        img = Image.open(os.path.join(self.root, path))
        img = img.convert('RGB')
        if isinstance(self.transforms, list):
            img1 = self.transforms[0](img)
            img2 = self.transforms[1](img)
            # self.img_show(img, img1)
            return img1, img2, target, index
        else:
            img = self.transforms(img)
            return img, target, index
    

    def img_show(self, img):
        _, fig = plt.subplots(1, 1)
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std*img + mean
        img = np.clip(img,0,1)
        fig.imshow(img)
        plt.show()


    def __len__(self):
        return len(self.imgs)



class DatasetProcessing_MSCOCO(Dataset):
    def __init__(self, root, transformations, use):
        self.root = root
        self.data_path = 'path to dataset'
        self.transforms = transformations

        if use == 'train':
            name_file = 'train.txt'
        elif use == 'database':
            name_file = 'database.txt'
        else:
            name_file = 'test.txt'
        
        with open(os.path.join(self.data_path, name_file), 'r') as fr:
            image_list = fr.readlines()
            images = [val.split()[0] for val in image_list]
            all_lab = [[int(la) for la in val.split()[1:]] for val in image_list]
        
        lab_pinds, lab_ninds = [], []
        for label in all_lab:
            lab_pind, lab_nind = [], []
            for i in range(len(label)):
                if label[i] == 1:
                    lab_pind.append(i)
                else:
                    lab_nind.append(i)
            lab_pinds.append(lab_pind)
            lab_ninds.append(lab_nind)
            
        print("data:{}, images:{}".format(name_file, len(images)))
        self.imgs = images
        self.all_lab = np.array(all_lab, dtype=np.float32)
        self.lab_pinds = lab_pinds
        self.lab_ninds = lab_ninds

    
    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.all_lab[index]
        img = Image.open(os.path.join(self.root, path))
        img = img.convert('RGB')
        if isinstance(self.transforms, list):
            img1 = self.transforms[0](img)
            img2 = self.transforms[1](img)
            # self.img_show(img, img1)
            return img1, img2, target, index
        else:
            img = self.transforms(img)
            return img, target, index
    

    def img_show(self, img, imgt):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        imgt = imgt.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        imgt = std*imgt + mean
        imgt = np.clip(imgt,0,1)
        plt.imshow(imgt)
        plt.show()


    def __len__(self):
        return len(self.imgs)