import math
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from d2l import torch as d2l
from matplotlib import pyplot as plt
def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def load_fashion_mnist():   
    # 读取数据集
    train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=transforms.ToTensor())
    test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=transforms.ToTensor())

    batch_size = 64
    train_iter = data.DataLoader(train,batch_size,shuffle=True,num_workers=0)
    test_iter = data.DataLoader(test,batch_size,shuffle=False,num_workers=0)
    print(next(iter(train_iter))[0].shape)
    return train_iter,test_iter


def try_gpu(i=0):
    if torch.cuda.device_count()>i:
        return torch.device(f'cuda:{i}')
    else:
         return torch.device('cpu')