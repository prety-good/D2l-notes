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

def load_fashion_mnist(batch_size = 64 , resize = None):   
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    
    train = torchvision.datasets.FashionMNIST(root="data", train=True, transform=trans)
    test = torchvision.datasets.FashionMNIST(root="data", train=False, transform=trans)
    
    train_iter = data.DataLoader(train,batch_size,shuffle=True,num_workers=0)
    test_iter = data.DataLoader(test,batch_size,shuffle=False,num_workers=0)
    print(next(iter(train_iter))[0].shape)
    return train_iter,test_iter


def try_gpu(i=0):
    if torch.cuda.device_count()>i:
        return torch.device(f'cuda:{i}')
    else:
         return torch.device('cpu')


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(np.transpose(img.numpy(),(1,2,0)))
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
