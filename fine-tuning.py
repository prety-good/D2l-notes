# 微调
import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from func import show_images
from matplotlib import pyplot as plt

# 热狗数据集 http://d2l-data.s3-accelerate.amazonaws.com/hotdog.zip
train_imgs = torchvision.datasets.ImageFolder(os.path.join('./data/hotdog', 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join('./data/hotdog', 'test'))
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
plt.show()

