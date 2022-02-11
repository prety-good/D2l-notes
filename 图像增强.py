import torch
import torchvision
from d2l import torch as d2l
from torch import nn
from torch.utils import data
from torchvision import transforms
from func import show_images
from matplotlib import pyplot as plt

# 左右翻转
transforms.RandomHorizontalFlip()
# 上下翻转
transforms.RandomVerticalFlip()
# 颜色    ：亮度、对比度、饱和度、色调
color_aug=transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# 随机裁剪
# scale：原图的大小    ratio：高宽比        ()：裁剪后图片统一调整的大小
shape_aug = transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# 常用的图像增强的组合方式
augs = transforms.Compose([transforms.RandomHorizontalFlip(), color_aug, shape_aug])


train_trans = transforms.Compose([transforms.RandomVerticalFlip(),transforms.ToTensor()])
test_trans = transforms.Compose([transforms.ToTensor()])


train = torchvision.datasets.FashionMNIST(root="data", train=True, transform=train_trans)
test = torchvision.datasets.FashionMNIST(root="data", train=False, transform=test_trans)

train_iter = data.DataLoader(train,batch_size=64,shuffle=True,num_workers=0)
test_iter = data.DataLoader(test,batch_size=64,shuffle=False,num_workers=0)
print(next(iter(train_iter))[0].shape)

show_images([train[i][0][0] for i in range(32)],4, 8, scale=0.8)
plt.show()

net = d2l.resnet18(10,1)

lr, num_epochs = 0.1, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, 'cuda:0')
d2l.plt.show()
