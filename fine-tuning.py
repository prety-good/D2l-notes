# 微调
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
from torch import nn
from d2l import torch as d2l
from func import show_images
from matplotlib import pyplot as plt

# 数据增强
# 使用RGB通道的均值和标准差，以标准化每个通道
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])
test_augs = transforms.Compose([
    torchvision.transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    normalize])

# 热狗数据集 http://d2l-data.s3-accelerate.amazonaws.com/hotdog.zip
train_imgs = torchvision.datasets.ImageFolder(os.path.join('./data/hotdog', 'train'),transform=train_augs)
test_imgs = torchvision.datasets.ImageFolder(os.path.join('./data/hotdog', 'test'),transform=test_augs)
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
plt.show()

train_iter = data.DataLoader(train_imgs,batch_size=128,shuffle=True,num_workers=0)
test_iter = data.DataLoader(test_imgs,batch_size=128,shuffle=False,num_workers=0)
print(next(iter(train_iter))[0].shape)

# 使用微调
net = torchvision.models.resnet18(pretrained=True)
print(net)
print(net.fc)

net.fc = nn.Linear(net.fc.in_features, 2)
nn.init.xavier_uniform_(net.fc.weight)


lr , num_epochs = 5e-4, 10

loss = nn.CrossEntropyLoss()
params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
trainer = torch.optim.SGD([{'params': params_1x},
                            {'params': net.fc.parameters(), 'lr': lr * 10}],
                        lr=lr, weight_decay=0.001)

d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,[torch.device('cuda:0')])


# 不使用微调
net = torchvision.models.resnet18()
net.fc = nn.Linear(net.fc.in_features, 2)
trainer = torch.optim.SGD(net.parameters(),lr = 0.05 , weight_decay = 0.001)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,[torch.device('cuda:0')])


plt.show()