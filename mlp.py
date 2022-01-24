# 多层感知机的手动实现和pytorch实现
import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


# 读取数据集
train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=transforms.ToTensor())
test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=transforms.ToTensor())

batch_size = 64
train_iter = data.DataLoader(train,batch_size,shuffle=True,num_workers=0)
test_iter = data.DataLoader(test,batch_size,shuffle=False,num_workers=0)
print(next(iter(train_iter))[0].shape)

w1=nn.Parameter(torch.randn(784, 256, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(256, requires_grad=True))
w2 = nn.Parameter(torch.randn(256, 10, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(10, requires_grad=True))
params = [w1, b1, w2, b2]


# 定义激活函数
def relu(x):
    return torch.max(x, torch.zeros_like(x))

# 定义模型
def net(x):
    x=x.reshape(-1,784)
    t1=relu(torch.matmul(x,w1)+b1)
    return (torch.matmul(t1,w2)+b2)

loss = nn.CrossEntropyLoss()

num_epochs, lr = 10, 0.1
optim = torch.optim.SGD(params, lr=lr)
for epoch in range(num_epochs):
    for X,y in train_iter:
        y_hat=net(X)
        l = loss(y_hat,y)
        optim.zero_grad()
        l.sum().backward()
        optim.step()
    if(epoch%2==0):
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        print('第{}轮训练，准确率：{}！'.format(epoch+1,test_acc))

# 借助d2l包的函数
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, optim)
d2l.predict_ch3(net, test_iter)
d2l.plt.show()

'''
    pytorch实现
'''
import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# 读取数据集
train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=transforms.ToTensor())
test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=transforms.ToTensor())
batch_size = 64
train_iter = data.DataLoader(train,batch_size,shuffle=True,num_workers=0)
test_iter = data.DataLoader(test,batch_size,shuffle=False,num_workers=0)
print(next(iter(train_iter))[0].shape)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10)
)
def init(m):
    if type(m)==nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
net.apply(init) # 初始化参数

num_epochs , lr = 10, 0.1
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(),lr = lr)
for epoch in range(num_epochs):
    for X,y in train_iter:
        y_hat=net(X)
        l = loss(y_hat,y)
        optim.zero_grad()
        l.sum().backward()
        optim.step()
    if(epoch%2==0):
        test_acc = d2l.evaluate_accuracy(net, test_iter)
        print('第{}轮训练，准确率：{}！'.format(epoch+1,test_acc))

# 借助d2l
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, optim)
d2l.plt.show()

