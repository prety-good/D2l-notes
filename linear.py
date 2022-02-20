# 线性回归的手动实现和pytorch实现
import random
import torch
from d2l import torch as d2l
# 生成n个带有噪声的数据
def make_data(w,b,n):
    X=torch.normal(0,1,(n,len(w)))
    y=torch.matmul(X,w)+b+torch.normal(0,0.01,[n])
    return X,y.reshape(-1,1)

w=torch.tensor([2.4,4,-3.5])
b=3
train_x,train_y=make_data(w,b,1000)
print(train_x.shape,train_y.shape)

# 查看散点图
d2l.set_figsize()
d2l.plt.scatter(train_x[:, 0].detach().numpy(), train_y.detach().numpy(), 1)
d2l.plt.show()

# 设定读取数据集的函数
def data_iter(batch_size, X, y):
    num = len(X)
    id = list(range(num))
    random.shuffle(id)
    for i in range(0, num, batch_size):
        batch_id = id[i : min(i+batch_size,num)]
        yield X[batch_id],y[batch_id]

# 看一下一个batch_size的形状
for X,y in data_iter(64,train_x,train_y):
    print(X.shape,y.shape)
    break


# 定义模型
def linear(x,w,b):
    return torch.matmul(x,w)+b

# 损失函数 均方损失
def loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2


# 小批量随机梯度下降
def SGD(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -=lr*param.grad / batch_size
            param.grad.zero_()

# 初始化模型的参数
w = torch.normal(0,0.01,(3,1),requires_grad= True)
b = torch.zeros(1,requires_grad=True)

net=linear
lr = 0.05
num_epochs = 5
batch_size = 32
for epoch in range(num_epochs):
    for X ,y in data_iter(batch_size,train_x,train_y):
        l=loss(net(X,w,b),y)
        l.sum().backward()
        SGD([w,b],lr,batch_size)
    with torch.no_grad():
        print("No.{}  loss {:f}".format(epoch+1,loss(net(train_x,w,b),train_y).mean()))




'''
    pytorch实现方法：
'''
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

w=torch.tensor([2.4,4,-3.5])
b=3
train_x,train_y=make_data(w,b,1000)
print(train_x.shape,train_y.shape)

# 数据集
batch_size = 32
lr = 0.03
num_epochs=5
dataset = data.TensorDataset(*(train_x,train_y))
data_iter = data.DataLoader(dataset,batch_size,shuffle=True)

# 网络结构
from torch import nn
net = nn.Sequential(
    nn.Linear(len(w),1)
)
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 损失函数
loss=nn.MSELoss()
optim = torch.optim.SGD(net.parameters(),lr)

for epoch in range(num_epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        optim.zero_grad()
        l.backward()
        optim.step()
    print("No.{}  loss {:f}".format(epoch+1, loss(net(train_x),train_y)))

