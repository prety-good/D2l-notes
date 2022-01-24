# softmax回归的手动实现和pytorch实现
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

# 读取数据集
train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=transforms.ToTensor())
test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=transforms.ToTensor())

batch_size = 64
train_iter = data.DataLoader(train,batch_size,shuffle=True,num_workers=0)
test_iter = data.DataLoader(test,batch_size,shuffle=False,num_workers=0)
print(next(iter(train_iter))[0].shape)

# 初始化参数
w = torch.normal(0,0.1,size = (28*28,10),requires_grad=True)
b = torch.zeros(10,requires_grad=True)

# 定义softmax层
def softmax(x):
    # 一行为一组数据 每行的和为1
    x_exp = torch.exp(x)
    row_sum = x_exp.sum(1,keepdim=True)
    return x_exp/row_sum

# 网络结构
def net(x):
    linear = torch.matmul(x.reshape(-1, w.shape[0]), w)+b
    out = softmax(linear)
    return out

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

# 计算预测正确的数量
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

lr =0.1
def SGD(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -=lr*param.grad / batch_size
            param.grad.zero_()


def train(net,train_X,loss,optim,epochs):
    for epoch in range(epochs):
        for X,y in train_X:
            y_hat = net(X)
            l = loss(y_hat,y)
            l.sum().backward()
            optim([w,b],lr,batch_size)

train(net,train_iter,cross_entropy,SGD,5)



'''
  pytorch实现
'''
import torch
from torch import nn
from d2l import torch as d2l


# 读取数据集
train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=transforms.ToTensor())
test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=transforms.ToTensor())

batch_size = 64
train_iter = data.DataLoader(train,batch_size,shuffle=True,num_workers=0)
test_iter = data.DataLoader(test,batch_size,shuffle=False,num_workers=0)

# PyTorch不会隐式地调整输入的形状。因此，我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(
    nn.Flatten(), 
    nn.Linear(784, 10)
    )

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

# softmax和交叉熵结合到了一起
# 在交叉熵损失函数中传递未规范化的预测，并同时计算softmax及其对数
loss = nn.CrossEntropyLoss()

optim = torch.optim.SGD(net.parameters(),lr=0.1)

num_epochs = 10

# 利用d2l封装的 可以显示图像 准确率
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, optim)
d2l.plt.show()


for epoch in range(num_epochs):
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        optim.zero_grad()
        l.sum().backward()
        optim.step()
