# dropout
from func import *
import torch
from d2l import torch as d2l
from torch import nn

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X

    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout) # 除以(1-p)是为了让均值不变

#测试
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))


# 定义网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784,256)
        self.lin2 = nn.Linear(256,256)
        self.lin3 = nn.Linear(256,10)
        self.relu = nn.ReLU()
    def forward(self,x):
        y1 = self.relu(self.lin1(x.reshape(-1,784)))
        if (1):
            y1 = dropout_layer(y1,0.2)
        y2 = self.relu(self.lin2(y1))
        if (1):
            y2 = dropout_layer(y2,0.5)
        out = self.lin3(y2)
        return out
net = Net()

num_epochs, lr = 10, 0.5
loss = nn.CrossEntropyLoss()
train_iter, test_iter = load_fashion_mnist()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()




'''
    pytorch实现dropout
'''

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256,256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256,10))

def init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, 0 , 0.01)
net.apply(init)


num_epochs, lr = 10, 0.5
loss = nn.CrossEntropyLoss()
train_iter, test_iter = load_fashion_mnist()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()    
