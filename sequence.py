# 序列模型
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data
import matplotlib.pyplot as plt

# 序列信息
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
# plt.show()

# 生成训练集
tau = 4
features = torch.zeros(T-tau,tau)
for i in range(tau):
    features[:,i] = x[i:T-tau+i]
label = x[tau:].reshape(-1,1)
print(features.shape,label.shape)

class Mydataset(data.Dataset):
    def __init__(self,x,y):
        super().__init__()
        self.x = x
        self.y = y 
    def __len__(self):
        return len(self.x)
    def __getitem__(self,index):
        return self.x[index],self.y[index]

n_train = 600
dataset = Mydataset(features[:n_train],label[:n_train])
train_iter = data.DataLoader(dataset,batch_size=128,shuffle=True,num_workers=0)
print(next(iter(train_iter))[0].shape)




# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

net = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 1))
net.apply(init_weights)

num_epochs,learning_rate = 5, 0.1

# 训练
loss = nn.MSELoss(reduction='none')
optim = torch.optim.Adam(net.parameters(),lr = learning_rate,weight_decay=5e-5)
for epoch in range(num_epochs):
    for X,y in train_iter:
        optim.zero_grad()
        l = loss(net(X),y)
        l.sum().backward()
        optim.step()
    print(f'epoch.{epoch},  loss:{l.sum()/l.numel():3f}')


# 在整个数据集上查看效果
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
plt.show()


# 多步预测
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
plt.show()



max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i+1）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i+1）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
plt.show()