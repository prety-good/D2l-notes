import torch
from torch import nn
from func import load_fashion_mnist
from d2l import torch as d2l

class Inception(nn.Module):
    def __init__(self,cin, c1, c2, c3, c4):
        super().__init__()
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(cin,c1,kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(cin,c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(cin,c3[0],kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2 = nn.Conv2d(cin,c4,kernel_size=1)
    def forward(self,X, show = False):
        p1 = torch.relu(self.p1_1(X))
        p2 = torch.relu(self.p2_2(torch.relu(self.p2_1(X))))
        p3 = torch.relu(self.p3_2(torch.relu(self.p3_1(X))))
        p4 = torch.relu(self.p4_2(torch.relu(self.p4_1(X))))
        if show:
            print(f'线路1输出：{p1.shape}\n线路2输出：{p2.shape}\n线路3输出：{p3.shape}\n线路4输出：{p4.shape}')
        return torch.cat((p1,p2,p3,p4),dim=1) # 通道合并

X = torch.randn(1,1,96,96)
net = Inception(1,6,(12,12),(3,3),3)
print('Inception块输出：',net(X,show=True).shape)


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

print('GoogLenet架构：',net)
for layer in net:
    X = layer(X)
    print("{0}outputshape:\t{1}".format(layer.__class__.__name__,X.shape))


batch_size = 128
train_iter,test_iter = load_fashion_mnist(batch_size , resize = 96)
num_epochs , lr = 10 , 0.1

# 借助d2l包训练并可视化结果
d2l.train_ch6(net,train_iter,test_iter,num_epochs,lr,"cuda:0")
d2l.plt.show()

def init_weight(m):
    if type(m)==nn.Conv2d or type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
net.cuda()
net.apply(init_weight)

loss = nn.CrossEntropyLoss()
optim =torch.optim.SGD(net.parameters(),lr)

print(f'training on {net[0][0].weight.device}')
for epoch in range(num_epochs):
    for X ,y in train_iter:
        optim.zero_grad()
        X ,y =X.cuda() , y.cuda()
        l = loss(net(X),y)
        l.sum().backward()
        optim.step()
print(d2l.evaluate_accuracy_gpu(net,train_iter,"cuda:0"),d2l.evaluate_accuracy_gpu(net,test_iter,"cuda:0"))
