import torch
from torch import nn
from func import load_fashion_mnist
from d2l import torch as d2l


class residual(nn.Module):
    def __init__(self, cin , cout, use1x1conv = False, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3 , padding = 1, stride=stride)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3 , padding = 1)
        if use1x1conv:
            self.conv3 = nn.Conv2d(cin, cout, kernel_size=1,  stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(cout)
        self.bn2 = nn.BatchNorm2d(cout)
    def forward(self,X):
        Y1 = self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(X)))))
        if self.conv3:
            Y = torch.relu(Y1 + self.conv3(X))
        else:
            Y = torch.relu(Y1+X)
        return Y

X = torch.randn(64,1,96,96)
print(residual(1,6)(X).shape,residual(1, 6, True, stride=2)(X).shape)


def get_resnetblocks(cin, cout, nums , is_first = False):
    block = []
    for i in range(nums):
        if i == 0 and not is_first:
            block.append(residual(cin,cout,use1x1conv=True,stride=2))
        else:
            block.append(residual(cout,cout))
    return block

# resnet
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*get_resnetblocks(64, 64, 2, is_first=True))
b3 = nn.Sequential(*get_resnetblocks(64, 128, 2))
b4 = nn.Sequential(*get_resnetblocks(128, 256, 2))
b5 = nn.Sequential(*get_resnetblocks(256, 512, 2))


net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

print('GoogLenet架构：',net)
for layer in net:
    X = layer(X)
    print("{0}outputshape:\t{1}".format(layer.__class__.__name__,X.shape))


lr, num_epochs = 0.1, 10
train_iter, test_iter = load_fashion_mnist(batch_size = 128, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, 'cuda:0')
d2l.plt.show()


# 训练
net = nn.Sequential(b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(), nn.Linear(512, 10))
def init_weight(m):
    if type(m)==nn.Conv2d or type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
net.cuda()
net.apply(init_weight)

loss = nn.CrossEntropyLoss()
optim =torch.optim.SGD(net.parameters(),lr)

print(f'training on {net[0][0].weight.device}')
for epoch in range(num_epochs):
    net.train()
    for X ,y in train_iter:
        optim.zero_grad()
        X ,y =X.cuda() , y.cuda()
        l = loss(net(X),y)
        l.sum().backward()
        optim.step()
print(d2l.evaluate_accuracy_gpu(net,train_iter,"cuda:0"),d2l.evaluate_accuracy_gpu(net,test_iter,"cuda:0"))






